/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */


#include "recon/Corrector_GPU.cuh"

#include "operators/OperatorProjectorDD_GPU.cuh"
#include "utils/ReconstructionUtils.hpp"

Corrector_GPU::Corrector_GPU() : mph_lastCopiedHostImage{nullptr} {}


void Corrector_GPU::precomputeAdditiveCorrectionFactors(
    const ProjectionData* measurements)
{
	ASSERT(measurements != nullptr);
	ASSERT_MSG(hasAdditiveCorrection(), "No additive corrections needed");

	auto additiveCorrections =
	    std::make_unique<ProjectionListOwned>(measurements);
	additiveCorrections->allocate();

	mph_additiveCorrections = std::move(additiveCorrections);

	const bin_t numBins = measurements->count();
	const bool histogrammedACFs = doesTotalACFComeFromHistogram();

	if (!histogrammedACFs && mp_attenuationImage != nullptr)
	{
		std::cout << "Forward projecting attenuation image to prepare for "
		             "additive corrections..."
		          << std::endl;

		// TODO: Once GPU Siddon is implemented, use it here instead of DD_GPU
		Util::forwProject(measurements->getScanner(), *mp_attenuationImage,
		                  *mph_additiveCorrections, OperatorProjector::DD_GPU);

		// TODO: This part would be faster if done on GPU (inside the projection
		//  kernel)
		std::cout << "Computing attenuation coefficient factors..."
		          << std::endl;
		Util::convertProjectionValuesToACF(*mph_inVivoAttenuationFactors);
	}

	float* additiveCorrectionsPtr = mph_additiveCorrections->getRawPointer();

	std::cout << "Precomputing additive factors using provided histograms..."
	          << std::endl;

	// TODO NOW: Add parallel
	for (bin_t bin = 0; bin < numBins; bin++)
	{
		const histo_bin_t histoBin = measurements->getHistogramBin(bin);

		const float randomsEstimate =
		    getRandomsEstimate(measurements, bin, histoBin);

		const float scatterEstimate = getScatterEstimate(histoBin);

		const float sensitivity = getSensitivity(histoBin);

		if (histogrammedACFs)
		{
			// ACF was not precomputed in the additive corrections buffer
			const float acf = getTotalACFFromHistogram(histoBin);
			additiveCorrectionsPtr[bin] =
			    (randomsEstimate + scatterEstimate) / (sensitivity * acf);
		}
		else if (mp_attenuationImage != nullptr)
		{
			// ACFs were precomputed in the additive corrections buffer
			const float acf = additiveCorrectionsPtr[bin];
			additiveCorrectionsPtr[bin] =
			    (randomsEstimate + scatterEstimate) / (sensitivity * acf);
		}
		else
		{
			// There is no attenuation
			additiveCorrectionsPtr[bin] =
			    (randomsEstimate + scatterEstimate) / (sensitivity);
		}
	}
}

void Corrector_GPU::precomputeInVivoAttenuationFactors(
    const ProjectionData* measurements)
{
	ASSERT(measurements != nullptr);
	ASSERT(hasInVivoAttenuation());

	auto inVivoAttenuationFactors =
	    std::make_unique<ProjectionListOwned>(measurements);
	inVivoAttenuationFactors->allocate();

	mph_inVivoAttenuationFactors = std::move(inVivoAttenuationFactors);

	if (doesInVivoACFComeFromHistogram())
	{
		ASSERT(mp_inVivoAcf != nullptr);

		const bin_t numBins = measurements->count();
		float* inVivoAttenuationFactorsPtr =
		    mph_inVivoAttenuationFactors->getRawPointer();

#pragma omp parallel for default(none) \
    firstprivate(numBins, measurements, inVivoAttenuationFactorsPtr)
		for (bin_t bin = 0; bin < numBins; bin++)
		{
			const histo_bin_t histoBin = measurements->getHistogramBin(bin);
			inVivoAttenuationFactorsPtr[bin] =
			    mp_inVivoAcf->getProjectionValueFromHistogramBin(histoBin);
		}
	}
	else if (mp_inVivoAttenuationImage != nullptr)
	{
		// TODO: Use GPU Siddon once available
		Util::forwProject(
		    measurements->getScanner(), *mp_inVivoAttenuationImage,
		    *mph_inVivoAttenuationFactors, OperatorProjector::DD_GPU);

		// TODO: This part would be faster if done on GPU (inside the projection
		//  kernel)
		std::cout << "Computing attenuation coefficient factors..."
		          << std::endl;
		Util::convertProjectionValuesToACF(*mph_inVivoAttenuationFactors);
	}

	// Not supposed to reach here
	ASSERT_MSG(false, "Unexpected error");
}

void Corrector_GPU::loadAdditiveCorrectionFactorsToTemporaryDeviceBuffer(
    const cudaStream_t* stream)
{
	loadPrecomputedCorrectionFactorsToTemporaryDeviceBuffer(
	    mph_additiveCorrections.get(), stream);
}

void Corrector_GPU::loadInVivoAttenuationFactorsToTemporaryDeviceBuffer(
    const cudaStream_t* stream)
{
	loadPrecomputedCorrectionFactorsToTemporaryDeviceBuffer(
	    mph_inVivoAttenuationFactors.get(), stream);
}

const ProjectionDataDevice* Corrector_GPU::getTemporaryDeviceBuffer() const
{
	return mpd_temporaryCorrectionFactors.get();
}

ProjectionDataDevice* Corrector_GPU::getTemporaryDeviceBuffer()
{
	return mpd_temporaryCorrectionFactors.get();
}

void Corrector_GPU::applyHardwareAttenuationFactorsToGivenDeviceBuffer(
    ProjectionDataDevice* destProjData, OperatorProjectorDevice* projector,
    const cudaStream_t* stream)
{
	ASSERT_MSG(mpd_temporaryCorrectionFactors != nullptr,
	           "Need to initialize temporary correction factors first");
	ASSERT(hasHardwareAttenuation());

	mpd_temporaryCorrectionFactors->allocateForProjValues(stream);

	if (mp_hardwareAcf != nullptr)
	{
		mpd_temporaryCorrectionFactors->loadProjValuesFromHostHistogram(
		    mp_hardwareAcf, stream);
		destProjData->multiplyProjValues(mpd_temporaryCorrectionFactors.get(),
		                                 stream);
	}
	else if (mp_hardwareAttenuationImage != nullptr)
	{
		ASSERT(projector != nullptr);
		initializeTemporaryDeviceImageIfNeeded(mp_hardwareAttenuationImage,
		                                       stream);
		// TODO: Design-wise, it would be better to call a static function here
		//  instead of relying on a projector given as argument
		projector->applyA(mpd_temporaryImage.get(),
		                  mpd_temporaryCorrectionFactors.get());
		mpd_temporaryCorrectionFactors->convertToACFsDevice(stream);
		destProjData->multiplyProjValues(mpd_temporaryCorrectionFactors.get(),
		                                 stream);
	}
}

void Corrector_GPU::loadPrecomputedCorrectionFactorsToTemporaryDeviceBuffer(
    const ProjectionList* factors, const cudaStream_t* stream)
{
	ASSERT_MSG(mpd_temporaryCorrectionFactors != nullptr,
	           "Need to initialize temporary correction factors first");

	// Will only allocate if necessary
	mpd_temporaryCorrectionFactors->allocateForProjValues(stream);

	mpd_temporaryCorrectionFactors->loadProjValuesFromHost(factors, stream);
}

void Corrector_GPU::initializeTemporaryDeviceImageIfNeeded(
    const Image* hostReference, const cudaStream_t* stream)
{
	ASSERT_MSG(hostReference != nullptr, "Null host-side image");
	ASSERT(hostReference->isMemoryValid());

	if (mph_lastCopiedHostImage != hostReference ||
	    mpd_temporaryImage == nullptr)
	{
		const ImageParams& referenceParams = hostReference->getParams();
		if (mpd_temporaryImage == nullptr ||
		    !referenceParams.isSameAs(mpd_temporaryImage->getParams()))
		{
			mpd_temporaryImage =
			    std::make_unique<ImageDeviceOwned>(referenceParams, stream);
			mpd_temporaryImage->allocate();
		}
		mpd_temporaryImage->copyFromHostImage(hostReference);

		mph_lastCopiedHostImage = hostReference;
	}
}

void Corrector_GPU::initializeTemporaryDeviceBuffer(
    const ProjectionDataDevice* master)
{
	mpd_temporaryCorrectionFactors =
	    std::make_unique<ProjectionDataDeviceOwned>(master);
}
