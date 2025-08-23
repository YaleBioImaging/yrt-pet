/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */


#include "yrt-pet/recon/Corrector_GPU.cuh"

#include "yrt-pet/operators/OperatorProjectorDD_GPU.cuh"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

namespace yrt
{
Corrector_GPU::Corrector_GPU(const Scanner& pr_scanner)
    : Corrector{pr_scanner}, mph_lastCopiedHostImage{nullptr}
{
}


void Corrector_GPU::precomputeAdditiveCorrectionFactors(
    const ProjectionData& measurements)
{
	ASSERT_MSG(hasAdditiveCorrection(measurements),
	           "No additive corrections needed");

	auto additiveCorrections =
	    std::make_unique<ProjectionListOwned>(&measurements);
	additiveCorrections->allocate();

	mph_additiveCorrections = std::move(additiveCorrections);

	const bin_t numBins = measurements.count();
	const bool histogrammedACFs = doesTotalACFComeFromHistogram();

	if (!histogrammedACFs && mp_attenuationImage != nullptr)
	{
		std::cout << "Forward projecting attenuation image to prepare for "
		             "additive corrections..."
		          << std::endl;

		util::forwProject(measurements.getScanner(), *mp_attenuationImage,
		                  *mph_additiveCorrections, OperatorProjector::SIDDON,
		                  true);

		// TODO: This part would be faster if done on GPU (inside the projection
		//  kernel)
		std::cout << "Computing attenuation coefficient factors..."
		          << std::endl;
		util::convertProjectionValuesToACF(*mph_additiveCorrections);
	}

	float* additiveCorrectionsPtr = mph_additiveCorrections->getRawPointer();

	std::cout << "Precomputing additive factors using provided histograms..."
	          << std::endl;

	const ProjectionData* measurementsPtr = &measurements;

	util::parallelForChunked(
	    numBins, globals::getNumThreads(),
	    [additiveCorrectionsPtr, measurementsPtr, histogrammedACFs, numBins,
	     this](bin_t bin, size_t /*tid*/)
	    {
		    const histo_bin_t histoBin = measurementsPtr->getHistogramBin(bin);

		    const float randomsEstimate =
		        getRandomsEstimate(*measurementsPtr, bin, histoBin);

		    const float scatterEstimate = getScatterEstimate(histoBin);

		    const float sensitivity = getSensitivity(histoBin);

		    float acf = 1.0f;

		    if (histogrammedACFs)
		    {
			    // ACF was not precomputed in the additive corrections buffer
			    acf = getTotalACFFromHistogram(histoBin);
		    }
		    else if (mp_attenuationImage != nullptr)
		    {
			    // ACFs were precomputed in the additive corrections buffer
			    acf = additiveCorrectionsPtr[bin];
		    }

		    if (acf > StabilityEpsilon && sensitivity > StabilityEpsilon)
		    {
			    additiveCorrectionsPtr[bin] =
			        (randomsEstimate + scatterEstimate) / (sensitivity * acf);
		    }
		    else
		    {
			    additiveCorrectionsPtr[bin] = 0.0f;
		    }
	    });
}

void Corrector_GPU::precomputeInVivoAttenuationFactors(
    const ProjectionData& measurements)
{
	ASSERT(hasInVivoAttenuation());

	auto inVivoAttenuationFactors =
	    std::make_unique<ProjectionListOwned>(&measurements);
	inVivoAttenuationFactors->allocate();

	mph_inVivoAttenuationFactors = std::move(inVivoAttenuationFactors);

	if (doesInVivoACFComeFromHistogram())
	{
		ASSERT(mp_inVivoAcf != nullptr);

		const bin_t numBins = measurements.count();
		float* inVivoAttenuationFactorsPtr =
		    mph_inVivoAttenuationFactors->getRawPointer();

		std::cout << "Gathering in-vivo ACFs to prepare for precorrection..."
		          << std::endl;

		util::parallelForChunked(
		    numBins, globals::getNumThreads(),
		    [numBins, &measurements, inVivoAttenuationFactorsPtr,
		     this](bin_t bin, size_t /*tid*/)
		    {
			    const histo_bin_t histoBin = measurements.getHistogramBin(bin);
			    inVivoAttenuationFactorsPtr[bin] =
			        mp_inVivoAcf->getProjectionValueFromHistogramBin(histoBin);
		    });
	}
	else if (mp_inVivoAttenuationImage != nullptr)
	{
		std::cout << "Forward-projecting in-vivo attenuation image to prepare "
		             "for precorrection..."
		          << std::endl;
		util::forwProject(measurements.getScanner(), *mp_inVivoAttenuationImage,
		                  *mph_inVivoAttenuationFactors,
		                  OperatorProjector::SIDDON, true);

		// TODO: This part would be faster if done on GPU (inside the projection
		//  kernel)
		std::cout << "Computing attenuation coefficient factors..."
		          << std::endl;
		util::convertProjectionValuesToACF(*mph_inVivoAttenuationFactors);
	}
	else
	{
		// Not supposed to reach here
		ASSERT_MSG(false, "Unexpected error");
	}
}

void Corrector_GPU::loadAdditiveCorrectionFactorsToTemporaryDeviceBuffer(
    GPULaunchConfig launchConfig)
{
	loadPrecomputedCorrectionFactorsToTemporaryDeviceBuffer(
	    mph_additiveCorrections.get(), launchConfig);
}

void Corrector_GPU::loadInVivoAttenuationFactorsToTemporaryDeviceBuffer(
    GPULaunchConfig launchConfig)
{
	loadPrecomputedCorrectionFactorsToTemporaryDeviceBuffer(
	    mph_inVivoAttenuationFactors.get(), launchConfig);
}

const ProjectionDataDevice* Corrector_GPU::getTemporaryDeviceBuffer() const
{
	return mpd_temporaryCorrectionFactors.get();
}

ProjectionDataDevice* Corrector_GPU::getTemporaryDeviceBuffer()
{
	return mpd_temporaryCorrectionFactors.get();
}

void Corrector_GPU::applyHardwareAttenuationToGivenDeviceBufferFromACFHistogram(
    ProjectionDataDevice* destProjData, GPULaunchConfig launchConfig)
{
	ASSERT_MSG(mpd_temporaryCorrectionFactors != nullptr,
	           "Need to initialize temporary correction factors first");
	ASSERT(hasHardwareAttenuation());
	ASSERT(mp_hardwareAcf != nullptr);
	ASSERT(mpd_temporaryCorrectionFactors->getPrecomputedBatchSize() > 0);

	mpd_temporaryCorrectionFactors->allocateForProjValues(
	    {launchConfig.stream, false});

	mpd_temporaryCorrectionFactors->loadProjValuesFromHostHistogram(
	    mp_hardwareAcf, {launchConfig.stream, false});
	destProjData->multiplyProjValues(mpd_temporaryCorrectionFactors.get(),
	                                 launchConfig);
}

void Corrector_GPU::
    applyHardwareAttenuationToGivenDeviceBufferFromAttenuationImage(
        ProjectionDataDevice* destProjData, OperatorProjectorDevice* projector,
        GPULaunchConfig launchConfig)
{
	ASSERT_MSG(mpd_temporaryCorrectionFactors != nullptr,
	           "Need to initialize temporary correction factors first");
	ASSERT(hasHardwareAttenuation());
	ASSERT(mp_hardwareAttenuationImage != nullptr);
	ASSERT(projector != nullptr);

	mpd_temporaryCorrectionFactors->allocateForProjValues(
	    {launchConfig.stream, false});
	initializeTemporaryDeviceImageIfNeeded(mp_hardwareAttenuationImage,
	                                       {launchConfig.stream, false});

	// TODO: Design-wise, it would be better to call a static function here
	//  instead of relying on a projector given as argument
	projector->applyA(mpd_temporaryImage.get(),
	                  mpd_temporaryCorrectionFactors.get(), false);
	mpd_temporaryCorrectionFactors->convertToACFsDevice(
	    {launchConfig.stream, false});
	destProjData->multiplyProjValues(mpd_temporaryCorrectionFactors.get(),
	                                 launchConfig);
}

void Corrector_GPU::loadPrecomputedCorrectionFactorsToTemporaryDeviceBuffer(
    const ProjectionList* factors, GPULaunchConfig launchConfig)
{
	ASSERT_MSG(mpd_temporaryCorrectionFactors != nullptr,
	           "Need to initialize temporary correction factors first");

	// Will only allocate if necessary
	mpd_temporaryCorrectionFactors->allocateForProjValues(
	    {launchConfig.stream, false});

	mpd_temporaryCorrectionFactors->loadProjValuesFromHost(factors,
	                                                       launchConfig);
}

void Corrector_GPU::initializeTemporaryDeviceImageIfNeeded(
    const Image* hostReference, GPULaunchConfig launchConfig)
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
			mpd_temporaryImage = std::make_unique<ImageDeviceOwned>(
			    referenceParams, launchConfig.stream);
		}
		if (!mpd_temporaryImage->isMemoryValid())
		{
			mpd_temporaryImage->allocate();
		}
		mpd_temporaryImage->copyFromHostImage(hostReference,
		                                      launchConfig.synchronize);

		mph_lastCopiedHostImage = hostReference;
	}
}

void Corrector_GPU::initializeTemporaryDeviceBuffer(
    const ProjectionDataDevice* master)
{
	ASSERT(master != nullptr);
	mpd_temporaryCorrectionFactors =
	    std::make_unique<ProjectionDataDeviceOwned>(master);
}

void Corrector_GPU::clearTemporaryDeviceBuffer()
{
	mpd_temporaryCorrectionFactors = nullptr;
}

}  // namespace yrt
