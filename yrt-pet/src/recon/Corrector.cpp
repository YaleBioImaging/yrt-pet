/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/Corrector.hpp"

#include "yrt-pet/operators/ProjectorSiddon.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/Tools.hpp"

namespace yrt
{

Corrector::Corrector(const Scanner& pr_scanner)
    : mr_scanner(pr_scanner),
      mp_randoms(nullptr),
      mp_scatter(nullptr),
      mp_acf(nullptr),
      mp_attenuationImage(nullptr),
      mp_inVivoAcf(nullptr),
      mp_inVivoAttenuationImage(nullptr),
      mp_hardwareAcf(nullptr),
      mp_hardwareAttenuationImage(nullptr),
      m_attenuationSetupComplete(false),
      mp_impliedTotalAttenuationImage(nullptr),
      mp_sensitivity(nullptr),
      m_invertSensitivity(false),
      m_globalScalingFactor(1.0f)
{
}

void Corrector::setup()
{
	if (!m_attenuationSetupComplete)
	{
		if (mp_acf != nullptr && mp_inVivoAcf != nullptr &&
		    mp_hardwareAcf == nullptr)
		{
			std::cout
			    << "Warning: Total ACF and in-vivo ACF specified, but no "
			       "hardware ACF specified. Assuming no hardware attenuation..."
			    << std::endl;
		}
		else if (mp_acf != nullptr && mp_inVivoAcf == nullptr &&
		         mp_hardwareAcf != nullptr)
		{
			std::cout
			    << "Warning: Total ACF and hardware ACF specified, but no "
			       "in-vivo ACF specified. Assuming no in-vivo attenuation..."
			    << std::endl;
		}
		if (mp_acf == nullptr && mp_inVivoAcf == nullptr &&
		    mp_hardwareAcf != nullptr)
		{
			// All ACF is Hardware ACF
			mp_acf = mp_hardwareAcf;
		}
		else if (mp_acf == nullptr && mp_inVivoAcf != nullptr &&
		         mp_hardwareAcf == nullptr)
		{
			// All ACF is in-vivo ACF
			mp_acf = mp_inVivoAcf;
		}
		else if (mp_acf != nullptr && mp_inVivoAcf == nullptr &&
		         mp_hardwareAcf == nullptr)
		{
			// User only specified total ACF, but not how it is distributed.
			// We assume that all ACFs come from hardware, which is
			// the case when there is no motion
			mp_hardwareAcf = mp_acf;
		}

		// Adjust attenuation image logic

		if (mp_inVivoAttenuationImage != nullptr &&
		    mp_hardwareAttenuationImage != nullptr &&
		    mp_attenuationImage == nullptr)
		{
			// Here, the hardware and in-vivo attenuation images were specified,
			// but the total attenuation image wasn't. The total attenuation
			// image should be the sum of the in-vivo and the hardware
			const ImageParams attParams =
			    mp_hardwareAttenuationImage->getParams();
			ASSERT_MSG(
			    attParams.isSameAs(mp_inVivoAttenuationImage->getParams()),
			    "Parameters mismatch between attenuation images");
			mp_impliedTotalAttenuationImage =
			    std::make_unique<ImageOwned>(attParams);
			mp_impliedTotalAttenuationImage->allocate();
			const float* hardwareAttenuationImage_ptr =
			    mp_hardwareAttenuationImage->getRawPointer();
			const float* inVivoAttenuationImage_ptr =
			    mp_inVivoAttenuationImage->getRawPointer();
			mp_impliedTotalAttenuationImage->operationOnEachVoxelParallel(
			    [hardwareAttenuationImage_ptr,
			     inVivoAttenuationImage_ptr](size_t voxelIndex)
			    {
				    return hardwareAttenuationImage_ptr[voxelIndex] +
				           inVivoAttenuationImage_ptr[voxelIndex];
			    });
			mp_attenuationImage = mp_impliedTotalAttenuationImage.get();
		}
		else if (mp_inVivoAttenuationImage == nullptr &&
		         mp_hardwareAttenuationImage != nullptr &&
		         mp_attenuationImage == nullptr)
		{
			// All attenuation is hardware attenuation
			mp_attenuationImage = mp_hardwareAttenuationImage;
		}
		else if (mp_inVivoAttenuationImage != nullptr &&
		         mp_hardwareAttenuationImage == nullptr &&
		         mp_attenuationImage == nullptr)
		{
			// All attenuation is in-vivo attenuation
			mp_attenuationImage = mp_inVivoAttenuationImage;
		}
		else if (mp_inVivoAttenuationImage == nullptr &&
		         mp_hardwareAttenuationImage == nullptr &&
		         mp_attenuationImage != nullptr)
		{
			// User only specified total attenuation, but not how it is
			// distributed. We assume that all the attenuation comes from
			// hardware, which is the case when there is no motion
			mp_hardwareAttenuationImage = mp_attenuationImage;
		}
		else if (mp_inVivoAttenuationImage != nullptr &&
		         mp_hardwareAttenuationImage == nullptr &&
		         mp_attenuationImage != nullptr)
		{
			std::cout
			    << "Warning: Hardware attenuation image not specified while "
			       "full attenuation and in-vivo attenuation is specified. "
			       "It will be assumed that there is no hardware attenuation."
			    << std::endl;
		}
		else if (mp_inVivoAttenuationImage == nullptr &&
		         mp_hardwareAttenuationImage != nullptr &&
		         mp_attenuationImage != nullptr)
		{
			std::cout
			    << "Warning: In-vivo attenuation image not specified while "
			       "full attenuation and hardware attenuation is specified. "
			       "It will be assumed that there is no in-vivo attenuation."
			    << std::endl;
		}
		m_attenuationSetupComplete = true;
	}

	// In case we need to backproject a uniform histogram:
	if (mp_hardwareAcf == nullptr && mp_sensitivity == nullptr &&
	    mp_uniformHistogram == nullptr)
	{
		mp_uniformHistogram = std::make_unique<UniformHistogram>(mr_scanner);
	}
}

void Corrector::setGlobalScalingFactor(float globalScalingFactor)
{
	m_globalScalingFactor = globalScalingFactor;
}

void Corrector::setSensitivityHistogram(const Histogram* pp_sensitivity)
{
	mp_sensitivity = pp_sensitivity;
}

void Corrector::setInvertSensitivity(bool invert)
{
	m_invertSensitivity = invert;
}

void Corrector::setRandomsHistogram(const Histogram* pp_randoms)
{
	mp_randoms = pp_randoms;
}

void Corrector::setScatterHistogram(const Histogram* pp_scatter)
{
	mp_scatter = pp_scatter;
}

void Corrector::setAttenuationImage(const Image* pp_attenuationImage)
{
	mp_attenuationImage = pp_attenuationImage;
	m_attenuationSetupComplete = false;
}

void Corrector::setACFHistogram(const Histogram* pp_acf)
{
	mp_acf = pp_acf;
	m_attenuationSetupComplete = false;
}

void Corrector::setHardwareAttenuationImage(
    const Image* pp_hardwareAttenuationImage)
{
	mp_hardwareAttenuationImage = pp_hardwareAttenuationImage;
	m_attenuationSetupComplete = false;
}

void Corrector::setHardwareACFHistogram(const Histogram* pp_hardwareAcf)
{
	mp_hardwareAcf = pp_hardwareAcf;
	m_attenuationSetupComplete = false;
}

void Corrector::setInVivoAttenuationImage(
    const Image* pp_inVivoAttenuationImage)
{
	mp_inVivoAttenuationImage = pp_inVivoAttenuationImage;
	m_attenuationSetupComplete = false;
}

void Corrector::setInVivoACFHistogram(const Histogram* pp_inVivoAcf)
{
	mp_hardwareAcf = pp_inVivoAcf;
	m_attenuationSetupComplete = false;
}

const Histogram* Corrector::getSensitivityHistogram() const
{
	return mp_sensitivity;
}

const ProjectionData* Corrector::getSensImgGenProjData() const
{
	// Returns the buffer that should be used to iterate over bins and compute
	//  LORs
	if (mp_sensitivity != nullptr)
	{
		return mp_sensitivity;
	}
	if (mp_hardwareAcf != nullptr)
	{
		return mp_hardwareAcf;
	}
	ASSERT(mp_uniformHistogram != nullptr);
	return mp_uniformHistogram.get();
}

bool Corrector::hasSensitivityHistogram() const
{
	return mp_sensitivity != nullptr;
}

float Corrector::getGlobalScalingFactor() const
{
	return m_globalScalingFactor;
}

bool Corrector::hasGlobalScalingFactor() const
{
	return std::abs(1.0f - m_globalScalingFactor) > EPS_FLT;
}

bool Corrector::hasAttenuation() const
{
	return hasHardwareAttenuation() || hasInVivoAttenuation();
}

bool Corrector::hasHardwareAttenuation() const
{
	return mp_hardwareAcf != nullptr || mp_hardwareAttenuationImage != nullptr;
}

bool Corrector::hasHardwareAttenuationImage() const
{
	return mp_hardwareAttenuationImage != nullptr;
}

const Image* Corrector::getHardwareAttenuationImage() const
{
	return mp_hardwareAttenuationImage;
}

bool Corrector::hasMultiplicativeCorrection() const
{
	// Has either hardware attenuation or sensitivity
	return hasHardwareAttenuation() || hasSensitivityHistogram();
}

bool Corrector::mustInvertSensitivity() const
{
	return m_invertSensitivity;
}

bool Corrector::doesHardwareACFComeFromHistogram() const
{
	return mp_hardwareAcf != nullptr;
}

float Corrector::getSensitivity(const histo_bin_t& histoBin) const
{
	if (mp_sensitivity != nullptr)
	{
		float sensitivity =
		    mp_sensitivity->getProjectionValueFromHistogramBin(histoBin);
		if (m_invertSensitivity && sensitivity != 0.0f)
		{
			sensitivity = 1.0f / sensitivity;
		}
		return sensitivity;
	}
	return 1.0f;
}

float Corrector::getHardwareACFFromHistogram(const histo_bin_t& histoBin) const
{
	ASSERT(mp_hardwareAcf != nullptr);
	return mp_hardwareAcf->getProjectionValueFromHistogramBin(histoBin);
}

bool Corrector::hasRandomsEstimates(const ProjectionData& measurements) const
{
	return mp_randoms != nullptr || measurements.hasRandomsEstimates();
}

bool Corrector::hasScatterEstimates() const
{
	return mp_scatter != nullptr;
}

bool Corrector::hasAdditiveCorrection(const ProjectionData& measurements) const
{
	return hasRandomsEstimates(measurements) || hasScatterEstimates();
}

bool Corrector::hasInVivoAttenuation() const
{
	return mp_inVivoAcf != nullptr || mp_inVivoAttenuationImage != nullptr;
}

const Image* Corrector::getInVivoAttenuationImage() const
{
	return mp_inVivoAttenuationImage;
}

bool Corrector::doesTotalACFComeFromHistogram() const
{
	return mp_acf != nullptr ||
	       (mp_hardwareAcf != nullptr && mp_inVivoAcf != nullptr);
}

bool Corrector::doesInVivoACFComeFromHistogram() const
{
	return mp_inVivoAcf != nullptr;
}

void Corrector::precomputeSensitivityFactors(const ProjectionData& measurements)
{
	ASSERT_MSG(hasSensitivityHistogram(), "No sensitivity correction needed");

	auto sensitivityCorrections =
	    std::make_unique<ProjectionListOwned>(&measurements);
	sensitivityCorrections->allocate();
	mp_sensitivityFactors = std::move(sensitivityCorrections);

	float* sensitivityCorrectionsPtr = mp_sensitivityFactors->getRawPointer();

	const bin_t numBins = measurements.count();
	std::cout << "Precomputing sensitivity corrections..." << std::endl;

	util::parallelForChunked(numBins, globals::getNumThreads(),
	                         [&measurements, sensitivityCorrectionsPtr,
	                          this](bin_t bin, size_t /*tid*/)
	                         {
		                         const histo_bin_t histoBin =
		                             measurements.getHistogramBin(bin);
		                         sensitivityCorrectionsPtr[bin] =
		                             getSensitivity(histoBin);
	                         });
}

void Corrector::precomputeAttenuationFactors(const ProjectionData& measurements)
{
	ASSERT_MSG(hasAttenuation(), "No attenuation correction needed");

	auto attenuationCorrections =
	    std::make_unique<ProjectionListOwned>(&measurements);
	attenuationCorrections->allocate();
	mp_attenuationFactors = std::move(attenuationCorrections);

	float* attenuationCorrectionsPtr = mp_attenuationFactors->getRawPointer();

	const bin_t numBins = measurements.count();
	std::cout << "Precomputing attenuation corrections..." << std::endl;

	util::parallelForChunked(numBins, globals::getNumThreads(),
	                         [&measurements, attenuationCorrectionsPtr,
	                          this](bin_t bin, size_t /*tid*/)
	                         {
		                         attenuationCorrectionsPtr[bin] =
		                             getAttenuationFactor(measurements, bin);
	                         });
}

void Corrector::precomputeRandomsEstimates(const ProjectionData& measurements)
{
	ASSERT_MSG(hasRandomsEstimates(measurements),
	           "No randoms correction needed");

	auto randomsEstimates =
	    std::make_unique<ProjectionListOwned>(&measurements);
	randomsEstimates->allocate();
	mp_randomsEstimates = std::move(randomsEstimates);

	float* randomsEstimatesPtr = mp_randomsEstimates->getRawPointer();

	const bin_t numBins = measurements.count();
	std::cout << "Gathering randoms estimates..." << std::endl;

	util::parallelForChunked(
	    numBins, globals::getNumThreads(),
	    [&measurements, randomsEstimatesPtr, this](bin_t bin, size_t /*tid*/)
	    { randomsEstimatesPtr[bin] = getRandomsEstimate(measurements, bin); });
}

void Corrector::precomputeScatterEstimates(const ProjectionData& measurements)
{
	ASSERT_MSG(hasScatterEstimates(), "No scatter correction needed");

	auto scatterEstimates =
	    std::make_unique<ProjectionListOwned>(&measurements);
	scatterEstimates->allocate();
	mp_scatterEstimates = std::move(scatterEstimates);

	float* scatterEstimatesPtr = mp_scatterEstimates->getRawPointer();

	const bin_t numBins = measurements.count();
	std::cout << "Gathering scatter estimates..." << std::endl;

	util::parallelForChunked(
	    numBins, globals::getNumThreads(),
	    [&measurements, scatterEstimatesPtr, this](bin_t bin, size_t /*tid*/)
	    {
		    const histo_bin_t histoBin = measurements.getHistogramBin(bin);
		    scatterEstimatesPtr[bin] = getScatterEstimate(histoBin);
	    });
}

void Corrector::precomputeInVivoAttenuationFactors(
    const ProjectionData& measurements)
{
	ASSERT_MSG(hasInVivoAttenuation(),
	           "No in-vivo attenuation corrections needed");

	auto inVivoAttenuationFactors =
	    std::make_unique<ProjectionListOwned>(&measurements);
	inVivoAttenuationFactors->allocate();
	mp_inVivoAttenuationFactors = std::move(inVivoAttenuationFactors);

	float* inVivoAttenuationFactorsPtr =
	    mp_inVivoAttenuationFactors->getRawPointer();

	const size_t numBins = measurements.count();
	std::cout << "Precomputing in-vivo attenuation corrections..." << std::endl;

	util::parallelForChunked(numBins, globals::getNumThreads(),
	                         [&measurements, inVivoAttenuationFactorsPtr,
	                          this](bin_t bin, size_t /*tid*/)
	                         {
		                         inVivoAttenuationFactorsPtr[bin] =
		                             getInVivoAttenuationFactor(measurements,
		                                                        bin);
	                         });
}

void Corrector::precomputeCorrectionFactors(const ProjectionData& measurements)
{
	if (hasSensitivityHistogram())
	{
		precomputeSensitivityFactors(measurements);
	}
	if (hasAttenuation())
	{
		precomputeAttenuationFactors(measurements);
	}
	if (hasRandomsEstimates(measurements))
	{
		precomputeRandomsEstimates(measurements);
	}
	if (hasScatterEstimates())
	{
		precomputeScatterEstimates(measurements);
	}
	if (hasInVivoAttenuation())
	{
		precomputeInVivoAttenuationFactors(measurements);
	}
}

void Corrector::resetAllPrecomputedFactors()
{
	mp_sensitivityFactors = nullptr;
	mp_attenuationFactors = nullptr;
	mp_randomsEstimates = nullptr;
	mp_scatterEstimates = nullptr;
	mp_inVivoAttenuationFactors = nullptr;
}

float Corrector::getPrecomputedSensitivityFactor(bin_t binId) const
{
	// TODO: Maybe these ASSERTs are too slow to put here
	ASSERT(mp_sensitivityFactors != nullptr &&
	       mp_sensitivityFactors->isMemoryValid());
	return mp_sensitivityFactors->getProjectionValue(binId);
}

float Corrector::getPrecomputedAttenuationFactor(bin_t binId) const
{
	ASSERT(mp_attenuationFactors != nullptr &&
	       mp_attenuationFactors->isMemoryValid());
	return mp_attenuationFactors->getProjectionValue(binId);
}

float Corrector::getPrecomputedRandomsEstimate(bin_t binId) const
{
	ASSERT(mp_randomsEstimates != nullptr &&
	       mp_randomsEstimates->isMemoryValid());
	return mp_randomsEstimates->getProjectionValue(binId);
}

float Corrector::getPrecomputedScatterEstimate(bin_t binId) const
{
	ASSERT(mp_scatterEstimates != nullptr &&
	       mp_scatterEstimates->isMemoryValid());
	return mp_scatterEstimates->getProjectionValue(binId);
}

float Corrector::getPrecomputedInVivoAttenuationFactor(bin_t binId) const
{
	ASSERT(mp_inVivoAttenuationFactors != nullptr &&
	       mp_inVivoAttenuationFactors->isMemoryValid());
	return mp_inVivoAttenuationFactors->getProjectionValue(binId);
}

void Corrector::assertMeasurementsMatchCache(
    const ProjectionData* measurements) const
{
	assertReferenceMatch(
	    mp_sensitivityFactors.get(), measurements,
	    "Sensitivity factors were not gathered for this dataset");
	assertReferenceMatch(
	    mp_attenuationFactors.get(), measurements,
	    "Attenuation factors were not computed for this dataset");
	assertReferenceMatch(
	    mp_randomsEstimates.get(), measurements,
	    "Randoms estimates were not gathered for this dataset");
	assertReferenceMatch(
	    mp_scatterEstimates.get(), measurements,
	    "Scatter estimates were not gathered for this dataset");
	assertReferenceMatch(
	    mp_inVivoAttenuationFactors.get(), measurements,
	    "In-vivo attenuation factors were not computed for this dataset");
}

float Corrector::getRandomsEstimate(const ProjectionData& measurements,
                                    bin_t binId) const
{
	if (mp_randoms != nullptr)
	{
		const histo_bin_t histoBin = measurements.getHistogramBin(binId);
		return mp_randoms->getProjectionValueFromHistogramBin(histoBin);
	}
	return measurements.getRandomsEstimate(binId);
}

float Corrector::getScatterEstimate(const histo_bin_t& histoBin) const
{
	if (mp_scatter != nullptr)
	{
		return mp_scatter->getProjectionValueFromHistogramBin(histoBin);
	}
	return 0.0f;
}

float Corrector::getTotalACFFromHistogram(const histo_bin_t& histoBin) const
{
	if (mp_acf != nullptr)
	{
		return mp_acf->getProjectionValueFromHistogramBin(histoBin);
	}
	if (mp_inVivoAcf != nullptr && mp_hardwareAcf != nullptr)
	{
		// Total ACF has to be computed from both components
		return mp_inVivoAcf->getProjectionValueFromHistogramBin(histoBin) *
		       mp_hardwareAcf->getProjectionValueFromHistogramBin(histoBin);
	}
	ASSERT_MSG(false, "No ACF histogram provided");
	return 0.0f;
}

float Corrector::getAttenuationFactorFromAttenuationImage(
    const ProjectionData& measurements, bin_t binId,
    const Image& attenuationImage)
{
	const Line3D lor = measurements.getLOR(binId);

	const float att =
	    ProjectorSiddon::singleForwardProjection(&attenuationImage, lor);

	return util::getAttenuationCoefficientFactor(att);
}

float Corrector::getAttenuationFactor(const ProjectionData& measurements,
                                      bin_t binId) const
{
	float acf;

	if (doesTotalACFComeFromHistogram())
	{
		const histo_bin_t histoBin = measurements.getHistogramBin(binId);

		acf = getTotalACFFromHistogram(histoBin);
	}
	else if (mp_attenuationImage != nullptr)
	{
		acf = getAttenuationFactorFromAttenuationImage(measurements, binId,
		                                               *mp_attenuationImage);
	}
	else
	{
		acf = 1.0f;
	}

	return acf;
}

float Corrector::getInVivoAttenuationFactor(const ProjectionData& measurements,
                                            bin_t binId) const
{
	const histo_bin_t histoBin = measurements.getHistogramBin(binId);

	if (mp_inVivoAcf != nullptr)
	{
		return mp_inVivoAcf->getProjectionValueFromHistogramBin(histoBin);
	}
	if (mp_inVivoAttenuationImage != nullptr)
	{
		return getAttenuationFactorFromAttenuationImage(
		    measurements, binId, *mp_inVivoAttenuationImage);
	}

	return 1.0f;
}

bool Corrector::assertReferenceMatch(const ProjectionList* projList,
                                     const ProjectionData* reference,
                                     const char* errorMessage)
{
	if (projList != nullptr && projList->isMemoryValid())
	{
		ASSERT_MSG(projList->getReference() == reference, errorMessage);
		return true;
	}
	return false;
}

}  // namespace yrt
