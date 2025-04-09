/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/Corrector.hpp"

#include "utils/Assert.hpp"
#include "utils/Tools.hpp"


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
      mp_impliedTotalAttenuationImage(nullptr),
      mp_sensitivity(nullptr),
      m_invertSensitivity(false),
      m_globalScalingFactor(1.0f),
      mp_tofHelper(nullptr),
      m_attenuationSetupComplete{false}
{
}

void Corrector::setSensitivityHistogram(const Histogram* pp_sensitivity)
{
	mp_sensitivity = pp_sensitivity;
}

void Corrector::setRandomsHistogram(const Histogram* pp_randoms)
{
	mp_randoms = pp_randoms;
}

void Corrector::setScatterHistogram(const Histogram* pp_scatter)
{
	mp_scatter = pp_scatter;
}

void Corrector::setGlobalScalingFactor(float globalScalingFactor)
{
	m_globalScalingFactor = globalScalingFactor;
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

void Corrector::setInvertSensitivity(bool invert)
{
	m_invertSensitivity = invert;
}

void Corrector::addTOF(float p_tofWidth_ps, int p_tofNumStd)
{
	mp_tofHelper =
	    std::make_unique<TimeOfFlightHelper>(p_tofWidth_ps, p_tofNumStd);
}

const Histogram* Corrector::getSensitivityHistogram() const
{
	return mp_sensitivity;
}

float Corrector::getGlobalScalingFactor() const
{
	return m_globalScalingFactor;
}

bool Corrector::hasGlobalScalingFactor() const
{
	return std::abs(1.0f - m_globalScalingFactor) > EPS_FLT;
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
			mp_impliedTotalAttenuationImage->copyFromImage(
			    mp_hardwareAttenuationImage);
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

bool Corrector::hasAdditiveCorrection() const
{
	return mp_randoms != nullptr || mp_scatter != nullptr;
}

bool Corrector::mustInvertSensitivity() const
{
	return m_invertSensitivity;
}

bool Corrector::hasInVivoAttenuation() const
{
	return mp_inVivoAcf != nullptr || mp_inVivoAttenuationImage != nullptr;
}

const Image* Corrector::getInVivoAttenuationImage() const
{
	return mp_inVivoAttenuationImage;
}

float Corrector::getRandomsEstimate(const ProjectionData& measurements,
                                    bin_t binId, histo_bin_t histoBin) const
{
	if (mp_randoms != nullptr)
	{
		return mp_randoms->getProjectionValueFromHistogramBin(histoBin);
	}
	return measurements.getRandomsEstimate(binId);
}

float Corrector::getScatterEstimate(histo_bin_t histoBin) const
{
	if (mp_scatter != nullptr)
	{
		// TODO: Support exception in case of a contiguous sinogram (future)
		return mp_scatter->getProjectionValueFromHistogramBin(histoBin);
	}
	return 0.0f;
}

float Corrector::getSensitivity(histo_bin_t histoBin) const
{
	if (mp_sensitivity != nullptr)
	{
		float sensitivity =
		    m_globalScalingFactor *
		    mp_sensitivity->getProjectionValueFromHistogramBin(histoBin);
		if (m_invertSensitivity && sensitivity != 0.0f)
		{
			sensitivity = 1.0f / sensitivity;
		}
		return sensitivity;
	}
	return m_globalScalingFactor;
}

float Corrector::getTotalACFFromHistogram(histo_bin_t histoBin) const
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
	ASSERT_MSG(false, "Unexpected error");
	return 0.0f;
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

bool Corrector::doesHardwareACFComeFromHistogram() const
{
	return mp_hardwareAcf != nullptr;
}
