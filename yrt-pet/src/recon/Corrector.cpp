/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/Corrector.hpp"

#include "utils/Assert.hpp"
#include "utils/Tools.hpp"


Corrector::Corrector()
    : mp_randoms(nullptr),
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
      mp_tofHelper(nullptr)
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
}

void Corrector::setACFHistogram(const Histogram* pp_acf)
{
	mp_acf = pp_acf;
}

void Corrector::setHardwareAttenuationImage(
    const Image* pp_hardwareAttenuationImage)
{
	mp_hardwareAttenuationImage = pp_hardwareAttenuationImage;
}

void Corrector::setHardwareACFHistogram(const Histogram* pp_hardwareAcf)
{
	mp_hardwareAcf = pp_hardwareAcf;
}

void Corrector::setInVivoAttenuationImage(
    const Image* pp_inVivoAttenuationImage)
{
	mp_inVivoAttenuationImage = pp_inVivoAttenuationImage;
}

void Corrector::setInVivoACFHistogram(const Histogram* pp_inVivoAcf)
{
	mp_hardwareAcf = pp_inVivoAcf;
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

void Corrector::setup()
{
	// Send warnings if needed
	if (mp_acf != nullptr && mp_inVivoAcf != nullptr &&
	    mp_hardwareAcf == nullptr)
	{
		std::cout << "Warning: Total ACF and in-vivo ACF specified, but no "
		             "hardware ACF specified."
		          << std::endl;
	}
	else if (mp_acf != nullptr && mp_inVivoAcf == nullptr &&
	         mp_hardwareAcf != nullptr)
	{
		std::cout << "Warning: Total ACF and hardware ACF specified, but no "
		             "in-vivo ACF specified."
		          << std::endl;
	}

	// Adjust attenuation image logic

	if (mp_inVivoAttenuationImage != nullptr &&
	    mp_hardwareAttenuationImage != nullptr &&
	    mp_attenuationImage == nullptr)
	{
		// Here, the hardware and in-vivo attenuation images were specified,
		// but the total attenuation image wasn't. The total attenuation image
		// should be the sum of the in-vivo and the hardware
		const ImageParams attParams = mp_hardwareAttenuationImage->getParams();
		ASSERT_MSG(attParams.isSameAs(mp_inVivoAttenuationImage->getParams()),
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
		// User only specified total attenuation, but not how it is distributed.
		// We assume that all the attenuation comes from hardware, which is
		// the case when there is no motion
		mp_hardwareAttenuationImage = mp_attenuationImage;
	}
	else if (mp_inVivoAttenuationImage != nullptr &&
	         mp_hardwareAttenuationImage == nullptr &&
	         mp_attenuationImage != nullptr)
	{
		std::cout << "Warning: Hardware attenuation image not specified while "
		             "full attenuation and in-vivo attenuation is specified. "
		             "It will be assumed that there is no hardware attenuation."
		          << std::endl;
	}
	else if (mp_inVivoAttenuationImage == nullptr &&
	         mp_hardwareAttenuationImage != nullptr &&
	         mp_attenuationImage != nullptr)
	{
		std::cout << "Warning: In-vivo attenuation image not specified while "
		             "full attenuation and hardware attenuation is specified. "
		             "It will be assumed that there is no in-vivo attenuation."
		          << std::endl;
	}
}

bool Corrector::hasMultiplicativeCorrection() const
{
	// Has either hardware attenuation or sensitivity
	return mp_hardwareAcf != nullptr ||
	       (mp_acf != nullptr && mp_inVivoAcf == nullptr) ||
	       mp_hardwareAttenuationImage != nullptr || mp_sensitivity != nullptr;
}

bool Corrector::hasAdditiveCorrection() const
{
	return mp_randoms != nullptr || mp_scatter != nullptr;
}

bool Corrector::hasInVivoAttenuation() const
{
	return mp_inVivoAcf != nullptr || mp_inVivoAttenuationImage != nullptr;
}
