/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/Corrector.hpp"

#include "operators/OperatorProjectorSiddon.hpp"
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

float Corrector::getMultiplicativeCorrectionFactor(
    const ProjectionData* measurements, bin_t binId) const
{
	ASSERT(measurements != nullptr);

	const histo_bin_t histoBin = measurements->getHistogramBin(binId);

	float sensitivity;
	if (mp_sensitivity != nullptr)
	{
		sensitivity =
		    m_globalScalingFactor *
		    mp_sensitivity->getProjectionValueFromHistogramBin(histoBin);
		if (m_invertSensitivity)
		{
			sensitivity = 1.0f / sensitivity;
		}
	}
	else
	{
		sensitivity = m_globalScalingFactor;
	}

	float acf;
	if (mp_hardwareAcf != nullptr)
	{
		// Hardware ACF
		acf = mp_hardwareAcf->getProjectionValueFromHistogramBin(histoBin);
	}
	else if (mp_acf != nullptr && mp_inVivoAcf == nullptr)
	{
		// All ACF is Hardware ACF (no motion)
		acf = mp_acf->getProjectionValueFromHistogramBin(histoBin);
	}
	else if (mp_hardwareAttenuationImage != nullptr)
	{
		acf = getAttenuationFactorFromAttenuationImage(
		    measurements, binId, mp_hardwareAttenuationImage);
	}
	else
	{
		acf = 1.0f;
	}

	return acf * sensitivity;
}

bool Corrector::hasMultiplicativeCorrection() const
{
	// Has either hardware attenuation or sensitivity
	return mp_hardwareAcf != nullptr ||
	       (mp_acf != nullptr && mp_inVivoAcf == nullptr) ||
	       mp_hardwareAttenuationImage != nullptr || mp_sensitivity != nullptr;
}

float Corrector::getAdditiveCorrectionFactor(const ProjectionData* measurements,
                                             bin_t binId) const
{
	const histo_bin_t histoBin = measurements->getHistogramBin(binId);

	float randomsEstimate;
	if (mp_randoms != nullptr)
	{
		randomsEstimate =
		    mp_randoms->getProjectionValueFromHistogramBin(histoBin);
	}
	else
	{
		randomsEstimate = measurements->getRandomsEstimate(binId);
	}

	float scatterEstimate = 0.0f;
	if (mp_scatter != nullptr)
	{
		// TODO: Support exception in case of a contiguous sinogram (future)
		scatterEstimate =
		    mp_scatter->getProjectionValueFromHistogramBin(histoBin);
	}

	float sensitivity;
	if (mp_sensitivity != nullptr)
	{
		sensitivity =
		    m_globalScalingFactor *
		    mp_sensitivity->getProjectionValueFromHistogramBin(histoBin);
		if (m_invertSensitivity)
		{
			sensitivity = 1.0f / sensitivity;
		}
	}
	else
	{
		sensitivity = m_globalScalingFactor;
	}

	float acf;
	if (mp_acf != nullptr)
	{
		acf = mp_acf->getProjectionValueFromHistogramBin(histoBin);
	}
	else if (mp_inVivoAcf != nullptr && mp_hardwareAcf != nullptr)
	{
		acf = mp_inVivoAcf->getProjectionValueFromHistogramBin(histoBin) *
		      mp_hardwareAcf->getProjectionValueFromHistogramBin(histoBin);
	}
	else if (mp_attenuationImage != nullptr)
	{
		acf = getAttenuationFactorFromAttenuationImage(measurements, binId,
		                                               mp_attenuationImage);
	}
	else
	{
		acf = 1.0f;
	}

	return (randomsEstimate + scatterEstimate) / (acf * sensitivity);
}

bool Corrector::hasAdditiveCorrection() const
{
	return mp_randoms != nullptr || mp_scatter != nullptr;
}

float Corrector::getInVivoAttenuationFactor(const ProjectionData* measurements,
                                            bin_t binId) const
{
	const histo_bin_t histoBin = measurements->getHistogramBin(binId);

	if (mp_inVivoAcf != nullptr)
	{
		return mp_inVivoAcf->getProjectionValueFromHistogramBin(histoBin);
	}
	if (mp_inVivoAttenuationImage != nullptr)
	{
		return getAttenuationFactorFromAttenuationImage(
		    measurements, binId, mp_inVivoAttenuationImage);
	}

	return 1.0f;
}

bool Corrector::hasInVivoAttenuation() const
{
	return mp_inVivoAcf != nullptr || mp_inVivoAttenuationImage != nullptr;
}

float Corrector::getAttenuationFactorFromAttenuationImage(
    const ProjectionData* measurements, bin_t binId,
    const Image* attenuationImage) const
{
	const Line3D lor = measurements->getLOR(binId);

	const float tofValue =
	    measurements->hasTOF() ? measurements->getTOFValue(binId) : 0.0f;

	const float att = OperatorProjectorSiddon::singleForwardProjection(
	    attenuationImage, lor, mp_tofHelper.get(), tofValue);

	return Util::getAttenuationCoefficientFactor(att);
}
