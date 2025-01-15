/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "datastruct/projection/Histogram.hpp"
#include "datastruct/projection/ProjectionList.hpp"
#include "operators/TimeOfFlight.hpp"

/*
 * This class provides the additive correction factors for each LOR given
 * measurements and individual correction components
 */
class Corrector
{

public:
	Corrector();

	void setInvertSensitivity(bool invert);
	void setGlobalScalingFactor(float globalScalingFactor);
	void setSensitivityHistogram(const Histogram* pp_sensitivity);
	void setRandomsHistogram(const Histogram* pp_randoms);
	void setScatterHistogram(const Histogram* pp_scatter);

	void setAttenuationImage(const Image* pp_attenuationImage);
	void setACFHistogram(const Histogram* pp_acf);
	void setHardwareAttenuationImage(const Image* pp_hardwareAttenuationImage);
	void setHardwareACFHistogram(const Histogram* pp_hardwareAcf);
	void setInVivoAttenuationImage(const Image* pp_inVivoAttenuationImage);
	void setInVivoACFHistogram(const Histogram* pp_inVivoAcf);

	void addTOF(float p_tofWidth_ps, int p_tofNumStd);

	// Simplify user input
	void setup();

	// For sensitivity image generation
	bool hasMultiplicativeCorrection() const;

	// For reconstruction
	bool hasAdditiveCorrection() const;
	bool hasInVivoAttenuation() const;

protected:

	// if nullptr, use getRandomsEstimate()
	const Histogram* mp_randoms;

	// Can also be a sinogram (once the format exists)
	const Histogram* mp_scatter;

	// Histogram of ACFs in case ACFs were already calculated
	const Histogram* mp_acf;           // total ACF
	const Image* mp_attenuationImage;  // total attenuation image

	// Distinction for motion correction
	const Histogram* mp_inVivoAcf;
	const Image* mp_inVivoAttenuationImage;
	const Histogram* mp_hardwareAcf;
	const Image* mp_hardwareAttenuationImage;

	// In case it is not specified and must be computed
	std::unique_ptr<ImageOwned> mp_impliedTotalAttenuationImage;

	// LOR sensitivity, can be nullptr, in which case all LORs are equally
	// sensitive
	const Histogram* mp_sensitivity;
	bool m_invertSensitivity;
	float m_globalScalingFactor;

	// Time of flight (For computing attenuation factors from attenuation image)
	std::unique_ptr<TimeOfFlightHelper> mp_tofHelper;
};
