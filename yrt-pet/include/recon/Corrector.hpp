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

	void setSensitivityHistogram(const Histogram* pp_sensitivity);
	void setRandomsHistogram(const Histogram* pp_randoms);
	void setScatterHistogram(const Histogram* pp_scatter);
	void setGlobalScalingFactor(float globalScalingFactor);

	void setAttenuationImage(const Image* pp_attenuationImage);
	void setACFHistogram(const Histogram* pp_acf);

	void setInvertSensitivity(bool invert);

	void addTOF(float p_tofWidth_ps, int p_tofNumStd);

	// Returns a ProjectionList of (randoms+scatter)/(acf*sensitivity) for each
	// LOR in 'measurements'
	std::unique_ptr<ProjectionList> getAdditiveCorrectionFactors(
	    const ProjectionData* measurements,
	    const BinIterator* binIter = nullptr) const;

	float getAdditiveCorrectionFactor(const ProjectionData* measurements,
	                                  bin_t binId) const;

	// TODO NOW: Add function to get individual correction factors
	// TODO NOW: Add function to get the individual correction factor for
	//  sensitivity image generation
	// TODO NOW: Add function to get the correction factor for in-vivo
	// attenuation image (for motion)

private:
	std::unique_ptr<ProjectionList> getAdditiveCorrectionFactorsHelper(
	    const ProjectionData* measurements, const BinIterator* binIter,
	    bool useBinIterator, bool useRandomsHistogram, bool useScatter,
	    bool useAttenuationImage, bool usePrecomputedACF, bool useSensitivity,
	    bool invertSensitivity, bool hasTOF) const;

	// if nullptr, use getRandomsEstimate()
	const Histogram* mp_randoms;

	// Can also be a sinogram
	const Histogram* mp_scatter;

	// In case ACFs were already calculated
	const Histogram* mp_acf;
	const Image* mp_attenuationImage;

	// LOR sensitivity, can be nullptr, in which case all LORs are equally
	// sensitive
	const Histogram* mp_sensitivity;
	bool m_invertSensitivity;
	float m_globalScalingFactor;

	// Time of flight
	std::unique_ptr<TimeOfFlightHelper> mp_tofHelper;
};
