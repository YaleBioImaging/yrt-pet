/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/Histogram.hpp"
#include "yrt-pet/operators/OperatorProjectorUpdater.hpp"
#include "yrt-pet/datastruct/projection/ProjectionList.hpp"
#include "yrt-pet/datastruct/projection/UniformHistogram.hpp"
#include "yrt-pet/operators/TimeOfFlight.hpp"

namespace yrt
{
/*
 * This class provides the additive correction factors for each LOR given
 * measurements and individual correction components
 */
class Corrector
{
public:
	explicit Corrector(const Scanner& pr_scanner);
	virtual ~Corrector() = default;

	virtual void precomputeAdditiveCorrectionFactors(
	    const ProjectionData& measurements) = 0;
	virtual void precomputeInVivoAttenuationFactors(
	    const ProjectionData& measurements) = 0;

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

	const Histogram* getSensitivityHistogram() const;
	float getGlobalScalingFactor() const;
	bool hasGlobalScalingFactor() const;

	// Simplify user input
	void setup();

	// For sensitivity image generation
	const ProjectionData* getSensImgGenProjData() const;
	bool hasSensitivityHistogram() const;
	bool hasHardwareAttenuation() const;
	bool hasHardwareAttenuationImage() const;
	const Image* getHardwareAttenuationImage() const;
	bool hasMultiplicativeCorrection() const;
	bool mustInvertSensitivity() const;
	bool doesHardwareACFComeFromHistogram() const;

	// For reconstruction
	bool hasAdditiveCorrection(const ProjectionData& measurements) const;
	bool hasInVivoAttenuation() const;
	const Image* getInVivoAttenuationImage() const;
	bool doesTotalACFComeFromHistogram() const;
	bool doesInVivoACFComeFromHistogram() const;

protected:
	static constexpr float StabilityEpsilon = EPS_FLT;

	// Helper functions
	float getRandomsEstimate(const ProjectionData& measurements, bin_t binId,
	                         histo_bin_t histoBin) const;
	float getScatterEstimate(histo_bin_t histoBin) const;
	float getSensitivity(histo_bin_t histoBin) const;
	float getTotalACFFromHistogram(histo_bin_t histoBin) const;


	const Scanner& mr_scanner;

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

	// In case no sensitivity histogram or ACF histogram was given
	std::unique_ptr<UniformHistogram> mp_uniformHistogram;

	// LOR sensitivity, can be nullptr, in which case all LORs are equally
	// sensitive
	const Histogram* mp_sensitivity;
	bool m_invertSensitivity;

	// Global scaling on the sensitivity
	float m_globalScalingFactor;

	// Time of flight (For computing attenuation factors from attenuation image)
	std::unique_ptr<TimeOfFlightHelper> mp_tofHelper;

	bool m_attenuationSetupComplete;

	// Updater for forward and back-projection
	std::shared_ptr<OperatorProjectorUpdater> mp_updater;
};
}  // namespace yrt
