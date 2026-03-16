/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/Histogram.hpp"
#include "yrt-pet/datastruct/projection/ProjectionList.hpp"
#include "yrt-pet/datastruct/projection/UniformHistogram.hpp"
#include "yrt-pet/operators/ProjectorUpdater.hpp"
#include "yrt-pet/operators/TimeOfFlight.hpp"

namespace yrt
{

// This class provides the additive correction factors for each LOR given
//  measurements and individual correction components
class Corrector
{
public:
	explicit Corrector(const Scanner& pr_scanner);
	virtual ~Corrector() = default;

	// Simplify user input
	void setup();

	// Setters for correction factors
	void setGlobalScalingFactor(float globalScalingFactor);
	void setSensitivityHistogram(const Histogram* pp_sensitivity);
	void setInvertSensitivity(bool invert);
	void setRandomsHistogram(const Histogram* pp_randoms);
	void setScatterHistogram(const Histogram* pp_scatter);

	// Setters for attenuation correction
	void setAttenuationImage(const Image* pp_attenuationImage);
	void setACFHistogram(const Histogram* pp_acf);
	void setHardwareAttenuationImage(const Image* pp_hardwareAttenuationImage);
	void setHardwareACFHistogram(const Histogram* pp_hardwareAcf);
	void setInVivoAttenuationImage(const Image* pp_inVivoAttenuationImage);
	void setInVivoACFHistogram(const Histogram* pp_inVivoAcf);

	// Getters for sensitivity image generation and reconstruction
	const Histogram* getSensitivityHistogram() const;
	const ProjectionData* getSensImgGenProjData() const;
	bool hasSensitivityHistogram() const;
	float getGlobalScalingFactor() const;
	bool hasGlobalScalingFactor() const;
	bool hasAttenuation() const;
	bool hasHardwareAttenuation() const;
	bool hasHardwareAttenuationImage() const;
	const Image* getHardwareAttenuationImage() const;
	bool hasMultiplicativeCorrection() const;
	bool mustInvertSensitivity() const;
	bool doesHardwareACFComeFromHistogram() const;
	float getSensitivity(const histo_bin_t& histoBin) const;
	float getHardwareACFFromHistogram(const histo_bin_t& histoBin) const;

	// Getters for reconstruction exclusively
	bool hasRandomsEstimates(const ProjectionData& measurements) const;
	bool hasScatterEstimates() const;
	bool hasAdditiveCorrection(const ProjectionData& measurements) const;
	bool hasInVivoAttenuation() const;
	const Image* getInVivoAttenuationImage() const;
	bool doesTotalACFComeFromHistogram() const;
	bool doesInVivoACFComeFromHistogram() const;

	// For use before reconstruction
	// Pre-computes a ProjectionList of the sensitivity (s_i)
	virtual void
	    precomputeSensitivityFactors(const ProjectionData& measurements);
	// Pre-computes a ProjectionList of the total ACFs (a_i)
	virtual void
	    precomputeAttenuationFactors(const ProjectionData& measurements);
	// Precompute a ProjectionList of the randoms estimates (r_i)
	virtual void precomputeRandomsEstimates(const ProjectionData& measurements);
	// Precompute a ProjectionList of the scatter estimates (\sigma_i)
	virtual void precomputeScatterEstimates(const ProjectionData& measurements);
	// Pre-computes a ProjectionList of in-vivo ACF a^(i)_i
	virtual void
	    precomputeInVivoAttenuationFactors(const ProjectionData& measurements);
	// This function calls all the precomputing functions above. Use before
	//  OSEM iterations
	virtual void
	    precomputeCorrectionFactors(const ProjectionData& measurements);
	// Deallocate the precomputed values
	void resetAllPrecomputedFactors();

	// For use within reconstruction
	// These function assumes that the "binId" provided is the one coming from
	//  the "measurements" used in the precomputing above
	float getPrecomputedSensitivityFactor(bin_t binId) const;
	float getPrecomputedAttenuationFactor(bin_t binId) const;
	float getPrecomputedRandomsEstimate(bin_t binId) const;
	float getPrecomputedScatterEstimate(bin_t binId) const;
	float getPrecomputedInVivoAttenuationFactor(bin_t binId) const;

	// To validate
	void assertMeasurementsMatchCache(const ProjectionData* measurements) const;

protected:
	static constexpr float StabilityEpsilon = EPS_FLT;

	// Helper functions
	float getRandomsEstimate(const ProjectionData& measurements,
	                         bin_t binId) const;
	float getScatterEstimate(const histo_bin_t& histoBin) const;
	float getTotalACFFromHistogram(const histo_bin_t& histoBin) const;

	// Given measurements, a bin, and an attenuation image, compute the
	//  appropriate attenuation factor
	static float getAttenuationFactorFromAttenuationImage(
	    const ProjectionData& measurements, bin_t binId,
	    const Image& attenuationImage);

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
	bool m_attenuationSetupComplete;

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

	// Pre-computed caches
	std::unique_ptr<ProjectionList> mp_sensitivityFactors;  // Sensitivity
	std::unique_ptr<ProjectionList> mp_attenuationFactors;  // Total ACF
	std::unique_ptr<ProjectionList> mp_randomsEstimates;    // Randoms estimates
	std::unique_ptr<ProjectionList> mp_scatterEstimates;    // Scatter estimates
	std::unique_ptr<ProjectionList> mp_inVivoAttenuationFactors;  // in-vivo ACF

private:
	// Functions used for precomputation only:
	// Return the total ACF
	float getAttenuationFactor(const ProjectionData& measurements,
	                           bin_t binId) const;
	// Return a^(i)_i (ACF precorrection)
	float getInVivoAttenuationFactor(const ProjectionData& measurements,
	                                 bin_t binId) const;
	// Checks whether the reference of "projList" is "reference"
	static bool assertReferenceMatch(const ProjectionList* projList,
	                                 const ProjectionData* reference,
	                                 const char* errorMessage);
};
}  // namespace yrt
