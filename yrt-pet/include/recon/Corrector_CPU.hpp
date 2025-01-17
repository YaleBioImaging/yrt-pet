/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "recon/Corrector.hpp"

class Corrector_CPU : public Corrector
{
public:
	explicit Corrector_CPU(const Scanner& pr_scanner);

	// Return sensitivity*attenuation
	float getMultiplicativeCorrectionFactor(const ProjectionData* measurements,
						bin_t binId) const;

	// Pre-computes a ProjectionList of (randoms+scatter)/(acf*sensitivity) for
	//  each LOR in 'measurements'
	void
	    precomputeAdditiveCorrectionFactors(const ProjectionData* measurements);
	// Pre-computes a ProjectionList of a^(i)_i for each LOR in measurements
	void precomputeInVivoAttenuationFactors(const ProjectionData* measurements);
	float getAdditiveCorrectionFactor(bin_t binId) const;
	float getInVivoAttenuationFactor(bin_t binId) const;
	const ProjectionData*
	    getCachedMeasurementsForAdditiveCorrectionFactors() const;
	const ProjectionData*
	    getCachedMeasurementsForInVivoAttenuationFactors() const;
	// TODO NOW: use const reference instead of pointers to ProjectionData
	//  measurements
private:
	// Functions used for precomputation only:
	// Return (randoms+scatter)/(sensitivity*attenuation)
	float getAdditiveCorrectionFactor(const ProjectionData* measurements,
	                                  bin_t binId) const;
	// Return a^(i)_i
	float getInVivoAttenuationFactor(const ProjectionData* measurements,
	                                 bin_t binId) const;
	// Helper functions:
	// Given measurements, a bin, and an attenuation image, compute the
	//  appropriate attenuation factor
	float getAttenuationFactorFromAttenuationImage(
	    const ProjectionData* measurements, bin_t binId,
	    const Image* attenuationImage) const;

	// Pre-computed caches
	std::unique_ptr<ProjectionList> mp_additiveCorrections;
	std::unique_ptr<ProjectionList> mp_inVivoAttenuationFactors;
};
