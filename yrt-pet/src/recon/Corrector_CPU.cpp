/*
* This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/Corrector_CPU.hpp"

#include "utils/Assert.hpp"


void Corrector_CPU::precomputeAdditiveCorrectionFactors(
    const ProjectionData* measurements)
{
	ASSERT(measurements != nullptr);

	auto additiveCorrections =
	    std::make_unique<ProjectionListOwned>(measurements);
	additiveCorrections->allocate();

	mp_additiveCorrections = std::move(additiveCorrections);
	float* additiveCorrectionsPtr = mp_additiveCorrections->getRawPointer();

	const size_t numBins = measurements->count();

#pragma omp parallel for default(none) \
    firstprivate(numBins, measurements, additiveCorrectionsPtr)
	for (bin_t bin = 0; bin < numBins; bin++)
	{
		additiveCorrectionsPtr[bin] =
		    getAdditiveCorrectionFactor(measurements, bin);
	}
}

void Corrector_CPU::precomputeInVivoAttenuationFactors(
    const ProjectionData* measurements)
{
	ASSERT(measurements != nullptr);

	auto inVivoAttenuationFactors =
	    std::make_unique<ProjectionListOwned>(measurements);
	inVivoAttenuationFactors->allocate();

	mp_inVivoAttenuationFactors = std::move(inVivoAttenuationFactors);
	float* inVivoAttenuationFactorsPtr =
	    mp_inVivoAttenuationFactors->getRawPointer();

	const size_t numBins = measurements->count();

#pragma omp parallel for default(none) \
    firstprivate(numBins, measurements, inVivoAttenuationFactorsPtr)
	for (bin_t bin = 0; bin < numBins; bin++)
	{
		inVivoAttenuationFactorsPtr[bin] =
		    getInVivoAttenuationFactor(measurements, bin);
	}
}

float Corrector_CPU::getAdditiveCorrectionFactor(bin_t binId) const
{
	ASSERT(mp_additiveCorrections != nullptr &&
	       mp_additiveCorrections->isMemoryValid());
	return mp_additiveCorrections->getRawPointer()[binId];
}

float Corrector_CPU::getInVivoAttenuationFactor(bin_t binId) const
{
	ASSERT(mp_inVivoAttenuationFactors != nullptr &&
	       mp_inVivoAttenuationFactors->isMemoryValid());
	return mp_inVivoAttenuationFactors->getRawPointer()[binId];
}

const ProjectionData*
    Corrector_CPU::getCachedMeasurementsForAdditiveCorrectionFactors() const
{
	if (mp_additiveCorrections != nullptr &&
	    mp_additiveCorrections->isMemoryValid())
	{
		return mp_additiveCorrections->getReference();
	}
	return nullptr;
}

const ProjectionData*
    Corrector_CPU::getCachedMeasurementsForInVivoAttenuationFactors() const
{
	if (mp_inVivoAttenuationFactors != nullptr &&
	    mp_inVivoAttenuationFactors->isMemoryValid())
	{
		return mp_inVivoAttenuationFactors->getReference();
	}
	return nullptr;
}

