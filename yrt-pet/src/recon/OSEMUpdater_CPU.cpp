/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/OSEMUpdater_CPU.hpp"

#include "datastruct/projection/ProjectionData.hpp"
#include "recon/Corrector_CPU.hpp"
#include "recon/OSEM_CPU.hpp"
#include "utils/Assert.hpp"

OSEMUpdater_CPU::OSEMUpdater_CPU(OSEM_CPU* pp_osem) : mp_osem(pp_osem) {}

void OSEMUpdater_CPU::computeEMUpdateImage(const Image& inputImage,
                                           Image& destImage) const
{
	const OperatorProjector* projector = mp_osem->getProjector();
	const BinIterator* binIter = projector->getBinIter();
	const bin_t numBins = binIter->size();
	const ProjectionData* measurements = mp_osem->getDataInput();
	const Corrector_CPU& corrector = mp_osem->getCorrector_CPU();

	const bool hasAdditiveCorrection = corrector.hasAdditiveCorrection();
	const bool hasInVivoAttenuation = corrector.hasInVivoAttenuation();

	ASSERT(projector != nullptr);
	ASSERT(binIter != nullptr);
	ASSERT(measurements != nullptr);

	if (hasAdditiveCorrection)
	{
		ASSERT_MSG(
		    measurements ==
		        corrector.getCachedMeasurementsForAdditiveCorrectionFactors(),
		    "Additive corrections were not computed for this set of "
		    "measurements");
	}
	if (hasInVivoAttenuation)
	{
		ASSERT_MSG(
		    measurements ==
		        corrector.getCachedMeasurementsForInVivoAttenuationFactors(),
		    "In-vivo attenuation factors were not computed for this set of "
		    "measurements");
	}

	// TODO NOW: ADD Parallel
	for (bin_t binIdx = 0; binIdx < numBins; binIdx++)
	{
		const bin_t bin = binIter->get(binIdx);

		const ProjectionProperties projectionProperties =
		    measurements->getProjectionProperties(bin);

		float update =
		    projector->forwardProjection(&inputImage, projectionProperties);

		if (hasAdditiveCorrection)
		{
			update += corrector.getAdditiveCorrectionFactor(bin);
		}

		if (hasInVivoAttenuation)
		{
			update *= corrector.getInVivoAttenuationFactor(bin);
		}

		const float measurement = measurements->getProjectionValue(bin);

		update = measurement / update;

		projector->backProjection(&destImage, projectionProperties, update);
	}
}
