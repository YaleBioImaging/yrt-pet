/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/OSEMUpdater_CPU.hpp"

#include "datastruct/projection/ProjectionData.hpp"
#include "recon/Corrector_CPU.hpp"
#include "recon/OSEM_CPU.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ProgressDisplayMultiThread.hpp"


OSEMUpdater_CPU::OSEMUpdater_CPU(OSEM_CPU* pp_osem) : mp_osem(pp_osem)
{
	ASSERT(mp_osem != nullptr);
}

void OSEMUpdater_CPU::computeSensitivityImage(Image& destImage) const
{
	const OperatorProjector* projector = mp_osem->getProjector();
	const BinIterator* binIter = projector->getBinIter();
	const bin_t numBins = binIter->size();
	const Corrector_CPU& corrector = mp_osem->getCorrector_CPU();
	const Corrector_CPU* correctorPtr = &corrector;
	const ProjectionData* sensImgGenProjData =
	    corrector.getSensImgGenProjData();
	Image* destImagePtr = &destImage;
	Util::ProgressDisplayMultiThread progressDisplay(Globals::get_num_threads(),
	                                                 numBins);

#pragma omp parallel for default(none)                                      \
    firstprivate(sensImgGenProjData, correctorPtr, projector, destImagePtr, \
                     binIter, numBins) shared(progressDisplay)
	for (bin_t binIdx = 0; binIdx < numBins; binIdx++)
	{
		progressDisplay.progress(omp_get_thread_num(), 1);

		const bin_t bin = binIter->get(binIdx);

		const ProjectionProperties projectionProperties =
		    sensImgGenProjData->getProjectionProperties(bin);

		const float projValue = correctorPtr->getMultiplicativeCorrectionFactor(
		    *sensImgGenProjData, bin);

		projector->backProjection(destImagePtr, projectionProperties,
		                          projValue);
	}
}

void OSEMUpdater_CPU::computeEMUpdateImage(const Image& inputImage,
                                           Image& destImage) const
{
	const OperatorProjector* projector = mp_osem->getProjector();
	const BinIterator* binIter = projector->getBinIter();
	const bin_t numBins = binIter->size();
	const ProjectionData* measurements = mp_osem->getDataInput();
	const Corrector_CPU& corrector = mp_osem->getCorrector_CPU();
	const Corrector_CPU* correctorPtr = &corrector;
	const Image* inputImagePtr = &inputImage;
	Image* destImagePtr = &destImage;

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

#pragma omp parallel for default(none) firstprivate(                        \
        hasAdditiveCorrection, hasInVivoAttenuation, binIter, measurements, \
            projector, correctorPtr, destImagePtr, inputImagePtr, numBins)
	for (bin_t binIdx = 0; binIdx < numBins; binIdx++)
	{
		const bin_t bin = binIter->get(binIdx);

		const ProjectionProperties projectionProperties =
		    measurements->getProjectionProperties(bin);

		float update =
		    projector->forwardProjection(inputImagePtr, projectionProperties);

		if (hasAdditiveCorrection)
		{
			update += correctorPtr->getAdditiveCorrectionFactor(bin);
		}

		if (hasInVivoAttenuation)
		{
			update *= correctorPtr->getInVivoAttenuationFactor(bin);
		}

		if (update > 1e-8f)  // to prevent numerical instability
		{
			const float measurement = measurements->getProjectionValue(bin);

			update = measurement / update;

			projector->backProjection(destImagePtr, projectionProperties,
			                          update);
		}
	}
}
