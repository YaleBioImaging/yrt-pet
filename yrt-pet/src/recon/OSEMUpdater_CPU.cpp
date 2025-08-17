/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/OSEMUpdater_CPU.hpp"

#include <omp.h>
#include <thread>

#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/recon/Corrector_CPU.hpp"
#include "yrt-pet/recon/OSEM_CPU.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"


namespace yrt
{

OSEMUpdater_CPU::OSEMUpdater_CPU(OSEM_CPU* pp_osem) : mp_osem(pp_osem)
{
	ASSERT(mp_osem != nullptr);
}

#if 0
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
	util::ProgressDisplayMultiThread progressDisplay(globals::getNumThreads(),
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
		                              projValue, tid);
	    });
}
#else
void OSEMUpdater_CPU::computeSensitivityImage(Image& destImage) const
{
	const OperatorProjector* projector = mp_osem->getProjector();
	const BinIterator* binIter = projector->getBinIter();
	const Corrector_CPU& corrector = mp_osem->getCorrector_CPU();
	const Corrector_CPU* correctorPtr = &corrector;
	const ProjectionData* sensImgGenProjData =
	    corrector.getSensImgGenProjData();
	Image* destImagePtr = &destImage;

	const bin_t numBinsMax = binIter->size();
	const int numThreads = globals::getNumThreads();
	const size_t blockSize = std::ceil(numBinsMax / (float)numThreads);

	auto binIterConstrained = projector->getBinIterContrained();
	auto& projPropManager = binIterConstrained->getPropertyManager();
	auto& consManager = binIterConstrained->getConstraintManager();
	auto constraintParams = projector->getConstraintParams();
	auto projectionProperties = projector->getProjectionProperties();

	util::ProgressDisplayMultiThread progressDisplay(globals::getNumThreads(),
	                                                 numBinsMax);

	util::parallel_do_indexed(
	    numThreads,
	    [blockSize, numBinsMax, sensImgGenProjData, consManager,
	     projPropManager, correctorPtr, projector, destImagePtr,
	     &progressDisplay, &binIter, &binIterConstrained, &constraintParams,
	     &projectionProperties](int tid)
	    {
		    for (size_t binIdx = tid * blockSize;
		         binIdx < std::min((tid + 1) * blockSize, numBinsMax); binIdx++)
		    {
			    bin_t bin = binIter->get(binIdx);
			    binIterConstrained->collectInfo(bin, tid, *sensImgGenProjData,
			                                    projectionProperties,
			                                    constraintParams);
			    if (binIterConstrained->isValid(consManager, constraintParams))
			    {
				    progressDisplay.progress(tid, 1);
				    sensImgGenProjData->getProjectionProperties(
				        projectionProperties, projPropManager, bin, tid);
				    const float projValue =
				        correctorPtr->getMultiplicativeCorrectionFactor(
				            *sensImgGenProjData, bin);
				    projector->backProjection(destImagePtr,
				                              projectionProperties, projValue);
			    }
		    }
	    });
}
#endif

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

	ASSERT(projector != nullptr);
	ASSERT(binIter != nullptr);
	ASSERT(measurements != nullptr);

	const bool hasAdditiveCorrection =
	    corrector.hasAdditiveCorrection(*measurements);
	const bool hasInVivoAttenuation = corrector.hasInVivoAttenuation();

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

	int numThreads = globals::getNumThreads();

	auto binIterConstrained = projector->getBinIterContrained();
	auto& projPropManager = binIterConstrained->getPropertyManager();
	auto& consManager = binIterConstrained->getConstraintManager();
	auto constraintParams = projector->getConstraintParams();
	auto projectionProperties = projector->getProjectionProperties();

	util::parallel_for_chunked(
	    numBins, numThreads,
	    [hasAdditiveCorrection, hasInVivoAttenuation, measurements,
	     inputImagePtr, consManager, projPropManager, correctorPtr, projector,
	     destImagePtr, &binIter, &binIterConstrained, &constraintParams,
	     &projectionProperties](int binIdx, int tid)
	    {
		    bin_t bin = binIter->get(binIdx);
		    binIterConstrained->collectInfo(bin, tid, *measurements,
		                                    projectionProperties,
		                                    constraintParams);
		    if (binIterConstrained->isValid(consManager, constraintParams))
		    {
			    measurements->getProjectionProperties(
			        projectionProperties, projPropManager, bin, tid);
			    float update = projector->forwardProjection(
			        inputImagePtr, projectionProperties, tid);
			    if (hasAdditiveCorrection)
			    {
				    update += correctorPtr->getAdditiveCorrectionFactor(bin);
			    }
			    if (hasInVivoAttenuation)
			    {
				    update *= correctorPtr->getInVivoAttenuationFactor(bin);
			    }
			    if (update > EPS_FLT)  // to prevent numerical instability
			    {
				    const float measurement =
				        measurements->getProjectionValue(bin);
				    update = measurement / update;
				    projector->backProjection(destImagePtr,
				                              projectionProperties, update);
			    }
		    }
	    });
}
}  // namespace yrt
