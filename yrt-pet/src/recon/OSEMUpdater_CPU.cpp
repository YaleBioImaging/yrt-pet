/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/OSEMUpdater_CPU.hpp"

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

	auto binFilter = projector->getBinFilter();
	auto& projPropManager = binFilter->getPropertyManager();
	auto& consManager = binFilter->getConstraintManager();
	auto constraintParams = projector->getConstraintParams();
	auto projectionProperties = projector->getProjectionProperties();
	BinFilter::CollectInfoFlags collectInfoFlags(false);
	binFilter->collectFlags(collectInfoFlags);

	util::ProgressDisplayMultiThread progressDisplay(globals::getNumThreads(),
	                                                 numBinsMax);

	util::parallelForChunked(
	    numBinsMax, numThreads,
	    [blockSize, numBinsMax, sensImgGenProjData, consManager,
	     projPropManager, collectInfoFlags, correctorPtr, projector,
	     destImagePtr, &progressDisplay, &binIter, &binFilter,
	     &constraintParams, &projectionProperties](size_t binIdx, int tid)
	    {
		    bin_t bin = binIter->get(binIdx);
		    binFilter->collectInfo(bin, tid, tid, *sensImgGenProjData,
		                           collectInfoFlags, projectionProperties,
		                           constraintParams);
		    if (binFilter->isValid(consManager, constraintParams, tid))
		    {
			    progressDisplay.progress(tid, 1);
			    sensImgGenProjData->getProjectionProperties(
			        projectionProperties, projPropManager, bin, tid);
			    const float projValue =
			        correctorPtr->getMultiplicativeCorrectionFactor(
			            *sensImgGenProjData, bin);
			    projector->backProjection(destImagePtr, projectionProperties,
			                              projValue, tid);
		    }
	    });
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

	auto binFilter = projector->getBinFilter();
	auto& projPropManager = binFilter->getPropertyManager();
	auto& consManager = binFilter->getConstraintManager();
	auto constraintParams = projector->getConstraintParams();
	auto projectionProperties = projector->getProjectionProperties();
	BinFilter::CollectInfoFlags collectInfoFlags(false);
	binFilter->collectFlags(collectInfoFlags);

	std::vector<float> projVal;
	projVal.resize(numThreads);
	for (int c = 0; c < numThreads; c++)
	{
		projVal[c] = 0.f;
	}

	util::parallelForChunked(
	    numBins, numThreads,
	    [hasAdditiveCorrection, hasInVivoAttenuation, measurements,
	     inputImagePtr, consManager, projPropManager, collectInfoFlags,
	     correctorPtr, projector, destImagePtr, &binIter, &binFilter,
	     &constraintParams, &projectionProperties,
	     &projVal](size_t binIdx, int tid)
	    {
		    bin_t bin = binIter->get(binIdx);
		    binFilter->collectInfo(bin, tid, tid, *measurements,
		                           collectInfoFlags, projectionProperties,
		                           constraintParams);
		    if (binFilter->isValid(consManager, constraintParams, tid))
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
				    projVal[tid] += update;
				    projector->backProjection(
				        destImagePtr, projectionProperties, update, tid);
			    }
		    }
	    });

	float tot_ProjVal = 0.f;
	for (int c = 0; c < numThreads; ++c)
	{
		tot_ProjVal += projVal[c];
	}
	printf("\n DEBUG: tot_count = %lu, projVal: %1.f\n", numBins, tot_ProjVal);

	printf("Before if getProjectorUpdaterType");
	if ((mp_osem->getProjectorUpdaterType() ==
	     OperatorProjectorParams::ProjectorUpdaterType::LR) ||
	    (mp_osem->getProjectorUpdaterType() ==
	     OperatorProjectorParams::ProjectorUpdaterType::LRDUALUPDATE))
	{
		auto projectorH =
		    dynamic_cast<OperatorProjector*>(mp_osem->getProjectorPtr());
		auto updater =
		    dynamic_cast<OperatorProjectorUpdaterLR*>(projectorH->getUpdater());
		printf("After if getProjectorUpdaterType: %d", updater->getUpdateH());
		if (updater->getUpdateH())
		{
			{
				auto H_old = updater->getHBasisWrite();
				const auto dims = H_old.getDims();
				float sum = 0.f;
				for (size_t r = 0; r < dims[0]; ++r)
				{
					for (size_t t = 0; t < dims[1]; ++t)
					{
						sum += H_old[r][t];
					}
				}
				printf("\n Before accumulate: sum(H_tid) = %f \n", sum);
			}
			updater->accumulateH();
			{
				auto H_old = updater->getHBasisWrite();
				auto dims = H_old.getDims();
				float sum = 0.f;
				for (size_t r = 0; r < dims[0]; ++r)
				{
					for (size_t t = 0; t < dims[1]; ++t)
					{
						sum += H_old[r][t];
					}
				}
				printf("\n After accumulate: sum(H_tid) = %f \n", sum);
			}
		}
	}
}
}  // namespace yrt
