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

void OSEMUpdater_CPU::backprojectBin(
    int tid, const BinIteratorConstrained& binIter,
    const OperatorProjector& projector, const Corrector_CPU* correctorPtr,
    const ProjectionData& sensImgGenProjData, Image* destImagePtr,
    util::ProgressDisplayMultiThread& progressDisplay, size_t numBinsMax,
    const BinIterator& binIterProj, size_t blockSize,
    std::set<ConstraintVariable>& consVariables,
    std::set<ProjectionPropertyType>& projVariables,
    ConstraintParams* infoT) const
{
	ProjectionProperties projectionProperties;
	for (size_t binIdx = tid * blockSize;
	     binIdx < std::min((tid + 1) * blockSize, numBinsMax); binIdx++)
	{
		bin_t bin = binIterProj.get(binIdx);
		binIter.collectInfo(bin, consVariables, projVariables,
		                    projectionProperties, *infoT);
		if (binIter.isValid(*infoT))
		{
			progressDisplay.progress(tid, 1);
			const ProjectionProperties projectionProperties =
			    sensImgGenProjData.getProjectionProperties(bin);
			const float projValue =
			    correctorPtr->getMultiplicativeCorrectionFactor(
			        sensImgGenProjData, projectionProperties.bin);
			projector.backProjection(destImagePtr, projectionProperties,
			                         projValue);
		}
	}
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
	const BinIterator* binIterProj = projector->getBinIter();
	const Corrector_CPU& corrector = mp_osem->getCorrector_CPU();
	const Corrector_CPU* correctorPtr = &corrector;
	const ProjectionData* sensImgGenProjData =
	    corrector.getSensImgGenProjData();
	Image* destImagePtr = &destImage;
	BinIteratorConstrained binIter(sensImgGenProjData, mp_osem->getConstraints());
	const bin_t numBinsMax = binIterProj->size();
	int numThreads = globals::getNumThreads();
	util::ProgressDisplayMultiThread progressDisplay(globals::getNumThreads(),
	                                                 numBinsMax);

	std::vector<std::thread> workers;
	std::vector<ConstraintParams> info;
	std::set<ConstraintVariable> consVariables = binIter.collectVariables();
	std::set<ProjectionPropertyType> projVariables = {};
	ConstraintParams infoTest;
	ProjectionProperties projPropsTest;
	binIter.collectInfo(0, consVariables, projVariables, projPropsTest,
	                    infoTest);
	info.resize(numThreads);
	size_t blockSize = std::ceil(numBinsMax / (float)numThreads);
	for (int i = 0; i < numThreads; i++)
	{
		info[i].reserve(infoTest.size());
		workers.emplace_back(std::thread(
		    [&binIter, projector, correctorPtr, sensImgGenProjData,
		     destImagePtr, &progressDisplay, numBinsMax, binIterProj, blockSize,
		     &consVariables, &projVariables, this](ConstraintParams* infoT, int tid)
		    {
			    backprojectBin(tid, binIter, *projector, correctorPtr,
			                   *sensImgGenProjData, destImagePtr,
			                   progressDisplay, numBinsMax, *binIterProj,
			                   blockSize, consVariables, projVariables, infoT);
		    }, &info[i], i));
	}

	for (auto& worker : workers)
	{
		worker.join();
	}
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

	util::parallelForChunked(
	    numBins, globals::getNumThreads(),
	    [binIter, measurements, projector, inputImagePtr, hasAdditiveCorrection,
	     hasInVivoAttenuation, correctorPtr,
	     destImagePtr](bin_t binIdx, size_t tid)
	    {
		    const bin_t bin = binIter->get(binIdx);

		    const ProjectionProperties projectionProperties =
		        measurements->getProjectionProperties(bin);

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
			    const float measurement = measurements->getProjectionValue(bin);

			    update = measurement / update;

			    projector->backProjection(destImagePtr, projectionProperties,
			                              update, tid);
		    }
	    });
}
}  // namespace yrt
