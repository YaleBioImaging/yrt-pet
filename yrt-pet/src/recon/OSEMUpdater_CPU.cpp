/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/OSEMUpdater_CPU.hpp"

#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/recon/Corrector_CPU.hpp"
#include "yrt-pet/recon/OSEM_CPU.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"

#include <thread>


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
			    progressDisplay.incrementProgress(tid, 1);
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

void OSEMUpdater_CPU::iterate(ImageBase& destImage) const
{
	const OSEM_CPU* osem = mp_osem;

	// OSEM subsets
	for (int subsetId = 0; subsetId < osem->num_OSEM_subsets; subsetId++)
	{
		std::cout << "OSEM subset " << subsetId + 1 << "/"
		          << osem->num_OSEM_subsets << "..." << std::endl;

		mp_projector->setBinIter(getBinIterators()[p_subsetId].get());
		loadSubsetInternal(subsetId, true);

		// SET TMP VARIABLES TO 0
		getImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO)
		    ->setValue(0.0);
		if ((isLowRank && projectorParams.updateH) || dualUpdate)
		{
			initializeHBasisTmpBuffer();
			SyncHostToDeviceHBasisWrite();
		}

		ImageBase* mlemImage_rp;
		if (flagImagePSF)
		{
			getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF)
			    ->setValue(0.0);
			// PSF
			imagePsf->applyA(
			    getMLEMImageBuffer(),
			    getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF));
			mlemImage_rp =
			    getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF);
		}
		else
		{
			mlemImage_rp = getMLEMImageBuffer();
		}

		if (!projectorParams.updateH || dualUpdate)
		{
			computeEMUpdateImage(
			    *mlemImage_rp,
			    *getImageTmpBuffer(
			        TemporaryImageSpaceBufferType::EM_RATIO));
		}
		else
		{
			// When updating H, destImage must be the actual image (and not
			// a zeroed buffer) to retrieve the value from the image during
			// backupdate
			computeEMUpdateImage(*mlemImage_rp, *mlemImage_rp);
		}

		// PSF
		if (flagImagePSF)
		{
			getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF)
			    ->setValue(0.0);
			imagePsf->applyAH(
			    getImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO),
			    getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF));
		}

		// UPDATE
		if (!projectorParams.updateH || dualUpdate)
		{
			ImageBase* updateImage =
			    flagImagePSF ?
			        getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF) :
			        getImageTmpBuffer(
			            TemporaryImageSpaceBufferType::EM_RATIO);
			applyImageUpdate(getMLEMImageBuffer(), updateImage,
			                 getSensImageBuffer(), EPS_FLT, isDynamic);
		}
		if (projectorParams.updateH || (dualUpdate && iter > 0))
		{
			// TODO: This is suboptimal as the update could be done on GPU
			//  instead of copying to Device to do it
			applyHUpdate();
		}

		if (dualUpdate)
		{
			printf("\nUpdating LR Sensitivity image scaling...\n");
			// Sync with Device side values
			Sync_cWUpdateDeviceToHost();

			std::fill(m_cWUpdate.begin(), m_cWUpdate.end(), 0.0f);
			std::fill(m_cHUpdate.begin(), m_cHUpdate.end(), 0.0f);
			generateWUpdateSensScaling(m_cWUpdate.data());
			generateHUpdateSensScaling(m_cHUpdate.data());

			// Sync back new Host side values to device
			Sync_cWUpdateHostToDevice();
		}
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


	util::parallelForChunked(
	    numBins, numThreads,
	    [hasAdditiveCorrection, hasInVivoAttenuation, measurements,
	     inputImagePtr, consManager, projPropManager, collectInfoFlags,
	     correctorPtr, projector, destImagePtr, &binIter, &binFilter,
	     &constraintParams, &projectionProperties](size_t binIdx, int tid)
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
				    projector->backProjection(
				        destImagePtr, projectionProperties, update, tid);
			    }
		    }
	    });

	if ((mp_osem->getProjectorUpdaterType() ==
	     OperatorProjectorParams::ProjectorUpdaterType::LR) ||
	    (mp_osem->getProjectorUpdaterType() ==
	     OperatorProjectorParams::ProjectorUpdaterType::LRDUALUPDATE))
	{
		auto projectorH =
		    dynamic_cast<OperatorProjector*>(mp_osem->getProjectorPtr());
		ASSERT(projectorH != nullptr);
		auto updater =
		    dynamic_cast<OperatorProjectorUpdaterLR*>(projectorH->getUpdater());
		ASSERT(updater != nullptr);
		if (updater->getUpdateH())
		{
			updater->accumulateH();
		}
	}
}
}  // namespace yrt
