/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/OSEM_GPU.cuh"

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionListDevice.cuh"
#include "yrt-pet/operators/DDKernels.cuh"
#include "yrt-pet/operators/DeviceSynchronized.cuh"
#include "yrt-pet/operators/OperatorPsfDevice.cuh"
#include "yrt-pet/operators/OperatorVarPsfDevice.cuh"
#include "yrt-pet/operators/ProjectorWrapper.cuh"
#include "yrt-pet/operators/SiddonKernels.cuh"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_osem_gpu(pybind11::module& m)
{
	auto c = py::class_<OSEM_GPU, OSEM>(m, "OSEM_GPU");
}
}  // namespace yrt
#endif

namespace yrt
{

OSEM_GPU::OSEM_GPU(const Scanner& pr_scanner)
    : OSEM(pr_scanner),
      mpd_sensImageBuffer(nullptr),
      mpd_mlemImage(nullptr),
      mpd_tmpImage1(nullptr),
      mpd_tmpImage2(nullptr)
{
	std::cout << "Creating an instance of OSEM GPU..." << std::endl;

	mp_corrector = std::make_unique<Corrector_GPU>(pr_scanner);
}

void OSEM_GPU::addImagePSF(const std::string& p_imagePsf_fname,
                           ImagePSFMode p_imagePSFMode)
{
	ASSERT_MSG(!p_imagePsf_fname.empty(), "Empty filename for Image-space PSF");
	if (p_imagePSFMode == ImagePSFMode::UNIFORM)
	{
		imagePsf = std::make_unique<OperatorPsfDevice>(p_imagePsf_fname,
		                                               getMainStream());
		m_imagePSFMode = ImagePSFMode::UNIFORM;
	}
	else
	{
		ASSERT_MSG(imageParams.isValid(),
		           "For spatially variant PSF, image parameters have to be set "
		           "before calling addImagePSF");
		imagePsf = std::make_unique<OperatorVarPsfDevice>(
		    p_imagePsf_fname, imageParams, getMainStream());
		m_imagePSFMode = ImagePSFMode::VARIANT;
	}

	flagImagePSF = true;
}

void OSEM_GPU::setupProjectorForSensImgGen()
{
	ASSERT(projectorParams.numRays > 0);

	// Note: The projector updater for the sensitivity image generation is
	//  always DEFAULT4D

	// Create a projector object only to list the projection properties needed
	// Create ProjectorParams object only for sensitivity image, without TOF
	ProjectorParams paramsForSens(scanner);
	paramsForSens.projectorType = projectorParams.projectorType;
	paramsForSens.projPsf_fname = projectorParams.projPsf_fname;
	paramsForSens.numRays = projectorParams.numRays;
	paramsForSens.projPropertyTypesExtra =
	    projectorParams.projPropertyTypesExtra;
	mp_projector = Projector::create(paramsForSens);

	const auto projPsfFilename = paramsForSens.projPsf_fname;
	if (!projPsfFilename.empty())
	{
		mp_projPsfManager =
		    std::make_unique<ProjectionPsfManagerDevice>(projPsfFilename);
	}
}

void OSEM_GPU::allocateForSensImgGen()
{
	auto imageParamsSens = getImageParamsForSensitivityImage();

	// Allocate for image space
	mpd_sensImageBuffer =
	    std::make_unique<ImageDeviceOwned>(imageParamsSens, getMainStream());
	mpd_sensImageBuffer->allocate(true);

	if (flagImagePSF)
	{
		if (m_imagePSFMode == ImagePSFMode::UNIFORM)
		{
			const auto imagePsfDevice =
			    dynamic_cast<OperatorPsfDevice*>(imagePsf.get());
			ASSERT(imagePsfDevice != nullptr);
			// This is done in order to more accurately compute the available
			//  device memory for the projection-space buffers below
			imagePsfDevice->allocateTemporaryDeviceImageIfNeeded(
			    imageParamsSens, {getMainStream(), true});
		}
		else
		{
			const auto imagePsfDevice =
			    dynamic_cast<OperatorVarPsfDevice*>(imagePsf.get());
			ASSERT(imagePsfDevice != nullptr);
		}
		mpd_tmpImage2 = std::make_unique<ImageDeviceOwned>(imageParamsSens,
		                                                   getMainStream());
		mpd_tmpImage2->allocate(false);
	}

	if (mp_corrector->hasHardwareAttenuationImage())
	{
		const Image* attImage = mp_corrector->getHardwareAttenuationImage();
		mpd_tmpImage1 =
		    std::make_unique<ImageDeviceOwned>(attImage->getParams());
		mpd_tmpImage1->allocate();
		mpd_tmpImage1->copyFromHostImage(attImage, false);
	}

	// Allocate for projection space
	auto properties = getNeededProperties(false);
	auto constraints = getConstraintsAsVectorOfPointers();
	mp_binLoader = std::make_unique<BinLoader>(constraints, properties);
	mpd_propStruct =
	    std::make_unique<PropStructDevice<ProjectionPropertyType>>(properties);

	const size_t memAvailable_bytes = getMemAvailable(DefaultMemoryShare);
	const size_t elementSize_bytes =
	    mp_binLoader->getProjectionPropertiesElementSize();
	const size_t maxNumElementsInBatch =
	    memAvailable_bytes /
	    (globals::ThreadsPerBlockData * elementSize_bytes) *
	    globals::ThreadsPerBlockData;

	m_batchSetups.clear();
	m_batchSetups.reserve(num_OSEM_subsets);
	for (size_t subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		m_batchSetups.emplace_back(m_binIterators.at(subsetId)->size(),
		                           maxNumElementsInBatch);
	}
	// This will take the largest batch size of all batches in all subsets, but
	//  it will use the GPUBatchSetup object in doing so, which will clamp
	//  the batch size if the number of elements needed is lower than what the
	//  memory allows
	const size_t maxBatchSize = getMaxBatchSize();

	std::cout << "Allocating projection properties..." << std::endl;
	mp_binLoader->allocate(maxBatchSize);  // Host-side buffer
	mpd_propStruct->allocate(maxBatchSize,
	                         {getMainStream(), true});  // Device-side buffer
}

std::unique_ptr<Image> OSEM_GPU::generateSensitivityImageForCurrentSubset()
{
	ImageDevice& sensImage = *mpd_sensImageBuffer;
	const int currentSubset = getCurrentOSEMSubset();
	const int numBatchesInCurrentSubset = getNumBatchesInSubset(currentSubset);
	const cudaStream_t* mainStream = getMainStream();
	// const cudaStream_t* auxStream = getAuxStream();

	const float globalScale = mp_corrector->getGlobalScalingFactor();

	// Reset sens image buffer
	sensImage.fillDevice(0.0f, true);

	for (int batch = 0; batch < numBatchesInCurrentSubset; batch++)
	{
		std::cout << "Batch " << batch + 1 << "/" << numBatchesInCurrentSubset
		          << "..." << std::endl;

		precomputeBatchPropsForSensImgGen(currentSubset, batch);
		loadPrecomputedBatchPropsToDevice(currentSubset, batch,
		                                  {mainStream, true});

		generateSensImageForLoadedBatch(currentSubset, batch,
		                                {mainStream, false});
	}

	if (mainStream != nullptr)
	{
		cudaStreamSynchronize(*mainStream);
	}

	if (flagImagePSF)
	{
		// This will run in the main stream and will synchronize
		imagePsf->applyAH(mpd_sensImageBuffer.get(), mpd_tmpImage2.get());

		mpd_tmpImage2->applyThresholdDevice(mpd_sensImageBuffer.get(), EPS_FLT,
		                                    0.0f, 0.0f, 1.0f, 0.0f, true);
		mpd_sensImageBuffer.swap(mpd_tmpImage2);
	}

	// All voxels lower than "hardThreshold" will be put to 0
	// All voxels higher than "hardThreshold" will be multiplied by the global
	//  scale factor
	std::cout << "Applying threshold..." << std::endl;
	mpd_sensImageBuffer->applyThresholdDevice(mpd_sensImageBuffer.get(),
	                                          hardThreshold, 0.0f, 0.0f,
	                                          globalScale, 0.0f, true);

	// Transfert to host and return
	auto img = std::make_unique<ImageOwned>(sensImage.getParams());
	img->allocate();
	mpd_sensImageBuffer->transferToHostMemory(img.get(), true);

	return img;
}

void OSEM_GPU::endSensImgGen()
{
	// Clear temporary buffers
	mpd_sensImageBuffer = nullptr;
	mpd_tmpImage1 = nullptr;
	mpd_tmpImage2 = nullptr;
	mp_binLoader = nullptr;
	mpd_propStruct = nullptr;
}

void OSEM_GPU::setupForDynamicRecon()
{
	OSEM::setupForDynamicRecon();
}

void OSEM_GPU::setupProjectorForRecon()
{
	// This projector is not used to project. It is only used to gather the
	//  list of required properties
	mp_projector = Projector::create(projectorParams);

	// Projection-space PSF
	const auto projPsfFilename = projectorParams.projPsf_fname;
	if (!projPsfFilename.empty())
	{
		mp_projPsfManager =
		    std::make_unique<ProjectionPsfManagerDevice>(projPsfFilename);
	}

	// TOF
	const bool projParamsHasTOF = projectorParams.hasTOF();
	const bool dataInputHasTOF = getDataInput()->hasTOF();

	if (dataInputHasTOF)
	{
		ASSERT_MSG_WARNING(
		    projParamsHasTOF,
		    "Acquisition has TOF data, but no TOF was added to the PET model.");
	}

	if (projParamsHasTOF)
	{
		ASSERT_MSG(dataInputHasTOF,
		           "TOF was added to the PET model, but no TOF information is "
		           "available in the acquisition.");

		mp_tofHelper =
		    std::make_unique<DeviceSynchronizedObject<TimeOfFlightHelper>>(
		        projectorParams.getTOFWidth_ps(),
		        projectorParams.getTOFNumStd());
	}

	// Creates the device-side updater
	setupProjectorUpdater(projectorParams);
}

void OSEM_GPU::allocateForRecon()
{
	// Allocate image-space buffers

	mpd_mlemImage =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getMainStream());
	mpd_tmpImage1 =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getMainStream());
	mpd_sensImageBuffer =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getMainStream());

	mpd_mlemImage->allocate(false);
	mpd_tmpImage1->allocate(false);
	mpd_sensImageBuffer->allocate(false);

	if (flagImagePSF)
	{
		mpd_tmpImage2 = std::make_unique<ImageDeviceOwned>(getImageParams(),
		                                                   getMainStream());
		mpd_tmpImage2->allocate(false);
	}

	// Initialize the MLEM image values to non-zero
	if (initialEstimate != nullptr)
	{
		mpd_mlemImage->copyFromImage(initialEstimate);
	}
	else
	{
		mpd_mlemImage->fill(INITIAL_VALUE_MLEM);
	}

	// Apply mask image (Use temporary buffer to avoid allocating a new one
	// unnecessarily)
	if (maskImage != nullptr)
	{
		mpd_tmpImage1->copyFromHostImage(maskImage, true);
	}
	else if (num_OSEM_subsets == 1 || usingListModeInput)
	{
		// No need to sum all sensitivity images, just use the only one
		mpd_tmpImage1->copyFromHostImage(getSensitivityImage(0), true);
	}
	else
	{
		std::cout << "Summing sensitivity images to generate mask image..."
		          << std::endl;

		for (int i = 0; i < num_OSEM_subsets; ++i)
		{
			mpd_sensImageBuffer->copyFromHostImage(getSensitivityImage(i),
			                                       false);
			mpd_sensImageBuffer->addFirstImageToSecondDevice(
			    mpd_tmpImage1.get(), false);
		}
	}
	mpd_mlemImage->applyThresholdBroadcastDevice(mpd_tmpImage1.get(), 0.0f,
	                                             0.0f, 0.0f, 1.0f, 0.0f, false);
	mpd_tmpImage1->fillDevice(0.0f, false);

	// Initialize device's sensitivity image with the host's
	if (usingListModeInput)
	{
		mpd_sensImageBuffer->transferToDeviceMemory(getSensitivityImage(0),
		                                            true);
	}

	// Allocate for projection space

	// YN: Opportunity for optimisation: if the bin loader is already
	//  initialized and allocated, and the properties held are the same as
	//  what comes out of "getNeededProperties(true)", and the allocated size is
	//  larger or equal than what is needed for the reconstruction, then there
	//  is no need to re-initialize and re-allocate the bin loader.
	//  The same would go for "mpd_propStruct" since it follows the same size as
	//  the bin loader.

	// Make sure the corrector buffer is properly defined
	const ProjectionData* dataInput = getDataInput();
	mp_corrector->precomputeCorrectionFactors(*dataInput);

	// Prepare property struct
	auto properties = getNeededProperties(true);
	const std::vector<Constraint*> constraints =
	    getConstraintsAsVectorOfPointers();
	mp_binLoader = std::make_unique<BinLoader>(constraints, properties);
	mpd_propStruct =
	    std::make_unique<PropStructDevice<ProjectionPropertyType>>(properties);

	// Compute the maximum possible elements in a batch
	const size_t memAvailable_bytes = getMemAvailable(DefaultMemoryShare);
	const unsigned int elementSize_bytes =
	    mp_binLoader->getProjectionPropertiesElementSize();
	const size_t maxNumElementsInBatch =
	    memAvailable_bytes /
	    (globals::ThreadsPerBlockData * elementSize_bytes) *
	    globals::ThreadsPerBlockData;

	m_batchSetups.clear();
	m_batchSetups.reserve(num_OSEM_subsets);
	for (size_t subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		m_batchSetups.emplace_back(m_binIterators.at(subsetId)->size(),
		                           maxNumElementsInBatch);
	}

	const size_t maxBatchSize = getMaxBatchSize();

	std::cout << "Allocating projection properties..." << std::endl;
	mp_binLoader->allocate(maxBatchSize);  // Host-side buffer
	mpd_propStruct->allocate(maxBatchSize,
	                         {getMainStream(), true});  // Device-side buffer
}

void OSEM_GPU::loadCurrentSubset(bool forRecon)
{
	// If doing recontruction with a histogram, transfer the correct sensitivity
	//  image to the device, since there is one for every subset.
	if (forRecon && !usingListModeInput)
	{
		// Loading the right sensitivity image to the device
		mpd_sensImageBuffer->transferToDeviceMemory(
		    getSensitivityImage(getCurrentOSEMSubset()), true);
	}
}

void OSEM_GPU::resetEMUpdateImage()
{
	mpd_tmpImage1->fill(0.0);
}

void OSEM_GPU::computeEMUpdateImage()
{
	// Forward PSF
	if (flagImagePSF)
	{
		mpd_tmpImage2->fill(0.0);
		imagePsf->applyA(mpd_mlemImage.get(), mpd_tmpImage2.get());

		// We swap here so that the "mlemImage" buffer stores the MLEM image
		//  with the PSF applied to it and the "tmpImage2" buffer stores the
		//  original MLEM image.
		mpd_mlemImage.swap(mpd_tmpImage2);
	}

	const int currentSubset = getCurrentOSEMSubset();
	const int numBatchesInCurrentSubset = getNumBatchesInSubset(currentSubset);

	const cudaStream_t* mainStream = getMainStream();
	// const cudaStream_t* auxStream = getAuxStream();

	for (int batch = 0; batch < numBatchesInCurrentSubset; batch++)
	{
		std::cout << "Batch " << batch + 1 << "/" << numBatchesInCurrentSubset
		          << "..." << std::endl;

		precomputeBatchPropsForRecon(currentSubset, batch);
		loadPrecomputedBatchPropsToDevice(currentSubset, batch,
		                                  {mainStream, true});

		computeEMUpdateImageForLoadedBatch(currentSubset, batch,
		                                   {mainStream, false});
	}

	if (mainStream != nullptr)
	{
		cudaStreamSynchronize(*mainStream);
	}

	// Backward PSF
	if (flagImagePSF)
	{
		// We swap again here so that the "mlemImage" buffer gets back to
		// storing the original
		//  MLEM image.
		//  This way, we can use the "tmpImage2" buffer to compute the PSF
		mpd_tmpImage2.swap(mpd_mlemImage);

		// YN: Is this initialization necessary ?
		mpd_tmpImage2->fill(0.0);
		imagePsf->applyAH(mpd_tmpImage1.get(), mpd_tmpImage2.get());

		// We swap these two buffers so that we can use "tmpImage1" to
		//  apply the image update
		mpd_tmpImage1.swap(mpd_tmpImage2);
	}
}

void OSEM_GPU::applyImageUpdate()
{
	mpd_mlemImage->updateEMThresholdDynamicDevice(
	    mpd_tmpImage1.get(), mpd_sensImageBuffer.get(), EPS_FLT, true);
}

void OSEM_GPU::completeSubset() {}

void OSEM_GPU::completeMLEMIteration() {}

void OSEM_GPU::endRecon()
{
	ASSERT(outImage != nullptr);

	// Transfer MLEM image Device to host
	mpd_mlemImage->transferToHostMemory(outImage.get(), true);

	// Clear temporary buffers
	mpd_mlemImage = nullptr;
	mpd_tmpImage1 = nullptr;
	mpd_tmpImage2 = nullptr;
	mpd_sensImageBuffer = nullptr;
	mp_corrector->resetAllPrecomputedFactors();
	mp_binLoader = nullptr;
	mpd_propStruct = nullptr;
}

const cudaStream_t* OSEM_GPU::getAuxStream() const
{
	return &m_auxStream.getStream();
}

const cudaStream_t* OSEM_GPU::getMainStream() const
{
	return &m_mainStream.getStream();
}

ImageBase* OSEM_GPU::getSensImageBuffer()
{
	return mpd_sensImageBuffer.get();
}

ImageBase* OSEM_GPU::getMLEMImageBuffer()
{
	return mpd_mlemImage.get();
}

ImageBase* OSEM_GPU::getEMUpdateImageBuffer()
{
	return mpd_tmpImage1.get();
}

const Corrector& OSEM_GPU::getCorrector() const
{
	return *mp_corrector;
}

Corrector& OSEM_GPU::getCorrector()
{
	return *mp_corrector;
}

void OSEM_GPU::setupProjectorUpdater(const ProjectorParams& params)
{
	// If using DEFAULT4D, leave the updater tp nullptr
	if (params.updaterType != UpdaterType::DEFAULT4D)
	{
		m_updaterContainer.initUpdater(params.updaterType);
	}
}

size_t OSEM_GPU::getMemAvailable(float shareOfMemoryToUse) const
{
	size_t memAvailable = globals::getAvailableVRAM(true);

	// Shrink memory according to the portion we want to use
	memAvailable = static_cast<size_t>(static_cast<float>(memAvailable) *
	                                   shareOfMemoryToUse);
	return memAvailable;
}

size_t OSEM_GPU::getBatchSize(int subsetId, int batchId) const
{
	return m_batchSetups[subsetId].getBatchSize(batchId);
}

size_t OSEM_GPU::getMaxBatchSize() const
{
	// Get the largest batch size in each subset's batch setup
	// The batch 0 is always the largest in the GPUBatchSetup
	size_t maxBatchSize = 0ull;
	for (size_t i = 0ull; i < m_batchSetups.size(); i++)
	{
		const size_t currBatchSize = m_batchSetups[i].getBatchSize(0);
		maxBatchSize = std::max(currBatchSize, maxBatchSize);
	}
	return maxBatchSize;
}

int OSEM_GPU::getNumBatchesInSubset(int subsetId) const
{
	return m_batchSetups[subsetId].getNumBatches();
}

void OSEM_GPU::precomputeBatchPropsForSensImgGen(int subsetId, int batchId)
{
	const int numThreads = globals::getNumThreads();

	// Create batched bin iterator
	const size_t batchSize = getBatchSize(subsetId, batchId);
	const size_t offset = batchId * getBatchSize(subsetId, 0);
	const BinIterator* binIter = getBinIterator(subsetId);
	const BinIteratorBatched binIterBatched{binIter, offset, batchSize};

	// This is the histogram used for the sensitivity image generation.
	//  It's either a uniform histogram, a sensitivity histogram, or an ACF
	//  histogram. It is used *only to gather the LORs*.
	const ProjectionData* sensImgGenProjData =
	    mp_corrector->getSensImgGenProjData();

	// Properties structure
	PropertyUnit* props = mp_binLoader->getProjectionPropertiesRawPointer();
	const ProjectionPropertyManager& propManager =
	    mp_binLoader->getPropertyManager();
	PropStruct<ProjectionPropertyType>* propStruct =
	    mp_binLoader->getProjectionPropStruct();
	size_t propElementSize = propStruct->getElementSize();

	// For constraints
	BinFilter::CollectInfoFlags collectInfoFlags(false);
	mp_binLoader->collectFlags(collectInfoFlags);

	const bool hasSensitivity =
	    propManager.has(ProjectionPropertyType::SENSITIVITY);
	const bool hasHardwareAttenuationFromHistogram =
	    mp_corrector->doesHardwareACFComeFromHistogram();

	util::parallelForChunked(
	    batchSize, numThreads,
	    [&binIterBatched, sensImgGenProjData, this, &propManager, props,
	     propStruct, propElementSize, hasSensitivity,
	     hasHardwareAttenuationFromHistogram,
	     &collectInfoFlags](size_t batchPos, int tid)
	    {
		    const bin_t bin = binIterBatched.get(batchPos);

		    mp_binLoader->collectInfo(bin, batchPos, tid, *sensImgGenProjData,
		                              collectInfoFlags);
		    if (mp_binLoader->verifyConstraints(tid))
		    {
			    // Gather LOR and projector-related properties
			    sensImgGenProjData->collectProjectionProperties(
			        propManager, props, batchPos, bin);

			    // Gather sensitivity and attenuation
			    if (hasSensitivity || hasHardwareAttenuationFromHistogram)
			    {
				    const histo_bin_t histoBin =
				        sensImgGenProjData->getHistogramBin(bin);

				    if (hasSensitivity)
				    {
					    propStruct->setValue(
					        batchPos, ProjectionPropertyType::SENSITIVITY,
					        mp_corrector->getSensitivity(histoBin));
				    }

				    if (hasHardwareAttenuationFromHistogram)
				    {
					    propStruct->setValue(
					        batchPos, ProjectionPropertyType::ATTENUATION,
					        mp_corrector->getHardwareACFFromHistogram(
					            histoBin));
				    }
			    }
		    }
		    else
		    {
			    // Invalid bin/event. Set properties to zeros
			    PropertyUnit* popStructPtrAtPos =
			        props + batchPos * propElementSize;
			    std::memset(popStructPtrAtPos, 0, propElementSize);
		    }
	    });
}

void OSEM_GPU::precomputeBatchPropsForRecon(int subsetId, int batchId)
{
	const int numThreads = globals::getNumThreads();

	// Create batched bin iterator
	const size_t batchSize = getBatchSize(subsetId, batchId);
	const size_t offset = batchId * getBatchSize(subsetId, 0);
	const BinIterator* binIter = getBinIterator(subsetId);
	const BinIteratorBatched binIterBatched{binIter, offset, batchSize};

	// Projection data used for gathering the LORs and measurements
	const ProjectionData* dataInput = getDataInput();
	ASSERT(dataInput != nullptr);
	ASSERT(dataInput->count() > 1);

	const bool isUniform = dataInput->isUniform();

	// Properties structure
	PropertyUnit* props = mp_binLoader->getProjectionPropertiesRawPointer();
	const ProjectionPropertyManager& propManager =
	    mp_binLoader->getPropertyManager();
	PropStruct<ProjectionPropertyType>* propStruct =
	    mp_binLoader->getProjectionPropStruct();
	size_t propElementSize = propStruct->getElementSize();

	// For constraints
	BinFilter::CollectInfoFlags collectInfoFlags(false);
	mp_binLoader->collectFlags(collectInfoFlags);

	const bool hasSensitivity =
	    propManager.has(ProjectionPropertyType::SENSITIVITY);
	const bool hasAttenuation =
	    propManager.has(ProjectionPropertyType::ATTENUATION);
	const bool hasRandomsEstimate =
	    propManager.has(ProjectionPropertyType::RANDOMS_ESTIMATE);
	const bool hasScatterEstimate =
	    propManager.has(ProjectionPropertyType::SCATTER_ESTIMATE);
	const bool hasAttenuationPrecorrection =
	    propManager.has(ProjectionPropertyType::ATTENUATION_PRECORRECTION);

	util::parallelForChunked(
	    batchSize, numThreads,
	    [&binIterBatched, dataInput, this, &propManager, props, propStruct,
	     isUniform, propElementSize, hasSensitivity, hasAttenuation,
	     hasRandomsEstimate, hasScatterEstimate, hasAttenuationPrecorrection,
	     &collectInfoFlags](size_t batchPos, int tid)
	    {
		    const bin_t bin = binIterBatched.get(batchPos);

		    mp_binLoader->collectInfo(bin, batchPos, tid, *dataInput,
		                              collectInfoFlags);
		    if (mp_binLoader->verifyConstraints(tid))
		    {
			    // Gather LOR and projector-related properties
			    dataInput->collectProjectionProperties(propManager, props,
			                                           batchPos, bin);

			    // Gather measurement if necessary
			    if (!isUniform)
			    {
				    propStruct->setValue(batchPos,
				                         ProjectionPropertyType::MEASUREMENT,
				                         dataInput->getProjectionValue(bin));
			    }

			    // Gather correction factors
			    if (hasSensitivity)
			    {
				    propStruct->setValue(
				        batchPos, ProjectionPropertyType::SENSITIVITY,
				        mp_corrector->getPrecomputedSensitivityFactor(bin));
			    }
			    if (hasAttenuation)
			    {
				    propStruct->setValue(
				        batchPos, ProjectionPropertyType::ATTENUATION,
				        mp_corrector->getPrecomputedAttenuationFactor(bin));
			    }
			    if (hasRandomsEstimate)
			    {
				    propStruct->setValue(
				        batchPos, ProjectionPropertyType::RANDOMS_ESTIMATE,
				        mp_corrector->getPrecomputedRandomsEstimate(bin));
			    }
			    if (hasScatterEstimate)
			    {
				    propStruct->setValue(
				        batchPos, ProjectionPropertyType::SCATTER_ESTIMATE,
				        mp_corrector->getPrecomputedScatterEstimate(bin));
			    }
			    if (hasAttenuationPrecorrection)
			    {
				    propStruct->setValue(
				        batchPos,
				        ProjectionPropertyType::ATTENUATION_PRECORRECTION,
				        mp_corrector->getPrecomputedInVivoAttenuationFactor(
				            bin));
			    }
		    }
		    else
		    {
			    // Invalid bin/event. Set properties to zeros
			    PropertyUnit* popStructPtrAtPos =
			        props + batchPos * propElementSize;
			    std::memset(popStructPtrAtPos, 0, propElementSize);
		    }
	    });
}

void OSEM_GPU::loadPrecomputedBatchPropsToDevice(int subsetId, int batchId,
                                                 GPULaunchConfig launchConfig)
{
	// Batch size in number of elements
	const size_t batchSize = getBatchSize(subsetId, batchId);

	// Copy properties from host to device
	mpd_propStruct->copyFromHost(*mp_binLoader->getProjectionPropStruct(),
	                             launchConfig, batchSize);
}

void OSEM_GPU::generateSensImageForLoadedBatch(int subsetId, int batchId,
                                               GPULaunchConfig launchConfig)
{
	const size_t batchSize = getBatchSize(subsetId, batchId);

	GPULaunchParams launchParams = util::initiateDeviceParameters(batchSize);

	CUImage sensImage = DeviceSynchronized::getCUImage(*mpd_sensImageBuffer);

	// The device pointer inside will be initialized to null if there is no
	//  attenuation image
	CUImage attImage{};
	if (mp_corrector->hasHardwareAttenuationImage())
	{
		ASSERT(mpd_tmpImage1 != nullptr);
		attImage = DeviceSynchronized::getCUImage(*mpd_tmpImage1);
	}

	const ProjectionPropertyManager* pd_projPropManager =
	    mpd_propStruct->getManagerDevicePointer();
	const PropertyUnit* pd_projectionProperties =
	    mpd_propStruct->getDevicePointer();

	// The device pointer inside will also be null if there's no
	//  projection-space PSF
	ProjectionPsfKernelStruct projPsfKernelStruct;
	if (mp_projPsfManager != nullptr)
	{
		projPsfKernelStruct.properties =
		    mp_projPsfManager->getProjectionPsfProperties();
		projPsfKernelStruct.kernels =
		    mp_projPsfManager->getKernelsDevicePointer();
	}

	const CUScannerParams scannerParams =
	    DeviceSynchronized::getCUScannerParams(scanner);
	const int numRays = projectorParams.numRays;

	if (launchConfig.stream != nullptr)
	{
		generateSensImage_kernel<<<launchParams.gridSize,
		                           launchParams.blockSize, 0,
		                           *launchConfig.stream>>>(
		    sensImage, attImage, pd_projPropManager, pd_projectionProperties,
		    projPsfKernelStruct, scannerParams, numRays,
		    projectorParams.projectorType, batchSize);
	}
	else
	{
		generateSensImage_kernel<<<launchParams.gridSize,
		                           launchParams.blockSize>>>(
		    sensImage, attImage, pd_projPropManager, pd_projectionProperties,
		    projPsfKernelStruct, scannerParams, numRays,
		    projectorParams.projectorType, batchSize);
	}

	synchronizeIfNeeded(launchConfig);
}

void OSEM_GPU::computeEMUpdateImageForLoadedBatch(int subsetId, int batchId,
                                                  GPULaunchConfig launchConfig)
{
	const size_t batchSize = getBatchSize(subsetId, batchId);
	GPULaunchParams launchParams = util::initiateDeviceParameters(batchSize);

	const ProjectionData* dataInput = getDataInput();

	auto inputImageForForwardProj_ptr =
	    dynamic_cast<ImageDevice*>(getMLEMImageBuffer());
	ASSERT(inputImageForForwardProj_ptr != nullptr);
	ASSERT(inputImageForForwardProj_ptr->isMemoryValid());
	CUImage inputImageForForwardProj =
	    DeviceSynchronized::getCUImage(*inputImageForForwardProj_ptr);

	auto destImageForBackproj_ptr =
	    dynamic_cast<ImageDevice*>(getEMUpdateImageBuffer());
	ASSERT(destImageForBackproj_ptr != nullptr);
	ASSERT(destImageForBackproj_ptr->isMemoryValid());
	CUImage destImageForBackproj =
	    DeviceSynchronized::getCUImage(*destImageForBackproj_ptr);

	const float globalScaleFactor = mp_corrector->getGlobalScalingFactor();
	const float measurementUniformValue = dataInput->getProjectionValue(0);

	const ProjectionPropertyManager* pd_projPropManager =
	    mpd_propStruct->getManagerDevicePointer();
	const PropertyUnit* pd_projectionProperties =
	    mpd_propStruct->getDevicePointer();

	UpdaterPointer pd_updater = m_updaterContainer.getUpdaterDevicePointer();

	const TimeOfFlightHelper* pd_tofHelper = nullptr;
	if (mp_tofHelper != nullptr)
	{
		pd_tofHelper = mp_tofHelper->getDevicePointer();
	}

	// The device pointer inside will also be null if there's no
	//  projection-space PSF
	ProjectionPsfKernelStruct projPsfKernelStruct;
	if (mp_projPsfManager != nullptr)
	{
		projPsfKernelStruct.properties =
		    mp_projPsfManager->getProjectionPsfProperties();
		projPsfKernelStruct.kernels =
		    mp_projPsfManager->getKernelsDevicePointer();
	}

	const CUScannerParams scannerParams =
	    DeviceSynchronized::getCUScannerParams(scanner);
	const int numRays = projectorParams.numRays;

	if (launchConfig.stream != nullptr)
	{
		if (pd_updater != nullptr)
		{
			computeEMUpdateImage_kernel<true>
			    <<<launchParams.gridSize, launchParams.blockSize, 0,
			       *launchConfig.stream>>>(
			        inputImageForForwardProj, destImageForBackproj,
			        globalScaleFactor, measurementUniformValue,
			        pd_projPropManager, pd_projectionProperties, pd_updater,
			        pd_tofHelper, projPsfKernelStruct, scannerParams, numRays,
			        projectorParams.projectorType, batchSize);
		}
		else
		{
			computeEMUpdateImage_kernel<false>
			    <<<launchParams.gridSize, launchParams.blockSize, 0,
			       *launchConfig.stream>>>(
			        inputImageForForwardProj, destImageForBackproj,
			        globalScaleFactor, measurementUniformValue,
			        pd_projPropManager, pd_projectionProperties,
			        nullptr /*No updater*/, pd_tofHelper, projPsfKernelStruct,
			        scannerParams, numRays, projectorParams.projectorType,
			        batchSize);
		}
	}
	else
	{
		if (pd_updater != nullptr)
		{
			computeEMUpdateImage_kernel<true>
			    <<<launchParams.gridSize, launchParams.blockSize>>>(
			        inputImageForForwardProj, destImageForBackproj,
			        globalScaleFactor, measurementUniformValue,
			        pd_projPropManager, pd_projectionProperties, pd_updater,
			        pd_tofHelper, projPsfKernelStruct, scannerParams, numRays,
			        projectorParams.projectorType, batchSize);
		}
		else
		{
			computeEMUpdateImage_kernel<false>
			    <<<launchParams.gridSize, launchParams.blockSize>>>(
			        inputImageForForwardProj, destImageForBackproj,
			        globalScaleFactor, measurementUniformValue,
			        pd_projPropManager, pd_projectionProperties,
			        nullptr /*No updater*/, pd_tofHelper, projPsfKernelStruct,
			        scannerParams, numRays, projectorParams.projectorType,
			        batchSize);
		}
	}

	synchronizeIfNeeded(launchConfig);
}

std::set<ProjectionPropertyType>
    OSEM_GPU::getNeededProperties(bool forRecon) const
{
	// Gather properties
	std::set<ProjectionPropertyType> properties;

	if (mp_corrector->hasSensitivityHistogram())
	{
		properties.insert(ProjectionPropertyType::SENSITIVITY);
	}

	if (!forRecon)
	{
		// Sensitivity image generation uses hardware attenuation
		if (mp_corrector->doesHardwareACFComeFromHistogram())
		{
			properties.insert(ProjectionPropertyType::ATTENUATION);
		}
	}

	if (forRecon)
	{
		if (!getDataInput()->isUniform())
		{
			// if "isUniform" is false, it means that the projection value
			//  differs between bins/events. Therefore, we need to load it
			//  in the prop struct.
			properties.insert(ProjectionPropertyType::MEASUREMENT);
		}

		// Reconstruction uses total attenuation
		if (mp_corrector->hasAttenuation())
		{
			properties.insert(ProjectionPropertyType::ATTENUATION);
		}

		const ProjectionData* dataInput = getDataInput();
		if (dataInput->hasDynamicFraming())
		{
			properties.insert(ProjectionPropertyType::DYNAMIC_FRAME);
		}
		if (dataInput->hasTOF() && projectorParams.hasTOF())
		{
			properties.insert(ProjectionPropertyType::TOF);
		}
		if (mp_corrector->hasRandomsEstimates(*dataInput))
		{
			properties.insert(ProjectionPropertyType::RANDOMS_ESTIMATE);
		}
		if (mp_corrector->hasScatterEstimates())
		{
			properties.insert(ProjectionPropertyType::SCATTER_ESTIMATE);
		}
		if (mp_corrector->hasInVivoAttenuation())
		{
			properties.insert(
			    ProjectionPropertyType::ATTENUATION_PRECORRECTION);
		}
	}
	properties.merge(mp_projector->getProjectionPropertyTypes());

	return properties;
}

__global__ void generateSensImage_kernel(
    CUImage sensImage, CUImage attImage,
    const ProjectionPropertyManager* pd_projPropManager,
    const PropertyUnit* pd_projectionProperties,
    ProjectionPsfKernelStruct projPsfKernelStruct,
    CUScannerParams scannerParams, int numRays, ProjectorType projectorType,
    size_t batchSize)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;

	if (eventId < batchSize)
	{
		const bool hasSensitivityFactor =
		    pd_projPropManager->has(ProjectionPropertyType::SENSITIVITY);
		const bool hasAttFactor =
		    pd_projPropManager->has(ProjectionPropertyType::ATTENUATION);
		const bool needToForwProjAtt = attImage.devicePointer != nullptr;

		// Gather LOR
		const float* lor = pd_projPropManager->getDataPtr<float>(
		    pd_projectionProperties, eventId, ProjectionPropertyType::LOR);
		static_assert(sizeof(Line3D) == sizeof(float3) * 2);
		const float3 p1Init{lor[0], lor[1], lor[2]};
		const float3 p2Init{lor[3], lor[4], lor[5]};

		// Subtract image offset
		float3 p1 = p1Init;
		float3 p2 = p2Init;
		p1.x -= sensImage.params.off_x;
		p1.y -= sensImage.params.off_y;
		p1.z -= sensImage.params.off_z;
		p2.x -= sensImage.params.off_x;
		p2.y -= sensImage.params.off_y;
		p2.z -= sensImage.params.off_z;

		float toBackproject = 1.0f;

		if (hasSensitivityFactor)
		{
			const float sensitivity = pd_projPropManager->getDataValue<float>(
			    pd_projectionProperties, eventId,
			    ProjectionPropertyType::SENSITIVITY);
			toBackproject = sensitivity;
		}

		if (hasAttFactor)
		{
			// Precomputed attenuation factor
			const float attenuationFactor =
			    pd_projPropManager->getDataValue<float>(
			        pd_projectionProperties, eventId,
			        ProjectionPropertyType::ATTENUATION);
			toBackproject *= attenuationFactor;
		}
		else if (needToForwProjAtt)
		{
			float attenuationFactor = 0.0f;

			// This time, subtract the offset of the attenuation image
			p1 = p1Init;
			p2 = p2Init;
			p1.x -= attImage.params.off_x;
			p1.y -= attImage.params.off_y;
			p1.z -= attImage.params.off_z;
			p2.x -= attImage.params.off_x;
			p2.y -= attImage.params.off_y;
			p2.z -= attImage.params.off_z;

			// Orientation is left blank since it is not used by the single-ray
			//  Siddon projection, which is what is used for forward-projecting
			//  on the attenuation image
			const float3 n1 = {};
			const float3 n2 = {};

			projectSiddon<true, false, true, false, false>(
			    attenuationFactor, attImage.devicePointer, nullptr, p1, p2, n1,
			    n2, 0, nullptr, 0.0f, scannerParams, attImage.params, 1,
			    eventId);

			attenuationFactor =
			    util::getAttenuationCoefficientFactor(attenuationFactor);
			toBackproject *= attenuationFactor;
		}

		// Gather detector orientation if available
		float3 n1 = {};
		float3 n2 = {};
		if (pd_projPropManager->has(ProjectionPropertyType::DET_ORIENT))
		{
			const float* detOrient = pd_projPropManager->getDataPtr<float>(
			    pd_projectionProperties, eventId,
			    ProjectionPropertyType::DET_ORIENT);
			n1 = {detOrient[0], detOrient[1], detOrient[2]};
			n2 = {detOrient[3], detOrient[4], detOrient[5]};
		}

		frame_t dynamicFrame = 0;
		if (pd_projPropManager->has(ProjectionPropertyType::DYNAMIC_FRAME))
		{
			dynamicFrame = pd_projPropManager->getDataValue<frame_t>(
			    pd_projectionProperties, eventId,
			    ProjectionPropertyType::DYNAMIC_FRAME);
		}

		projectAny<false, false, false>(toBackproject, sensImage, nullptr, p1,
		                                p2, n1, n2, dynamicFrame, nullptr, 0.0f,
		                                projPsfKernelStruct, scannerParams,
		                                numRays, eventId, projectorType);
	}
}

template <bool UseUpdater>
__global__ void computeEMUpdateImage_kernel(
    CUImage forwImage, CUImage emImage, float globalScaleFactor,
    float measurementUniformValue,
    const ProjectionPropertyManager* pd_projPropManager,
    const PropertyUnit* pd_projectionProperties, UpdaterPointer pd_updater,
    const TimeOfFlightHelper* pd_tofHelper,
    ProjectionPsfKernelStruct projPsfKernelStruct,
    CUScannerParams scannerParams, int numRays, ProjectorType projectorType,
    size_t batchSize)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;

	if (eventId < batchSize)
	{
		const bool hasMeasurement =  // Non-uniform measurements
		    pd_projPropManager->has(ProjectionPropertyType::MEASUREMENT);
		const bool hasSensitivityFactor =
		    pd_projPropManager->has(ProjectionPropertyType::SENSITIVITY);
		const bool hasAttenuationFactor =
		    pd_projPropManager->has(ProjectionPropertyType::ATTENUATION);
		const bool hasRandomsEstimate =
		    pd_projPropManager->has(ProjectionPropertyType::RANDOMS_ESTIMATE);
		const bool hasScatterEstimate =
		    pd_projPropManager->has(ProjectionPropertyType::SCATTER_ESTIMATE);
		const bool hasAttenuationPrecorrectionFactor = pd_projPropManager->has(
		    ProjectionPropertyType::ATTENUATION_PRECORRECTION);

		// Gather measurement
		float measurement = measurementUniformValue;
		if (hasMeasurement)
		{
			measurement = pd_projPropManager->getDataValue<float>(
			    pd_projectionProperties, eventId,
			    ProjectionPropertyType::MEASUREMENT);
		}

		// Gather multiplicative correction factors
		float sensitivity = 1.0f;
		float attenuation = 1.0f;
		if (hasSensitivityFactor)
		{
			sensitivity = pd_projPropManager->getDataValue<float>(
			    pd_projectionProperties, eventId,
			    ProjectionPropertyType::SENSITIVITY);
		}
		if (hasAttenuationFactor)
		{
			attenuation = pd_projPropManager->getDataValue<float>(
			    pd_projectionProperties, eventId,
			    ProjectionPropertyType::ATTENUATION);
		}

		// Gather dynamic frame
		frame_t dynamicFrame = 0;
		if (pd_projPropManager->has(ProjectionPropertyType::DYNAMIC_FRAME))
		{
			dynamicFrame = pd_projPropManager->getDataValue<frame_t>(
			    pd_projectionProperties, eventId,
			    ProjectionPropertyType::DYNAMIC_FRAME);
		}

		// If measurement is zero (or its multiplicative factors are zero),
		//  or if the dynamic frame is negative (disabled event), we can skip
		//  the LOR.
		if (measurement != 0.0f && attenuation != 0.0f && sensitivity != 0.0f &&
		    dynamicFrame >= 0)
		{
			// Gather LOR
			const float* lor = pd_projPropManager->getDataPtr<float>(
			    pd_projectionProperties, eventId, ProjectionPropertyType::LOR);
			static_assert(sizeof(Line3D) == sizeof(float3) * 2);
			float3 p1{lor[0], lor[1], lor[2]};
			float3 p2{lor[3], lor[4], lor[5]};

			// Subtract image offset
			p1.x -= forwImage.params.off_x;
			p1.y -= forwImage.params.off_y;
			p1.z -= forwImage.params.off_z;
			p2.x -= forwImage.params.off_x;
			p2.y -= forwImage.params.off_y;
			p2.z -= forwImage.params.off_z;

			// Gather detector orientation if available
			float3 n1 = {};
			float3 n2 = {};
			if (pd_projPropManager->has(ProjectionPropertyType::DET_ORIENT))
			{
				const float* detOrient = pd_projPropManager->getDataPtr<float>(
				    pd_projectionProperties, eventId,
				    ProjectionPropertyType::DET_ORIENT);
				n1 = {detOrient[0], detOrient[1], detOrient[2]};
				n2 = {detOrient[3], detOrient[4], detOrient[5]};
			}

			// Gather TOF
			float tofValue = 0;
			const bool hasTOF =
			    pd_projPropManager->has(ProjectionPropertyType::TOF);
			if (hasTOF)
			{
				tofValue = pd_projPropManager->getDataValue<float>(
				    pd_projectionProperties, eventId,
				    ProjectionPropertyType::TOF);
			}

			// Forward project
			float update = 0.0f;
			if (hasTOF)
			{
				projectAny<true, true, UseUpdater>(
				    update, forwImage, pd_updater, p1, p2, n1, n2, dynamicFrame,
				    pd_tofHelper, tofValue, projPsfKernelStruct, scannerParams,
				    numRays, eventId, projectorType);
			}
			else
			{
				projectAny<true, false, UseUpdater>(
				    update, forwImage, pd_updater, p1, p2, n1, n2, dynamicFrame,
				    pd_tofHelper, tofValue, projPsfKernelStruct, scannerParams,
				    numRays, eventId, projectorType);
			}

			// Apply correction factors
			update *= globalScaleFactor;
			update *= sensitivity;
			update *= attenuation;
			if (hasRandomsEstimate)
			{
				update += pd_projPropManager->getDataValue<float>(
				    pd_projectionProperties, eventId,
				    ProjectionPropertyType::RANDOMS_ESTIMATE);
			}
			if (hasScatterEstimate)
			{
				update += pd_projPropManager->getDataValue<float>(
				    pd_projectionProperties, eventId,
				    ProjectionPropertyType::SCATTER_ESTIMATE);
			}

			// YN: If we want to compute the log-likelihood, it would have to be
			//  done here (Question: Does the log-likelihood calculation include
			//  the precorrection ?)

			if (hasAttenuationPrecorrectionFactor)
			{
				update *= pd_projPropManager->getDataValue<float>(
				    pd_projectionProperties, eventId,
				    ProjectionPropertyType::ATTENUATION_PRECORRECTION);
			}

			if (fabsf(update) > EPS_FLT)  // to prevent numerical instability
			{
				// Divide measurements
				update = measurement / update;

				// Apply multiplicative correction factors for the backward
				update *= globalScaleFactor;
				update *= sensitivity;
				update *= attenuation;

				// Backproject
				if (hasTOF)
				{
					projectAny<false, true, UseUpdater>(
					    update, emImage, pd_updater, p1, p2, n1, n2,
					    dynamicFrame, pd_tofHelper, tofValue,
					    projPsfKernelStruct, scannerParams, numRays, eventId,
					    projectorType);
				}
				else
				{
					projectAny<false, false, UseUpdater>(
					    update, emImage, pd_updater, p1, p2, n1, n2,
					    dynamicFrame, pd_tofHelper, tofValue,
					    projPsfKernelStruct, scannerParams, numRays, eventId,
					    projectorType);
				}
			}
		}
	}
}

}  // namespace yrt
