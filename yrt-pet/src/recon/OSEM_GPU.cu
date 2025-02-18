/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/OSEM_GPU.cuh"

#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "operators/OperatorProjectorDD_GPU.cuh"
#include "operators/OperatorProjectorSiddon_GPU.cuh"
#include "operators/OperatorPsfDevice.cuh"
#include "utils/Assert.hpp"

OSEM_GPU::OSEM_GPU(const Scanner& pr_scanner)
    : OSEM(pr_scanner),
      mpd_sensImageBuffer(nullptr),
      mpd_mlemImage(nullptr),
      mpd_mlemImageTmpEMRatio(nullptr),
      mpd_mlemImageTmpPsf(nullptr),
      mpd_tempSensDataInput(nullptr),
      mpd_dat(nullptr),
      mpd_datTmp(nullptr),
      m_current_OSEM_subset(-1)
{
	mp_corrector = std::make_unique<Corrector_GPU>(pr_scanner);

	std::cout << "Creating an instance of OSEM GPU" << std::endl;
}

OSEM_GPU::~OSEM_GPU() = default;

const Corrector& OSEM_GPU::getCorrector() const
{
	return *mp_corrector;
}

Corrector& OSEM_GPU::getCorrector()
{
	return *mp_corrector;
}

const Corrector_GPU& OSEM_GPU::getCorrector_GPU() const
{
	return *mp_corrector;
}

Corrector_GPU& OSEM_GPU::getCorrector_GPU()
{
	return *mp_corrector;
}

void OSEM_GPU::setupOperatorsForSensImgGen()
{
	getBinIterators().clear();
	getBinIterators().reserve(num_OSEM_subsets);

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		// Create and add Bin Iterator
		getBinIterators().push_back(
		    mp_corrector->getSensImgGenProjData()->getBinIter(num_OSEM_subsets,
		                                                      subsetId));
	}
	// Create ProjectorParams object
	OperatorProjectorParams projParams(
	    nullptr /* Will be set later at each subset loading */, scanner, 0.f, 0,
	    flagProjPSF ? projPsf_fname : "", numRays);

	if (projectorType == OperatorProjector::DD)
	{
		mp_projector = std::make_unique<OperatorProjectorDD_GPU>(
		    projParams, getMainStream(), getAuxStream());
	}
	else if (projectorType == OperatorProjector::SIDDON)
	{
		mp_projector = std::make_unique<OperatorProjectorSiddon_GPU>(
		    projParams, getMainStream(), getAuxStream());
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}

	mp_updater = std::make_unique<OSEMUpdater_GPU>(this);
}

void OSEM_GPU::allocateForSensImgGen()
{
	// Allocate for image space
	mpd_sensImageBuffer =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_sensImageBuffer->allocate(true);

	// Allocate for projection space
	auto tempSensDataInput = std::make_unique<ProjectionDataDeviceOwned>(
	    scanner, mp_corrector->getSensImgGenProjData(), num_OSEM_subsets);
	mpd_tempSensDataInput = std::move(tempSensDataInput);

	// Make sure the corrector buffer is properly defined
	mp_corrector->initializeTemporaryDeviceBuffer(mpd_tempSensDataInput.get());
}

std::unique_ptr<Image> OSEM_GPU::getLatestSensitivityImage(bool isLastSubset)
{
	(void)isLastSubset;  // Copy flag is obsolete since the data is not yet on
	// Host-side
	auto img = std::make_unique<ImageOwned>(getImageParams());
	img->allocate();
	mpd_sensImageBuffer->transferToHostMemory(img.get(), true);
	return img;
}

void OSEM_GPU::computeSensitivityImage(ImageBase& destImage)
{
	auto& destImageDevice = dynamic_cast<ImageDevice&>(destImage);
	mp_updater->computeSensitivityImage(destImageDevice);
}

void OSEM_GPU::endSensImgGen()
{
	// Clear temporary buffers
	mpd_sensImageBuffer = nullptr;
	mp_corrector->clearTemporaryDeviceBuffer();
	mpd_tempSensDataInput = nullptr;
}

void OSEM_GPU::setupOperatorsForRecon()
{
	getBinIterators().clear();
	getBinIterators().reserve(num_OSEM_subsets);

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		getBinIterators().push_back(
		    getDataInput()->getBinIter(num_OSEM_subsets, subsetId));
	}

	// Create ProjectorParams object
	OperatorProjectorParams projParams(
	    nullptr /* Will be set later at each subset loading */, scanner,
	    flagProjTOF ? tofWidth_ps : 0.f, flagProjTOF ? tofNumStd : 0,
	    flagProjPSF ? projPsf_fname : "", numRays);

	if (projectorType == OperatorProjector::DD)
	{
		mp_projector = std::make_unique<OperatorProjectorDD_GPU>(
		    projParams, getMainStream(), getAuxStream());
	}
	else if (projectorType == OperatorProjector::SIDDON)
	{
		mp_projector = std::make_unique<OperatorProjectorSiddon_GPU>(
		    projParams, getMainStream(), getAuxStream());
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}

	mp_updater = std::make_unique<OSEMUpdater_GPU>(this);
}

void OSEM_GPU::allocateForRecon()
{
	// Allocate image-space buffers
	mpd_mlemImage =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_mlemImageTmpEMRatio =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_mlemImageTmpPsf =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_sensImageBuffer =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_mlemImage->allocate(false);
	mpd_mlemImageTmpEMRatio->allocate(false);
	mpd_mlemImageTmpPsf->allocate(false);
	mpd_sensImageBuffer->allocate(false);

	// Initialize the MLEM image values to non-zero
	if (initialEstimate != nullptr)
	{
		mpd_mlemImage->copyFromImage(initialEstimate);
	}
	else
	{
		mpd_mlemImage->setValue(INITIAL_VALUE_MLEM);
	}

	// Apply mask image (Use temporary buffer to avoid allocating a new one
	// unnecessarily)
	if (maskImage != nullptr)
	{
		mpd_mlemImageTmpEMRatio->copyFromHostImage(maskImage);
	}
	else if (num_OSEM_subsets == 1 || usingListModeInput)
	{
		// No need to sum all sensitivity images, just use the only one
		mpd_mlemImageTmpEMRatio->copyFromHostImage(getSensitivityImage(0));
	}
	else
	{
		std::cout << "Summing sensitivity images to generate mask image..."
		          << std::endl;
		for (int i = 0; i < num_OSEM_subsets; ++i)
		{
			mpd_sensImageBuffer->copyFromHostImage(getSensitivityImage(i));
			mpd_sensImageBuffer->addFirstImageToSecond(
			    mpd_mlemImageTmpEMRatio.get());
		}
	}
	mpd_mlemImage->applyThreshold(mpd_mlemImageTmpEMRatio.get(), 0.0f, 0.0f,
	                              0.0f, 1.0f, 0.0f);
	mpd_mlemImageTmpEMRatio->setValue(0.0f);

	// Initialize device's sensitivity image with the host's
	if (usingListModeInput)
	{
		mpd_sensImageBuffer->transferToDeviceMemory(getSensitivityImage(0),
		                                            true);
	}

	// Allocate projection-space buffers

	// Use the already-computed BinIterators instead of recomputing them
	std::vector<const BinIterator*> binIteratorPtrList;
	for (const auto& subsetBinIter : getBinIterators())
		binIteratorPtrList.push_back(subsetBinIter.get());

	const ProjectionData* dataInput = getDataInput();
	auto dat = std::make_unique<ProjectionDataDeviceOwned>(
	    scanner, dataInput, binIteratorPtrList, 0.4f);
	auto datTmp = std::make_unique<ProjectionDataDeviceOwned>(dat.get());

	mpd_dat = std::move(dat);
	mpd_datTmp = std::move(datTmp);

	// Make sure the corrector buffer is properly defined
	mp_corrector->initializeTemporaryDeviceBuffer(mpd_dat.get());

	if (mp_corrector->hasAdditiveCorrection())
	{
		mp_corrector->precomputeAdditiveCorrectionFactors(*dataInput);
	}
	if (mp_corrector->hasInVivoAttenuation())
	{
		mp_corrector->precomputeInVivoAttenuationFactors(*dataInput);
	}
}

void OSEM_GPU::endRecon()
{
	ASSERT(outImage != nullptr);

	// Transfer MLEM image Device to host
	mpd_mlemImage->transferToHostMemory(outImage.get(), true);

	// Clear temporary buffers
	mpd_mlemImage = nullptr;
	mpd_mlemImageTmpEMRatio = nullptr;
	mpd_mlemImageTmpPsf = nullptr;
	mpd_sensImageBuffer = nullptr;
	mp_corrector->clearTemporaryDeviceBuffer();
	mpd_dat = nullptr;
	mpd_datTmp = nullptr;
}

ImageBase* OSEM_GPU::getSensImageBuffer()
{
	return mpd_sensImageBuffer.get();
}

const ProjectionDataDeviceOwned*
    OSEM_GPU::getSensitivityDataDeviceBuffer() const
{
	return mpd_tempSensDataInput.get();
}

ProjectionDataDeviceOwned* OSEM_GPU::getSensitivityDataDeviceBuffer()
{
	return mpd_tempSensDataInput.get();
}

ImageBase* OSEM_GPU::getMLEMImageBuffer()
{
	return mpd_mlemImage.get();
}

ImageBase* OSEM_GPU::getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType type)
{
	if (type == TemporaryImageSpaceBufferType::EM_RATIO)
	{
		return mpd_mlemImageTmpEMRatio.get();
	}
	if (type == TemporaryImageSpaceBufferType::PSF)
	{
		return mpd_mlemImageTmpPsf.get();
	}
	throw std::runtime_error("Unknown Temporary image type");
}

const ProjectionData* OSEM_GPU::getMLEMDataBuffer()
{
	return mpd_dat.get();
}

ProjectionData* OSEM_GPU::getMLEMDataTmpBuffer()
{
	return mpd_datTmp.get();
}

OperatorProjectorDevice* OSEM_GPU::getProjector() const
{
	auto* deviceProjector =
	    dynamic_cast<OperatorProjectorDevice*>(mp_projector.get());
	ASSERT(deviceProjector != nullptr);
	return deviceProjector;
}

int OSEM_GPU::getNumBatches(int subsetId, bool forRecon) const
{
	ASSERT(mpd_dat != nullptr);
	ASSERT(mpd_tempSensDataInput != nullptr);

	if (forRecon)
	{
		return mpd_dat->getNumBatches(subsetId);
	}
	return mpd_tempSensDataInput->getNumBatches(subsetId);
}

int OSEM_GPU::getCurrentOSEMSubset() const
{
	return m_current_OSEM_subset;
}

ProjectionDataDeviceOwned* OSEM_GPU::getMLEMDataTmpDeviceBuffer()
{
	return mpd_datTmp.get();
}

const ProjectionDataDeviceOwned* OSEM_GPU::getMLEMDataTmpDeviceBuffer() const
{
	return mpd_datTmp.get();
}

ProjectionDataDeviceOwned* OSEM_GPU::getMLEMDataDeviceBuffer()
{
	return mpd_dat.get();
}

const ProjectionDataDeviceOwned* OSEM_GPU::getMLEMDataDeviceBuffer() const
{
	return mpd_dat.get();
}

void OSEM_GPU::loadBatch(int batchId, bool forRecon)
{
	std::cout << "Loading batch " << batchId + 1 << "/"
	          << getNumBatches(m_current_OSEM_subset, forRecon) << "..."
	          << std::endl;
	if (forRecon)
	{
		mpd_dat->loadEventLORs(m_current_OSEM_subset, batchId,
		                       getAuxStream());
		mpd_dat->allocateForProjValues(getAuxStream());
		mpd_dat->loadProjValuesFromReference(getAuxStream());

		mpd_datTmp->allocateForProjValues(getAuxStream());
	}
	else
	{
		mpd_tempSensDataInput->loadEventLORs(m_current_OSEM_subset, batchId,
		                                     getAuxStream());
		mpd_tempSensDataInput->allocateForProjValues(getAuxStream());
		mpd_tempSensDataInput->loadProjValuesFromReference(getAuxStream());
	}
}

void OSEM_GPU::loadSubset(int subsetId, bool forRecon)
{
	m_current_OSEM_subset = subsetId;

	if (forRecon && !usingListModeInput)
	{
		// Loading the right sensitivity image to the device
		mpd_sensImageBuffer->transferToDeviceMemory(
		    getSensitivityImage(m_current_OSEM_subset), true);
	}
}

void OSEM_GPU::addImagePSF(const std::string& p_imageSpacePsf_fname)
{
	ASSERT_MSG(!p_imageSpacePsf_fname.empty(),
	           "Empty filename for Image-space PSF");
	imagePsf = std::make_unique<OperatorPsfDevice>(p_imageSpacePsf_fname,
	                                                    getMainStream());
	flagImagePSF = true;
}

void OSEM_GPU::completeMLEMIteration() {}

void OSEM_GPU::computeEMUpdateImage(const ImageBase& inputImage,
                                    ImageBase& destImage)
{
	auto& inputImageHost = dynamic_cast<const ImageDevice&>(inputImage);
	auto& destImageHost = dynamic_cast<ImageDevice&>(destImage);
	mp_updater->computeEMUpdateImage(inputImageHost, destImageHost);
}

const cudaStream_t* OSEM_GPU::getAuxStream() const
{
	// TODO: Add parallel loading
	// return &m_auxStream.getStream();
	return &m_mainStream.getStream();
}

const cudaStream_t* OSEM_GPU::getMainStream() const
{
	return &m_mainStream.getStream();
}