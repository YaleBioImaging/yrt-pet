/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/OSEM_GPU.cuh"

#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "operators/OperatorProjectorDD_GPU.cuh"
#include "operators/OperatorPsfDevice.cuh"
#include "utils/Assert.hpp"

OSEM_GPU::OSEM_GPU(const Scanner& pr_scanner)
    : OSEM(pr_scanner),
      mpd_sensImageBuffer(nullptr),
      mpd_tempSensDataInput(nullptr),
      mpd_mlemImage(nullptr),
      mpd_mlemImageTmp(nullptr),
      mpd_dat(nullptr),
      mpd_datTmp(nullptr),
      m_current_OSEM_subset(-1)
{
	std::cout << "Creating an instance of OSEM GPU" << std::endl;

	// Since the only available projector in GPU right now is DD_GPU:
	projectorType = OperatorProjector::DD_GPU;
}

OSEM_GPU::~OSEM_GPU() = default;

void OSEM_GPU::setupOperatorsForSensImgGen()
{
	ASSERT_MSG(projectorType == OperatorProjector::ProjectorType::DD_GPU,
	           "No viable projector provided");

	getBinIterators().clear();
	getBinIterators().reserve(num_OSEM_subsets);

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		// Create and add Bin Iterator
		getBinIterators().push_back(
		    getSensDataInput()->getBinIter(num_OSEM_subsets, subsetId));

		// Create ProjectorParams object
	}
	OperatorProjectorParams projParams(
	    nullptr /* Will be set later at each subset loading */, scanner, 0.f, 0,
	    flagProjPSF ? projSpacePsf_fname : "", numRays);

	mp_projector = std::make_unique<OperatorProjectorDD_GPU>(
	    projParams, getMainStream(), getAuxStream());

	if (attenuationImageForBackprojection != nullptr)
	{
		mp_projector->setAttImageForBackprojection(
		    attenuationImageForBackprojection);
	}
}

void OSEM_GPU::allocateForSensImgGen()
{
	// Allocate for image space
	mpd_sensImageBuffer =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_sensImageBuffer->allocate(true);

	// Allocate for projection space
	auto tempSensDataInput = std::make_unique<ProjectionDataDeviceOwned>(
	    scanner, getSensDataInput(), num_OSEM_subsets);
	mpd_tempSensDataInput = std::move(tempSensDataInput);
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

void OSEM_GPU::endSensImgGen()
{
	// Clear temporary buffers
	mpd_sensImageBuffer = nullptr;
	mpd_tempSensDataInput = nullptr;
}

void OSEM_GPU::setupOperatorsForRecon()
{
	ASSERT_MSG(projectorType == OperatorProjector::ProjectorType::DD_GPU,
	           "No viable projector provided");

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
	    flagProjPSF ? projSpacePsf_fname : "", numRays);

	mp_projector = std::make_unique<OperatorProjectorDD_GPU>(
	    projParams, getMainStream(), getAuxStream());
	if (attenuationImageForForwardProjection != nullptr)
	{
		mp_projector->setAttenuationImage(attenuationImageForForwardProjection);
	}
	if (addHis != nullptr)
	{
		mp_projector->setAddHisto(addHis);
	}
}

void OSEM_GPU::allocateForRecon()
{
	// Allocate image-space buffers
	mpd_mlemImage =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_mlemImageTmp =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_sensImageBuffer =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_mlemImage->allocate(false);
	mpd_mlemImageTmp->allocate(false);
	mpd_sensImageBuffer->allocate(false);

	// Initialize the MLEM image values to non zero
	mpd_mlemImage->setValue(INITIAL_VALUE_MLEM);

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

	auto dat = std::make_unique<ProjectionDataDeviceOwned>(
	    scanner, getDataInput(), binIteratorPtrList, 0.4f);
	auto datTmp = std::make_unique<ProjectionDataDeviceOwned>(dat.get());

	mpd_dat = std::move(dat);
	mpd_datTmp = std::move(datTmp);
}

void OSEM_GPU::endRecon()
{
	ASSERT(outImage != nullptr);

	// Transfer MLEM image Device to host
	mpd_mlemImage->transferToHostMemory(outImage.get(), true);

	// Clear temporary buffers
	mpd_mlemImage = nullptr;
	mpd_mlemImageTmp = nullptr;
	mpd_sensImageBuffer = nullptr;
	mpd_dat = nullptr;
	mpd_datTmp = nullptr;
}

ImageBase* OSEM_GPU::getSensImageBuffer()
{
	return mpd_sensImageBuffer.get();
}

ProjectionData* OSEM_GPU::getSensDataInputBuffer()
{
	return mpd_tempSensDataInput.get();
}

ImageBase* OSEM_GPU::getMLEMImageBuffer()
{
	return mpd_mlemImage.get();
}

ImageBase* OSEM_GPU::getMLEMImageTmpBuffer()
{
	return mpd_mlemImageTmp.get();
}

ProjectionData* OSEM_GPU::getMLEMDataBuffer()
{
	return mpd_dat.get();
}

ProjectionData* OSEM_GPU::getMLEMDataTmpBuffer()
{
	return mpd_datTmp.get();
}

int OSEM_GPU::getNumBatches(int subsetId, bool forRecon) const
{
	if (forRecon)
	{
		return mpd_dat->getNumBatches(subsetId);
	}
	return mpd_tempSensDataInput->getNumBatches(subsetId);
}

void OSEM_GPU::loadBatch(int batchId, bool forRecon)
{
	std::cout << "Loading batch " << batchId + 1 << "/"
	          << getNumBatches(m_current_OSEM_subset, forRecon) << "..."
	          << std::endl;
	if (forRecon)
	{
		mpd_dat->loadEventLORs(m_current_OSEM_subset, batchId, imageParams,
		                       getAuxStream());
		mpd_dat->allocateForProjValues(getAuxStream());
		mpd_dat->loadProjValuesFromReference(getAuxStream());

		mpd_datTmp->allocateForProjValues(getAuxStream());
	}
	else
	{
		mpd_tempSensDataInput->loadEventLORs(m_current_OSEM_subset, batchId,
		                                     imageParams, getAuxStream());
		mpd_tempSensDataInput->allocateForProjValues(getAuxStream());
		mpd_tempSensDataInput->loadProjValuesFromReference(getAuxStream());
	}
	std::cout << "Batch " << batchId + 1 << " loaded." << std::endl;
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
	imageSpacePsf = std::make_unique<OperatorPsfDevice>(p_imageSpacePsf_fname,
	                                                    getMainStream());
	flagImagePSF = true;
}

void OSEM_GPU::completeMLEMIteration() {}

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