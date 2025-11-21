/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/OSEM_GPU.cuh"

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionDataDevice.cuh"
#include "yrt-pet/operators/OperatorProjectorDD_GPU.cuh"
#include "yrt-pet/operators/OperatorProjectorSiddon_GPU.cuh"
#include "yrt-pet/operators/OperatorPsfDevice.cuh"
#include "yrt-pet/utils/Assert.hpp"

namespace yrt
{
OSEM_GPU::OSEM_GPU(const Scanner& pr_scanner)
    : OSEM(pr_scanner),
      mpd_sensImageBuffer(nullptr),
      mpd_mlemImage(nullptr),
      mpd_mlemImageTmpEMRatio(nullptr),
      mpd_imageTmpPsf(nullptr),
      mpd_tempSensDataInput(nullptr),
      mpd_dat(nullptr),
      mpd_datTmp(nullptr),
      m_current_OSEM_subset(-1)
{
	mp_corrector = std::make_unique<Corrector_GPU>(pr_scanner);

	std::cout << "Creating an instance of OSEM GPU..." << std::endl;
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

std::pair<size_t, size_t>
    OSEM_GPU::calculateMemProj(float shareOfMemoryToUse) const
{
	size_t memAvailable = globals::getDeviceInfo(true);

	// Shrink memory according to the portion we want to use
	memAvailable = static_cast<size_t>(static_cast<float>(memAvailable) *
	                                   shareOfMemoryToUse);

	auto projPropManager = mp_projector->getBinFilter()->getPropertyManager();
	const size_t memoryUsagePerLOR = projPropManager.getElementSize();
	return {memAvailable, memoryUsagePerLOR};
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

	std::vector<Constraint*> constraints;
	if (m_constraints.size() > 0)
	{
		for (auto& constraint : m_constraints)
		{
			constraints.emplace_back(constraint.get());
		}
	}

	OperatorProjectorParams projParams{scanner};
	projParams.projPsf_fname = projectorParams.projPsf_fname;
	projParams.numRays = projectorParams.numRays;

	if (projectorType == OperatorProjector::DD)
	{
		mp_projector = std::make_unique<OperatorProjectorDD_GPU>(
		    projParams, constraints, getMainStream(), getAuxStream());
	}
	else if (projectorType == OperatorProjector::SIDDON)
	{
		mp_projector = std::make_unique<OperatorProjectorSiddon_GPU>(
		    projParams, constraints, getMainStream(), getAuxStream());
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
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getMainStream());
	mpd_sensImageBuffer->allocate(true);

	if (flagImagePSF)
	{
		const auto imagePsfDevice =
		    dynamic_cast<OperatorPsfDevice*>(imagePsf.get());
		ASSERT(imagePsfDevice != nullptr);
		// This is done in order to more accurately compute the available
		//  device memory for the projection-space buffers below
		imagePsfDevice->allocateTemporaryDeviceImageIfNeeded(
		    getImageParams(), {getMainStream(), true});

		mpd_imageTmpPsf = std::make_unique<ImageDeviceOwned>(getImageParams(),
		                                                     getMainStream());
		mpd_imageTmpPsf->allocate(false);
	}

	if (mp_corrector->hasHardwareAttenuationImage())
	{
		mp_corrector->initializeTemporaryDeviceImageIfNeeded(
		    mp_corrector->getHardwareAttenuationImage(),
		    {getMainStream(), true});
	}

	// Allocate for projection space
	auto [memAvailable, memoryUsagePerLOR] =
	    calculateMemProj(DefaultMemoryShare);
	int numFloatValues =
	    static_cast<int>(mp_corrector->isTemporaryDeviceBufferNeededForSens());
	mpd_tempSensDataInput = std::make_unique<ProjectionDataDeviceOwned>(
	    scanner, mp_corrector->getSensImgGenProjData(),
	    memoryUsagePerLOR + numFloatValues * sizeof(float), memAvailable,
	    num_OSEM_subsets);

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
	std::vector<Constraint*> constraints;
	if (m_constraints.size() > 0)
	{
		for (auto& constraint : m_constraints)
		{
			constraints.emplace_back(constraint.get());
		}
	}

	if (projectorType == OperatorProjector::DD)
	{
		mp_projector = std::make_unique<OperatorProjectorDD_GPU>(
		    projectorParams, constraints, getMainStream(), getAuxStream());
	}
	else if (projectorType == OperatorProjector::SIDDON)
	{
		mp_projector = std::make_unique<OperatorProjectorSiddon_GPU>(
		    projectorParams, constraints, getMainStream(), getAuxStream());
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}

	setupProjectorUpdater();
	mp_updater = std::make_unique<OSEMUpdater_GPU>(this);
}

void OSEM_GPU::allocateForRecon()
{
	// Allocate image-space buffers
	mpd_mlemImage =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getMainStream());
	mpd_mlemImageTmpEMRatio =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getMainStream());
	mpd_sensImageBuffer =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getMainStream());

	mpd_mlemImage->allocate(false);
	mpd_mlemImageTmpEMRatio->allocate(false);
	mpd_sensImageBuffer->allocate(false);

	if (flagImagePSF)
	{
		mpd_imageTmpPsf = std::make_unique<ImageDeviceOwned>(getImageParams(),
		                                                     getMainStream());
		mpd_imageTmpPsf->allocate(false);
	}

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
		mpd_mlemImageTmpEMRatio->copyFromHostImage(maskImage, true);
	}
	else if (num_OSEM_subsets == 1 || usingListModeInput)
	{
		// No need to sum all sensitivity images, just use the only one
		mpd_mlemImageTmpEMRatio->copyFromHostImage(getSensitivityImage(0),
		                                           true);
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
			    mpd_mlemImageTmpEMRatio.get(), false);
		}
	}
	mpd_mlemImage->applyThresholdDevice(mpd_mlemImageTmpEMRatio.get(), 0.0f,
	                                    0.0f, 0.0f, 1.0f, 0.0f, false);
	mpd_mlemImageTmpEMRatio->setValueDevice(0.0f, false);

	// Initialize device's sensitivity image with the host's
	if (usingListModeInput)
	{
		mpd_sensImageBuffer->transferToDeviceMemory(getSensitivityImage(0),
		                                            true);
	}

	// Use the already-computed BinIterators instead of recomputing them
	std::vector<const BinIterator*> binIteratorPtrList;
	for (const auto& subsetBinIter : getBinIterators())
		binIteratorPtrList.push_back(subsetBinIter.get());

	// Allocate projection-space buffers
	auto [memAvailable, memoryUsagePerLOR] =
	    calculateMemProj(DefaultMemoryShare);
	const ProjectionData* dataInput = getDataInput();
	// Add one float for mpd_datTmp (float for mpd_dat is accounted for in
	// createBatchSetups)
	int numFloatValues =
	    1 +
	    static_cast<int>(
	        mp_corrector->isTemporaryDeviceBufferNeededForRecon(*dataInput));
	auto dat = std::make_unique<ProjectionDataDeviceOwned>(
	    scanner, dataInput, binIteratorPtrList,
	    memoryUsagePerLOR + numFloatValues * sizeof(float), memAvailable);
	auto datTmp = std::make_unique<ProjectionDataDeviceOwned>(dat.get());

	mpd_dat = std::move(dat);
	mpd_datTmp = std::move(datTmp);

	// Make sure the corrector buffer is properly defined
	mp_corrector->initializeTemporaryDeviceBuffer(mpd_dat.get());

	if (mp_corrector->hasAdditiveCorrection(*dataInput))
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
	mpd_imageTmpPsf = nullptr;
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

ImageBase* OSEM_GPU::getImageTmpBuffer(TemporaryImageSpaceBufferType type)
{
	if (type == TemporaryImageSpaceBufferType::EM_RATIO)
	{
		return mpd_mlemImageTmpEMRatio.get();
	}
	if (type == TemporaryImageSpaceBufferType::PSF)
	{
		ASSERT(flagImagePSF);
		return mpd_imageTmpPsf.get();
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

void OSEM_GPU::addImagePSF(const std::string& p_imagePsf_fname,
                           ImagePSFMode p_imagePSFMode)
{
	ASSERT_MSG(!p_imagePsf_fname.empty(), "Empty filename for Image-space PSF");
	if (p_imagePSFMode == UNIFORM)
	{
		imagePsf = std::make_unique<OperatorPsfDevice>(p_imagePsf_fname,
		                                               getMainStream());
	}
	else
	{
		ASSERT_MSG(false, "Spatially variant PSF not implemented in GPU yet");
	}

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
	return &m_auxStream.getStream();
}

const cudaStream_t* OSEM_GPU::getMainStream() const
{
	return &m_mainStream.getStream();
}

void OSEM_GPU::setupProjectorUpdater()
{
	auto projector = reinterpret_cast<OperatorProjectorDevice*>(mp_projector.get());
	//projector->setupUpdater(projectorParams);
}

Array2DBase<float>* OSEM_GPU::getHBasisTmpBuffer()
{
	return nullptr;
}
void OSEM_GPU::allocateHBasisTmpBuffer()
{
}

void OSEM_GPU::initializeHBasisTmpBuffer() {}

void OSEM_GPU::generateWUpdateSensScaling(float* c) {}

void OSEM_GPU::generateHUpdateSensScaling(float* c) {}

}  // namespace yrt
