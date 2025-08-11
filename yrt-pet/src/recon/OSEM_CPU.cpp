/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/OSEM_CPU.hpp"

#include "yrt-pet/datastruct/projection/ProjectionList.hpp"
#include "yrt-pet/operators/OperatorProjectorDD.hpp"
#include "yrt-pet/operators/OperatorProjectorSiddon.hpp"
#include "yrt-pet/recon/Corrector_CPU.hpp"
#include "yrt-pet/utils/Assert.hpp"

#include <utility>

namespace yrt
{
OSEM_CPU::OSEM_CPU(const Scanner& pr_scanner)
    : OSEM(pr_scanner),
      mp_tempSensImageBuffer{nullptr},
      mp_mlemImageTmpEMRatio{nullptr},
      mp_datTmp{nullptr},
      m_current_OSEM_subset{-1}
{
	mp_corrector = std::make_unique<Corrector_CPU>(pr_scanner);

	std::cout << "Creating an instance of OSEM CPU..." << std::endl;
}

OSEM_CPU::~OSEM_CPU() = default;

const Image* OSEM_CPU::getOutputImage() const
{
	return outImage.get();
}

const Corrector& OSEM_CPU::getCorrector() const
{
	return *mp_corrector;
}

Corrector& OSEM_CPU::getCorrector()
{
	return *mp_corrector;
}

const Corrector_CPU& OSEM_CPU::getCorrector_CPU() const
{
	return *mp_corrector;
}

void OSEM_CPU::allocateForSensImgGen()
{
	auto imageParamsSens = getImageParams();
	imageParamsSens.num_frames = 1;
	auto tempSensImageBuffer = std::make_unique<ImageOwned>(imageParamsSens);
	tempSensImageBuffer->allocate();
	mp_tempSensImageBuffer = std::move(tempSensImageBuffer);

	if (flagImagePSF)
	{
		mp_imageTmpPsf = std::make_unique<ImageOwned>(imageParamsSens);
		reinterpret_cast<ImageOwned*>(mp_imageTmpPsf.get())->allocate();
	}
}

void OSEM_CPU::setupOperatorsForSensImgGen()
{
	// TODO: Unify this in OSEM (avoids the copy-paste)
	getBinIterators().clear();
	getBinIterators().reserve(num_OSEM_subsets);

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		// Create and add Bin Iterator
		getBinIterators().push_back(
		    mp_corrector->getSensImgGenProjData()->getBinIter(num_OSEM_subsets,
		                                                      subsetId));
	}

	// Create ProjectorParams object only for sensitivity image, without TOF
	// Todo: projectorUpdaterType for sens image is just DEFAULT3D?
	OperatorProjectorParams projParams(
	    nullptr /* Will be set later at each subset loading */, scanner,
	    OperatorProjectorParams::DEFAULT3D, 0.f, 0,
	    !projectorParams.projPsf_fname.empty() ? projectorParams.projPsf_fname : "",
	    projectorParams.numRays);

	if (projectorType == OperatorProjector::ProjectorType::SIDDON)
	{
		mp_projector = std::make_unique<OperatorProjectorSiddon>(projParams);
	}
	else if (projectorType == OperatorProjector::ProjectorType::DD)
	{
		mp_projector = std::make_unique<OperatorProjectorDD>(projParams);
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}

	setupProjectorUpdater();
	mp_updater = std::make_unique<OSEMUpdater_CPU>(this);
}

std::unique_ptr<Image> OSEM_CPU::getLatestSensitivityImage(bool isLastSubset)
{
	// This will dereference mp_tempSensImageBuffer
	auto img = std::move(mp_tempSensImageBuffer);

	// Which requires another allocation for the next subset
	if (!isLastSubset)
	{
		allocateForSensImgGen();
	}

	return img;
}

void OSEM_CPU::computeSensitivityImage(ImageBase& destImage)
{
	auto& destImageHost = dynamic_cast<Image&>(destImage);
	mp_updater->computeSensitivityImage(destImageHost);
}

void OSEM_CPU::endSensImgGen()
{
	// Clear temporary buffers
	mp_tempSensImageBuffer = nullptr;
}

ImageBase* OSEM_CPU::getSensImageBuffer()
{
	if (mp_tempSensImageBuffer != nullptr)
	{
		return mp_tempSensImageBuffer.get();
	}
	// In case we are not currently generating the sensitivity image
	return getSensitivityImage(usingListModeInput ? 0 : m_current_OSEM_subset);
}

const ProjectionData* OSEM_CPU::getSensitivityBuffer() const
{
	// Since in the CPU version, the projection data is unchanged from the
	// original and stays in the Host.
	return getSensitivityHistogram();
}

ImageBase* OSEM_CPU::getMLEMImageBuffer()
{
	return outImage.get();
}

ImageBase* OSEM_CPU::getImageTmpBuffer(TemporaryImageSpaceBufferType type)
{
	if (type == TemporaryImageSpaceBufferType::EM_RATIO)
	{
		return mp_mlemImageTmpEMRatio.get();
	}
	if (type == TemporaryImageSpaceBufferType::PSF)
	{
		return mp_imageTmpPsf.get();
	}
	throw std::runtime_error("Unknown Temporary image type");
}

const ProjectionData* OSEM_CPU::getMLEMDataBuffer()
{
	return getDataInput();
}

ProjectionData* OSEM_CPU::getMLEMDataTmpBuffer()
{
	return mp_datTmp.get();
}

const OperatorProjector* OSEM_CPU::getProjector() const
{
	const auto* hostProjector =
	    dynamic_cast<const OperatorProjector*>(mp_projector.get());
	ASSERT(hostProjector != nullptr);
	return hostProjector;
}

void OSEM_CPU::setupOperatorsForRecon()
{
	getBinIterators().clear();
	getBinIterators().reserve(num_OSEM_subsets);

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		getBinIterators().push_back(
		    getDataInput()->getBinIter(num_OSEM_subsets, subsetId));
	}

	if (projectorType == OperatorProjector::SIDDON)
	{
		mp_projector = std::make_unique<OperatorProjectorSiddon>(projectorParams);
	}
	else if (projectorType == OperatorProjector::DD)
	{
		mp_projector = std::make_unique<OperatorProjectorDD>(projectorParams);
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}

	setupProjectorUpdater();
	mp_updater = std::make_unique<OSEMUpdater_CPU>(this);
}

void OSEM_CPU::allocateForRecon()
{
	// Allocate for projection-space buffers
	const ProjectionData* dataInput = getDataInput();
	mp_datTmp = std::make_unique<ProjectionListOwned>(dataInput);
	reinterpret_cast<ProjectionListOwned*>(mp_datTmp.get())->allocate();

	// Allocate for image-space buffers
	mp_mlemImageTmpEMRatio = std::make_unique<ImageOwned>(getImageParams());
	reinterpret_cast<ImageOwned*>(mp_mlemImageTmpEMRatio.get())->allocate();
	if (flagImagePSF)
	{
		mp_imageTmpPsf = std::make_unique<ImageOwned>(getImageParams());
		reinterpret_cast<ImageOwned*>(mp_imageTmpPsf.get())->allocate();
	}

	// Initialize output image
	if (initialEstimate != nullptr)
	{
		getMLEMImageBuffer()->copyFromImage(initialEstimate);
	}
	else
	{
		getMLEMImageBuffer()->setValue(INITIAL_VALUE_MLEM);
	}

	// Apply mask image
	std::cout << "Applying threshold..." << std::endl;
	auto applyMask = [this](const Image* maskImage) -> void
	{
		getMLEMImageBuffer()->applyThreshold(maskImage, 0.0f, 0.0f, 0.0f, 1.0f,
		                                     0.0f);
	};
	if (maskImage != nullptr)
	{
		applyMask(maskImage);
	}
	else if (num_OSEM_subsets == 1 || usingListModeInput)
	{
		// No need to sum all sensitivity images, just use the only one
		applyMask(getSensitivityImage(0));
	}
	else
	{
		std::cout << "Summing sensitivity images to generate mask image..."
		          << std::endl;
		for (int i = 0; i < num_OSEM_subsets; ++i)
		{
			getSensitivityImage(i)->addFirstImageToSecond(
			    mp_mlemImageTmpEMRatio.get());
		}
		applyMask(mp_mlemImageTmpEMRatio.get());
	}
	mp_mlemImageTmpEMRatio->setValue(0.0f);

	if (mp_corrector->hasAdditiveCorrection(*dataInput))
	{
		mp_corrector->precomputeAdditiveCorrectionFactors(*dataInput);
	}
	if (mp_corrector->hasInVivoAttenuation())
	{
		mp_corrector->precomputeInVivoAttenuationFactors(*dataInput);
	}
}

void OSEM_CPU::endRecon()
{
	// Clear temporary buffers
	mp_mlemImageTmpEMRatio = nullptr;
	mp_datTmp = nullptr;
}

void OSEM_CPU::loadSubset(int subsetId, bool forRecon)
{
	(void)forRecon;
	m_current_OSEM_subset = subsetId;
}

void OSEM_CPU::computeEMUpdateImage(const ImageBase& inputImage,
                                    ImageBase& destImage)
{
	auto& inputImageHost = dynamic_cast<const Image&>(inputImage);
	auto& destImageHost = dynamic_cast<Image&>(destImage);
	mp_updater->computeEMUpdateImage(inputImageHost, destImageHost);
}

void OSEM_CPU::completeMLEMIteration() {}

void OSEM_CPU::setupProjectorUpdater()
{
	auto projector = reinterpret_cast<OperatorProjector*>(mp_projector.get());
	projector->setupUpdater(projectorParams);
}


}  // namespace yrt
