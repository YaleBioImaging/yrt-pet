/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/OSEM_CPU.hpp"

#include "datastruct/projection/ProjectionList.hpp"
#include "operators/OperatorProjectorDD.hpp"
#include "operators/OperatorProjectorSiddon.hpp"
#include "utils/Assert.hpp"

#include <utility>

OSEM_CPU::OSEM_CPU(const Scanner& pr_scanner)
    : OSEM(pr_scanner),
      mp_tempSensImageBuffer{nullptr},
      mp_mlemImageTmp{nullptr},
      mp_datTmp{nullptr},
      m_current_OSEM_subset{-1}
{
	std::cout << "Creating an instance of OSEM CPU" << std::endl;
}

OSEM_CPU::~OSEM_CPU() = default;


void OSEM_CPU::allocateForSensImgGen()
{
	auto tempSensImageBuffer = std::make_unique<ImageOwned>(getImageParams());
	tempSensImageBuffer->allocate();
	mp_tempSensImageBuffer = std::move(tempSensImageBuffer);
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
		    getSensDataInput()->getBinIter(num_OSEM_subsets, subsetId));
	}

	// Create ProjectorParams object
	OperatorProjectorParams projParams(
	    nullptr /* Will be set later at each subset loading */, scanner, 0.f, 0,
	    flagProjPSF ? projSpacePsf_fname : "", numRays);

	if (projectorType == OperatorProjector::ProjectorType::SIDDON)
	{
		mp_projector = std::make_unique<OperatorProjectorSiddon>(projParams);
	}
	else if (projectorType == OperatorProjector::ProjectorType::DD)
	{
		mp_projector = std::make_unique<OperatorProjectorDD>(projParams);
	}

	if (attenuationImageForBackprojection != nullptr)
	{
		mp_projector->setAttImageForBackprojection(
		    attenuationImageForBackprojection);
	}
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

ProjectionData* OSEM_CPU::getSensDataInputBuffer()
{
	// Since in the CPU version, the projection data is unchanged from the
	// original and stays in the Host.
	return getSensDataInput();
}

ImageBase* OSEM_CPU::getMLEMImageBuffer()
{
	return outImage.get();
}

ImageBase* OSEM_CPU::getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType type)
{
	(void)type;  // IN CPU, use the same buffer for PSF as for EM ratio since we
	             // only have one batch
	return mp_mlemImageTmp.get();
}

ProjectionData* OSEM_CPU::getMLEMDataBuffer()
{
	return getDataInput();
}

ProjectionData* OSEM_CPU::getMLEMDataTmpBuffer()
{
	return mp_datTmp.get();
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

	// Create ProjectorParams object
	OperatorProjectorParams projParams(
	    nullptr /* Will be set later at each subset loading */, scanner,
	    flagProjTOF ? tofWidth_ps : 0.f, flagProjTOF ? tofNumStd : 0,
	    flagProjPSF ? projSpacePsf_fname : "", numRays);

	if (projectorType == OperatorProjector::SIDDON)
	{
		mp_projector = std::make_unique<OperatorProjectorSiddon>(projParams);
	}
	else if (projectorType == OperatorProjector::DD)
	{
		mp_projector = std::make_unique<OperatorProjectorDD>(projParams);
	}

	if (attenuationImageForForwardProjection != nullptr)
	{
		mp_projector->setAttenuationImage(attenuationImageForForwardProjection);
	}
	if (addHis != nullptr)
	{
		mp_projector->setAddHisto(addHis);
	}
}

void OSEM_CPU::allocateForRecon()
{
	// Allocate for projection-space buffers
	mp_datTmp = std::make_unique<ProjectionListOwned>(getDataInput());
	reinterpret_cast<ProjectionListOwned*>(mp_datTmp.get())->allocate();

	// Allocate for image-space buffers
	mp_mlemImageTmp = std::make_unique<ImageOwned>(getImageParams());
	reinterpret_cast<ImageOwned*>(mp_mlemImageTmp.get())->allocate();

	// Initialize output image
	getMLEMImageBuffer()->setValue(INITIAL_VALUE_MLEM);

	// Apply mask image
	std::cout << "Applying threshold" << std::endl;
	auto applyMask = [this](const Image* maskImage) -> void
	{ getMLEMImageBuffer()->applyThreshold(maskImage, 0.0, 0.0, 0.0, 0.0, 1); };
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
			    mp_mlemImageTmp.get());
		}
		std::cout << "Done summing." << std::endl;
		applyMask(mp_mlemImageTmp.get());
	}
	mp_mlemImageTmp->setValue(0.0f);
	std::cout << "Threshold applied" << std::endl;
}

void OSEM_CPU::endRecon()
{
	// Clear temporary buffers
	mp_mlemImageTmp = nullptr;
	mp_datTmp = nullptr;
}

void OSEM_CPU::loadBatch(int batchId, bool forRecon)
{
	// No-op on CPU
	(void)forRecon;
	(void)batchId;
}

void OSEM_CPU::loadSubset(int subsetId, bool forRecon)
{
	(void)forRecon;
	m_current_OSEM_subset = subsetId;
}

void OSEM_CPU::completeMLEMIteration() {}

void OSEM_CPU::prepareEMAccumulation()
{
	if (flagImagePSF)
	{
		// This is because in CPU, we use the same buffer to backproject the EM
		// as to store the PSF'd MLEM image
		mp_mlemImageTmp->setValue(0.0);
	}
}
