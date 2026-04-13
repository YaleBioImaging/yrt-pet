/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/OSEM_CPU.hpp"

#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"
#include "yrt-pet/operators/OperatorVarPsf.hpp"
#include "yrt-pet/operators/ProjectorDD.hpp"
#include "yrt-pet/operators/ProjectorSiddon.hpp"
#include "yrt-pet/recon/Corrector_CPU.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"

#include <utility>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_osem_cpu(pybind11::module& m)
{
	auto c = py::class_<OSEM_CPU, OSEM>(m, "OSEM_CPU");
}
}  // namespace yrt
#endif

namespace yrt
{

OSEM_CPU::OSEM_CPU(const Scanner& pr_scanner)
    : OSEM(pr_scanner),
      mp_tempSensImageBuffer{nullptr},
      mp_mlemImageTmpEMRatio{nullptr}
{
	std::cout << "Creating an instance of OSEM CPU..." << std::endl;

	mp_corrector = std::make_unique<Corrector_CPU>(pr_scanner);
}

void OSEM_CPU::addImagePSF(const std::string& p_imagePsf_fname,
                           ImagePSFMode p_imagePSFMode)
{
	ASSERT_MSG(!p_imagePsf_fname.empty(), "Empty filename for Image-space PSF");
	if (p_imagePSFMode == ImagePSFMode::UNIFORM)
	{
		imagePsf = std::make_unique<OperatorPsf>(p_imagePsf_fname);
		m_imagePSFMode = ImagePSFMode::UNIFORM;
	}
	else
	{
		ASSERT_MSG(imageParams.isValid(),
		           "For spatially variant PSF, image parameters have to be set "
		           "before calling addImagePSF");
		imagePsf =
		    std::make_unique<OperatorVarPsf>(p_imagePsf_fname, imageParams);
		m_imagePSFMode = ImagePSFMode::VARIANT;
	}
	flagImagePSF = true;
}

void OSEM_CPU::setupProjectorForSensImgGen()
{
	ASSERT(projectorParams.numRays > 0);

	// Note: The projector updater for the sensitivity image generation is
	//  always DEFAULT4D

	// Create ProjectorParams object only for sensitivity image, without TOF
	ProjectorParams projParams(scanner);
	projParams.projectorType = projectorParams.projectorType;
	projParams.projPsf_fname = projectorParams.projPsf_fname;
	projParams.numRays = projectorParams.numRays;

	mp_projector = Projector::create(projParams);
}

void OSEM_CPU::allocateForSensImgGen()
{
	auto imageParamsSens = getImageParamsForSensitivityImage();

	auto tempSensImageBuffer = std::make_unique<ImageOwned>(imageParamsSens);
	tempSensImageBuffer->allocate();
	mp_tempSensImageBuffer = std::move(tempSensImageBuffer);

	if (flagImagePSF)
	{
		mp_imageTmpPsf = std::make_unique<ImageOwned>(imageParams);
		reinterpret_cast<ImageOwned*>(mp_imageTmpPsf.get())->allocate();
	}

	initBinLoaderIfNeeded(false);
}

std::unique_ptr<Image> OSEM_CPU::generateSensitivityImageForCurrentSubset()
{
	Image* sensImagePtr = mp_tempSensImageBuffer.get();
	ASSERT(sensImagePtr != nullptr);
	const Projector* projector = mp_projector.get();
	ASSERT(projector != nullptr);
	const BinIterator* binIter = getBinIterator(getCurrentOSEMSubset());
	ASSERT(binIter != nullptr);

	const Corrector_CPU& corrector = getCorrector_CPU();
	const ProjectionData* sensImgGenProjData =
	    corrector.getSensImgGenProjData();
	ASSERT(sensImgGenProjData != nullptr);

	const float globalScale = mp_corrector->getGlobalScalingFactor();

	// Reset sens image buffer
	sensImagePtr->fill(0.0f);

	mp_binLoader->parallelDoOnBins<true>(
	    *sensImgGenProjData, *binIter,
	    [&corrector, &sensImgGenProjData, &projector,
	     &sensImagePtr](const ProjectionPropertyManager& propManager,
	                    PropertyUnit* props, size_t pos, bin_t bin)
	    {
		    const float projValue = corrector.getMultiplicativeCorrectionFactor(
		        *sensImgGenProjData, bin);
		    projector->backProjection(sensImagePtr, propManager, props, pos,
		                              projValue);
	    });

	if (flagImagePSF)
	{
		imagePsf->applyAH(mp_tempSensImageBuffer.get(), mp_imageTmpPsf.get());

		mp_imageTmpPsf->applyThreshold(mp_tempSensImageBuffer.get(), EPS_FLT,
		                               0.0f, 0.0f, 1.0f, 0.0f);
		mp_tempSensImageBuffer.swap(mp_imageTmpPsf);
	}

	// All voxels lower than "hardThreshold" will be put to 0
	std::cout << "Applying threshold..." << std::endl;
	mp_tempSensImageBuffer->applyThreshold(mp_tempSensImageBuffer.get(),
	                                       hardThreshold, 0.0f, 0.0f,
	                                       globalScale, 0.0f);

	// Return generated image, but allocate for the next subset if needed
	// This will dereference mp_tempSensImageBuffer
	auto img = std::move(mp_tempSensImageBuffer);

	// Which requires another allocation for the next subset (if there is one)
	if (getCurrentOSEMSubset() != num_OSEM_subsets - 1)
	{
		auto tempSensImageBuffer =
		    std::make_unique<ImageOwned>(getImageParamsForSensitivityImage());
		tempSensImageBuffer->allocate();
		mp_tempSensImageBuffer = std::move(tempSensImageBuffer);
	}

	return img;
}

void OSEM_CPU::endSensImgGen()
{
	// Clear temporary buffers
	mp_tempSensImageBuffer = nullptr;
}

void OSEM_CPU::setupForDynamicRecon()
{
	OSEM::setupForDynamicRecon();
}

void OSEM_CPU::setupProjectorForRecon()
{
	std::vector<Constraint*> constraints = getConstraintsAsVectorOfPointers();

	if (projectorParams.projectorType == ProjectorType::SIDDON)
	{
		mp_projector = std::make_unique<ProjectorSiddon>(projectorParams);
	}
	else if (projectorParams.projectorType == ProjectorType::DD)
	{
		mp_projector = std::make_unique<ProjectorDD>(projectorParams);
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}
}

void OSEM_CPU::allocateForRecon()
{
	// Allocate for projection-space buffers
	const ProjectionData* dataInput = getDataInput();

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
		outImage->copyFromImage(initialEstimate);
	}
	else
	{
		outImage->fill(INITIAL_VALUE_MLEM);
	}

	// Apply mask function
	auto applyMask = [this](const Image* maskImage) -> void
	{
		outImage->applyThresholdBroadcast(maskImage, 0.0f, 0.0f, 0.0f, 1.0f,
		                                  0.0f);
	};

	// Apply mask image
	std::cout << "Applying threshold..." << std::endl;
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
	mp_mlemImageTmpEMRatio->fill(0.0f);

	mp_corrector->precomputeCorrectionFactors(*dataInput);

	initBinLoaderIfNeeded(true);
}

void OSEM_CPU::loadCurrentSubset(bool /*forRecon*/) {}

void OSEM_CPU::resetEMUpdateImage()
{
	mp_mlemImageTmpEMRatio->fill(0.0);
}

void OSEM_CPU::computeEMUpdateImage()
{
	if (flagImagePSF)
	{
		mp_imageTmpPsf->fill(0.0);
		imagePsf->applyA(outImage.get(), mp_imageTmpPsf.get());

		// We swap here so that the outImage buffer stores the MLEM image with
		//  the PSF applied to it and the imageTmpPsf buffer stores the original
		//  MLEM image.
		outImage.swap(mp_imageTmpPsf);
	}

	const Image* inputImageForForwardProj = outImage.get();
	Image* destImageForBackproj =
	    dynamic_cast<Image*>(getEMUpdateImageBuffer());
	ASSERT(destImageForBackproj != nullptr);

	const Projector* projector = mp_projector.get();
	ASSERT(projector != nullptr);
	const BinIterator* binIter = getBinIterator(getCurrentOSEMSubset());
	ASSERT(binIter != nullptr);
	const ProjectionData* measurements = getDataInput();
	ASSERT(measurements != nullptr);
	const Corrector_CPU& corrector = getCorrector_CPU();

	const float globalScaleFactor = corrector.getGlobalScalingFactor();
	const bool hasSensitivity = corrector.hasSensitivityHistogram();
	const bool hasAttenuation = corrector.hasAttenuation();
	const bool hasScatterEstimates = corrector.hasScatterEstimates();
	const bool hasRandomsEstimates =
	    corrector.hasRandomsEstimates(*measurements);
	const bool hasInVivoAttenuation = corrector.hasInVivoAttenuation();

	corrector.assertMeasurementsMatchCache(measurements);

	mp_binLoader->parallelDoOnBins<false>(
	    *measurements, *binIter,
	    [&projector, &corrector, &measurements, hasRandomsEstimates,
	     hasScatterEstimates, hasInVivoAttenuation, hasAttenuation,
	     hasSensitivity, globalScaleFactor, &destImageForBackproj,
	     &inputImageForForwardProj](
	        const ProjectionPropertyManager& propManager, PropertyUnit* props,
	        size_t pos, bin_t bin)
	    {
		    float update = projector->forwardProjection(
		        inputImageForForwardProj, propManager, props, pos);

		    if (hasSensitivity)
		    {
			    update *= corrector.getPrecomputedSensitivityFactor(bin);
		    }
		    if (hasAttenuation)
		    {
			    update *= corrector.getPrecomputedAttenuationFactor(bin);
		    }
		    update *= globalScaleFactor;

		    if (hasRandomsEstimates)
		    {
			    update += corrector.getPrecomputedRandomsEstimate(bin);
		    }
		    if (hasScatterEstimates)
		    {
			    update += corrector.getPrecomputedScatterEstimate(bin);
		    }

		    if (hasInVivoAttenuation)
		    {
			    update *= corrector.getPrecomputedInVivoAttenuationFactor(bin);
		    }

		    if (std::abs(update) > EPS_FLT)  // to prevent numerical instability
		    {
			    const float measurement = measurements->getProjectionValue(bin);
			    update = measurement / update;

			    if (hasSensitivity)
			    {
				    update *= corrector.getPrecomputedSensitivityFactor(bin);
			    }
			    if (hasAttenuation)
			    {
				    update *= corrector.getPrecomputedAttenuationFactor(bin);
			    }
			    update *= globalScaleFactor;

			    projector->backProjection(destImageForBackproj, propManager,
			                              props, pos, update);
		    }
	    });

	// Backward PSF
	if (flagImagePSF)
	{
		// We swap again here so that outImage gets back to storing the original
		//  MLEM image. This way, we can use the imageTmpPsf buffer to compute
		//  the PSF
		mp_imageTmpPsf.swap(outImage);

		// YN: Is this initialization necessary ?
		mp_imageTmpPsf->fill(0.0);
		imagePsf->applyAH(mp_mlemImageTmpEMRatio.get(), mp_imageTmpPsf.get());

		// We swap these two buffers so that we can use mlemImageTmpEMRatio to
		//  apply the image update
		mp_mlemImageTmpEMRatio.swap(mp_imageTmpPsf);
	}
}

void OSEM_CPU::applyImageUpdate()
{
	// Apply update using the correct sensitivity image
	const ImageBase* sensImage = getSensImageBuffer();

	// Apply the update on the outImage buffer
	outImage->updateEMThresholdDynamic(mp_mlemImageTmpEMRatio.get(), sensImage,
	                                   EPS_FLT);
}


void OSEM_CPU::completeSubset() {}

void OSEM_CPU::completeMLEMIteration() {}

void OSEM_CPU::endRecon()
{
	// Clear temporary buffers
	mp_mlemImageTmpEMRatio = nullptr;
}

ImageBase* OSEM_CPU::getSensImageBuffer()
{
	// In case we are currently generating the sensitivity image
	if (mp_tempSensImageBuffer != nullptr)
	{
		return mp_tempSensImageBuffer.get();
	}
	// In case we are reconstructing
	return getSensitivityImage(usingListModeInput ? 0 : getCurrentOSEMSubset());
}

ImageBase* OSEM_CPU::getMLEMImageBuffer()
{
	return outImage.get();
}

ImageBase* OSEM_CPU::getEMUpdateImageBuffer()
{
	return mp_mlemImageTmpEMRatio.get();
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

void OSEM_CPU::initBinLoaderIfNeeded(bool forRecon)
{
	if (mp_binLoader == nullptr)
	{
		std::vector<Constraint*> constraints =
		    getConstraintsAsVectorOfPointers();
		std::set<ProjectionPropertyType> properties =
		    getNeededProperties(forRecon);

		mp_binLoader = std::make_unique<BinLoader>(constraints, properties);
	}

	if (!mp_binLoader->isAllocated())
	{
		const int numThreads = globals::getNumThreads();
		mp_binLoader->allocate(numThreads);
	}
}

std::set<ProjectionPropertyType>
    OSEM_CPU::getNeededProperties(bool forRecon) const
{
	std::set<ProjectionPropertyType> properties;

	// The correction factors are gathered directly from the corrector
	//  instead of being stored in the projection properties structure

	properties.merge(mp_projector->getProjectionPropertyTypes());

	if (forRecon && getDataInput()->hasDynamicFraming())
	{
		// Dynamic frame is necessary if the input data has dynamic framing
		properties.insert(ProjectionPropertyType::DYNAMIC_FRAME);
	}

	return properties;
}

}  // namespace yrt
