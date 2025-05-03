/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/OSEM.hpp"

#include "datastruct/IO.hpp"
#include "datastruct/image/Image.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/projection/ListMode.hpp"
#include "datastruct/projection/ProjectionData.hpp"
#include "datastruct/projection/ProjectionList.hpp"
#include "datastruct/projection/UniformHistogram.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "operators/OperatorProjector.hpp"
#include "operators/OperatorProjectorDD.hpp"
#include "operators/OperatorProjectorSiddon.hpp"
#include "operators/OperatorPsf.hpp"
#include "operators/OperatorVarPsf.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

using namespace pybind11::literals;

void py_setup_osem(pybind11::module& m)
{
	auto c = py::class_<OSEM>(m, "OSEM");

	// This returns a python list of the sensitivity images
	c.def(
	    "generateSensitivityImages",
	    [](OSEM& self, const std::string& out_fname,
	       bool saveToMemory) -> py::list
	    {
		    py::list pySensImagesList;
		    if (!saveToMemory)
		    {
			    self.generateSensitivityImages(out_fname);
			    return pySensImagesList;
		    }

		    std::vector<std::unique_ptr<Image>> sensImages;
		    self.generateSensitivityImages(sensImages, out_fname);
		    for (size_t i = 0; i < sensImages.size(); i++)
		    {
			    pySensImagesList.append(std::move(sensImages[i]));
		    }
		    return pySensImagesList;
	    },
	    "out_fname"_a = "", "saveToMemory"_a = true);

	c.def("getExpectedSensImagesAmount", &OSEM::getExpectedSensImagesAmount);

	c.def("setSensitivityImage", &OSEM::setSensitivityImage, "sens_image"_a,
	      "subset"_a = 0);
	c.def("setSensitivityImages",
	      static_cast<void (OSEM::*)(const pybind11::list& pySensImgList)>(
	          &OSEM::setSensitivityImages));

	c.def("reconstruct", &OSEM::reconstruct, "out_fname"_a = "");
	c.def("summary", &OSEM::summary);

	c.def("setSensitivityHistogram", &OSEM::setSensitivityHistogram,
	      "sens_his"_a);
	c.def("getSensitivityHistogram", &OSEM::getSensitivityHistogram);
	c.def("setGlobalScalingFactor", &OSEM::setGlobalScalingFactor,
	      "global_scale"_a);
	c.def("setInvertSensitivity", &OSEM::setInvertSensitivity,
	      "invert"_a = true);
	c.def("getDataInput", &OSEM::getDataInput);
	c.def("setDataInput", &OSEM::setDataInput, "proj_data"_a);
	c.def("addTOF", &OSEM::addTOF, "tof_width_ps"_a, "tof_num_std"_a);
	c.def("addProjPSF", &OSEM::addProjPSF, "proj_psf_fname"_a);
	c.def("addImagePSF", &OSEM::addImagePSF, "image_psf_fname"_a);
	c.def("setSaveIterRanges", &OSEM::setSaveIterRanges, "range_list"_a,
	      "path"_a);
	c.def("setListModeEnabled", &OSEM::setListModeEnabled, "enabled"_a);
	c.def("setProjector", &OSEM::setProjector, "projector_name"_a);
	c.def("setImageParams", &OSEM::setImageParams, "params"_a);
	c.def("getImageParams", &OSEM::getImageParams);
	c.def("isListModeEnabled", &OSEM::isListModeEnabled);
	c.def("setRandomsHistogram", &OSEM::setRandomsHistogram, "randoms_his"_a);
	c.def("setScatterHistogram", &OSEM::setScatterHistogram, "scatter_his"_a);
	c.def("setAttenuationImage", &OSEM::setAttenuationImage, "att_image"_a);
	c.def("setACFHistogram", &OSEM::setACFHistogram, "acf_his"_a);
	c.def("setHardwareAttenuationImage", &OSEM::setHardwareAttenuationImage,
	      "att_hardware"_a);
	c.def("setHardwareACFHistogram", &OSEM::setHardwareACFHistogram,
	      "acf_hardware_his"_a);
	c.def("setInVivoAttenuationImage", &OSEM::setInVivoAttenuationImage,
	      "att_invivo"_a);
	c.def("setInVivoACFHistogram", &OSEM::setInVivoACFHistogram,
	      "acf_invivo_his"_a);

	c.def_readwrite("num_MLEM_iterations", &OSEM::num_MLEM_iterations);
	c.def_readwrite("num_OSEM_subsets", &OSEM::num_OSEM_subsets);
	c.def_readwrite("hardThreshold", &OSEM::hardThreshold);
	c.def_readwrite("numRays", &OSEM::numRays);
	c.def_readwrite("projectorType", &OSEM::projectorType);
	c.def_readwrite("maskImage", &OSEM::maskImage);
	c.def_readwrite("initialEstimate", &OSEM::initialEstimate);
}
#endif

OSEM::OSEM(const Scanner& pr_scanner)
    : num_MLEM_iterations(DEFAULT_NUM_ITERATIONS),
      num_OSEM_subsets(1),
      hardThreshold(DEFAULT_HARD_THRESHOLD),
      numRays(1),
      projectorType(OperatorProjector::SIDDON),
      scanner(pr_scanner),
      maskImage(nullptr),
      initialEstimate(nullptr),
      flagImagePSF(false),
      imagePsf(nullptr),
	  imageVarPsf(nullptr),
      flagProjPSF(false),
      flagProjTOF(false),
      tofWidth_ps(0.0f),
      tofNumStd(0),
      saveIterRanges(),
      usingListModeInput(false),
      needToMakeCopyOfSensImage(false),
      outImage(nullptr),
      mp_dataInput(nullptr),
      mp_copiedSensitivityImage(nullptr)
{
}

void OSEM::generateSensitivityImages(const std::string& out_fname)
{
	std::vector<std::unique_ptr<Image>> dummy;
	generateSensitivityImagesCore(true, out_fname, false, dummy);
}

void OSEM::generateSensitivityImages(
    std::vector<std::unique_ptr<Image>>& sensImages,
    const std::string& out_fname)
{
	if (out_fname.empty())
	{
		generateSensitivityImagesCore(false, "", true, sensImages);
	}
	else
	{
		generateSensitivityImagesCore(true, out_fname, true, sensImages);
	}
}

void OSEM::generateSensitivityImageForLoadedSubset()
{
	getSensImageBuffer()->setValue(0.0);

	computeSensitivityImage(*getSensImageBuffer());

	if (flagImagePSF)
	{
		if (imgpsfmode == UNIFORM)imagePsf->applyAH(getSensImageBuffer(), getSensImageBuffer());
		else imageVarPsf->applyAH(getSensImageBuffer(), getSensImageBuffer());
	}

	std::cout << "Applying threshold..." << std::endl;
	getSensImageBuffer()->applyThreshold(getSensImageBuffer(), hardThreshold,
	                                     0.0, 0.0, 1.0, 0.0);
}

void OSEM::generateSensitivityImagesCore(
    bool saveOnDisk, const std::string& out_fname, bool saveOnMemory,
    std::vector<std::unique_ptr<Image>>& sensImages)
{
	ASSERT_MSG(imageParams.isValid(), "Image parameters not valid/set");
	ASSERT_MSG(num_OSEM_subsets > 0, "Not enough OSEM subsets");

	Corrector& corrector = getCorrector();

	// This is done to make sure we only make one sensitivity image if we're on
	// ListMode
	const int originalNumOSEMSubsets = num_OSEM_subsets;
	if (usingListModeInput)
	{
		num_OSEM_subsets = 1;
	}

	corrector.setup();

	initializeForSensImgGen();

	sensImages.clear();

	const int numDigitsInFilename =
	    num_OSEM_subsets > 1 ? Util::numberOfDigits(num_OSEM_subsets - 1) : 1;

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		std::cout << "OSEM subset " << subsetId + 1 << "/" << num_OSEM_subsets
		          << "..." << std::endl;

		// Load subset
		loadSubsetInternal(subsetId, false);

		// Generate sensitivity image for loaded subset
		generateSensitivityImageForLoadedSubset();

		// Save sensitivity image
		auto generatedImage =
		    getLatestSensitivityImage(subsetId == num_OSEM_subsets - 1);

		if (saveOnDisk)
		{
			std::cout << "Saving image to disk..." << std::endl;
			std::string outFileName = out_fname;
			if (num_OSEM_subsets != 1)
			{
				outFileName = Util::addBeforeExtension(
				    out_fname,
				    std::string("_subset") +
				        Util::padZeros(subsetId, numDigitsInFilename));
			}
			generatedImage->writeToFile(outFileName);
		}

		if (saveOnMemory)
		{
			sensImages.push_back(std::move(generatedImage));
		}
	}

	endSensImgGen();

	// Restore original value
	num_OSEM_subsets = originalNumOSEMSubsets;
}

int OSEM::getExpectedSensImagesAmount() const
{
	if (usingListModeInput)
	{
		return 1;
	}
	return num_OSEM_subsets;
}

void OSEM::setSensitivityImages(const std::vector<Image*>& sensImages)
{
	ImageParams imageParams;
	m_sensitivityImages.clear();

	for (size_t i = 0; i < sensImages.size(); i++)
	{
		auto sensImage = sensImages[i];

		ASSERT(sensImage != nullptr);
		ASSERT_MSG(sensImage->getParams().isValid(),
		           "Invalid image parameters");

		if (i == 0)
		{
			imageParams = sensImage->getParams();
		}
		else
		{
			ASSERT_MSG(sensImage->getParams().isSameAs(imageParams),
			           "Image parameters mismatch");
		}
		m_sensitivityImages.push_back(sensImage);
	}
	setImageParams(imageParams);
}

void OSEM::setSensitivityImages(
    const std::vector<std::unique_ptr<Image>>& sensImages)
{
	std::vector<Image*> sensImages_raw;
	for (size_t i = 0; i < sensImages.size(); i++)
	{
		sensImages_raw.push_back(sensImages[i].get());
	}
	setSensitivityImages(sensImages_raw);
}

#if BUILD_PYBIND11
void OSEM::setSensitivityImages(const pybind11::list& pySensImgList)
{
	std::vector<Image*> sensImages_raw;
	for (size_t i = 0; i < pySensImgList.size(); i++)
	{
		sensImages_raw.push_back(pySensImgList[i].cast<Image*>());
	}
	setSensitivityImages(sensImages_raw);
}
#endif

void OSEM::setSensitivityImage(Image* sensImage, int subset)
{
	if (usingListModeInput)
	{
		ASSERT_MSG(subset == 0, "In List-Mode reconstruction, only one "
		                        "sensitivity image is needed");
	}
	else if (subset >= num_OSEM_subsets)
	{
		std::string errorMessage = "Subset index too high. The expected number "
		                           "of sensitivity images is ";
		errorMessage += std::to_string(num_OSEM_subsets) + ". Subset " +
		                std::to_string(subset) + " does not exist.";
		ASSERT_MSG(false, errorMessage.c_str());
	}
	const size_t expectedSize = usingListModeInput ? 1 : num_OSEM_subsets;
	if (m_sensitivityImages.size() != expectedSize)
	{
		if (subset != 0)
		{
			m_sensitivityImages.resize(expectedSize);
		}
		else
		{
			m_sensitivityImages.resize(1);
		}
	}

	ASSERT(sensImage != nullptr);
	ASSERT_MSG(sensImage->getParams().isValid(), "Invalid image parameters");

	if (imageParams.isValid())
	{
		ASSERT_MSG(sensImage->getParams().isSameAs(imageParams),
		           "Image parameters mismatch");
	}
	else
	{
		setImageParams(sensImage->getParams());
	}

	m_sensitivityImages[subset] = sensImage;
}

void OSEM::loadSubsetInternal(int p_subsetId, bool p_forRecon)
{
	mp_projector->setBinIter(getBinIterators()[p_subsetId].get());
	loadSubset(p_subsetId, p_forRecon);
}

void OSEM::initializeForSensImgGen()
{
	setupOperatorsForSensImgGen();
	allocateForSensImgGen();
}

void OSEM::initializeForRecon()
{
	setupOperatorsForRecon();
	allocateForRecon();
}

void OSEM::setSensitivityHistogram(const Histogram* pp_sensitivityData)
{
	getCorrector().setSensitivityHistogram(pp_sensitivityData);
}

const Histogram* OSEM::getSensitivityHistogram() const
{
	return getCorrector().getSensitivityHistogram();
}

const ProjectionData* OSEM::getDataInput() const
{
	return mp_dataInput;
}

void OSEM::setDataInput(const ProjectionData* pp_dataInput)
{
	mp_dataInput = pp_dataInput;
	if (dynamic_cast<const ListMode*>(mp_dataInput))
	{
		usingListModeInput = true;
	}
	else
	{
		usingListModeInput = false;
	}
}

void OSEM::addTOF(float p_tofWidth_ps, int p_tofNumStd)
{
	tofWidth_ps = p_tofWidth_ps;
	tofNumStd = p_tofNumStd;
	flagProjTOF = true;
}

void OSEM::addProjPSF(const std::string& pr_projPsf_fname)
{
	projPsf_fname = pr_projPsf_fname;
	flagProjPSF = !projPsf_fname.empty();
}

void OSEM::addImagePSF(const std::string& p_imagePsf_fname)
{
	ASSERT_MSG(!p_imagePsf_fname.empty(), "Empty filename for Image-space PSF");
	if (imgpsfmode == UNIFORM)
	{
		imagePsf = std::make_unique<OperatorPsf>(p_imagePsf_fname);
	}
	else 
	{
		imageVarPsf = std::make_unique<OperatorVarPsf>(p_imagePsf_fname);
	}
	flagImagePSF = true;
}

void OSEM::setSaveIterRanges(Util::RangeList p_saveIterList,
                             const std::string& p_saveIterPath)
{
	saveIterRanges = p_saveIterList;
	saveIterPath = p_saveIterPath;
}

void OSEM::setListModeEnabled(bool enabled)
{
	usingListModeInput = enabled;
}

void OSEM::setProjector(const std::string& projectorName)
{
	projectorType = IO::getProjector(projectorName);
}

bool OSEM::isListModeEnabled() const
{
	return usingListModeInput;
}

void OSEM::enableNeedToMakeCopyOfSensImage()
{
	needToMakeCopyOfSensImage = true;
}

ImageParams OSEM::getImageParams() const
{
	return imageParams;
}

void OSEM::setImageParams(const ImageParams& params)
{
	imageParams = params;
}

void OSEM::setRandomsHistogram(const Histogram* pp_randoms)
{
	getCorrector().setRandomsHistogram(pp_randoms);
}

void OSEM::setScatterHistogram(const Histogram* pp_scatter)
{
	getCorrector().setScatterHistogram(pp_scatter);
}

void OSEM::setGlobalScalingFactor(float globalScalingFactor)
{
	getCorrector().setGlobalScalingFactor(globalScalingFactor);
}

void OSEM::setAttenuationImage(const Image* pp_attenuationImage)
{
	getCorrector().setAttenuationImage(pp_attenuationImage);
}

void OSEM::setACFHistogram(const Histogram* pp_acf)
{
	getCorrector().setACFHistogram(pp_acf);
}

void OSEM::setHardwareAttenuationImage(const Image* pp_hardwareAttenuationImage)
{
	getCorrector().setHardwareAttenuationImage(pp_hardwareAttenuationImage);
}

void OSEM::setHardwareACFHistogram(const Histogram* pp_hardwareAcf)
{
	getCorrector().setHardwareACFHistogram(pp_hardwareAcf);
}

void OSEM::setInVivoAttenuationImage(const Image* pp_inVivoAttenuationImage)
{
	getCorrector().setInVivoAttenuationImage(pp_inVivoAttenuationImage);
}

void OSEM::setInVivoACFHistogram(const Histogram* pp_inVivoAcf)
{
	getCorrector().setInVivoACFHistogram(pp_inVivoAcf);
}

void OSEM::setInvertSensitivity(bool invert)
{
	getCorrector().setInvertSensitivity(invert);
}

const Image* OSEM::getSensitivityImage(int subsetId) const
{
	if (mp_copiedSensitivityImage != nullptr)
	{
		return mp_copiedSensitivityImage.get();
	}
	return m_sensitivityImages.at(subsetId);
}

Image* OSEM::getSensitivityImage(int subsetId)
{
	if (mp_copiedSensitivityImage != nullptr)
	{
		return mp_copiedSensitivityImage.get();
	}
	return m_sensitivityImages.at(subsetId);
}

std::unique_ptr<ImageOwned> OSEM::reconstruct(const std::string& out_fname)
{
	ASSERT_MSG(mp_dataInput != nullptr, "Data input unspecified");
	ASSERT_MSG(!m_sensitivityImages.empty(), "Sensitivity image(s) not set");
	ASSERT_MSG(num_OSEM_subsets > 0, "Not enough OSEM subsets");
	ASSERT_MSG(num_MLEM_iterations > 0, "Not enough MLEM iterations");

	if (!imageParams.isValid())
	{
		imageParams = m_sensitivityImages[0]->getParams();
	}

	const int expectedNumberOfSensImages = getExpectedSensImagesAmount();
	if (expectedNumberOfSensImages !=
	    static_cast<int>(m_sensitivityImages.size()))
	{
		throw std::logic_error("The number of sensitivity images provided does "
		                       "not match the number of subsets. Expected " +
		                       std::to_string(expectedNumberOfSensImages) +
		                       " but received " +
		                       std::to_string(m_sensitivityImages.size()));
	}

	outImage = std::make_unique<ImageOwned>(imageParams);
	outImage->allocate();

	if (usingListModeInput)
	{
		if (needToMakeCopyOfSensImage)
		{
			std::cout << "Arranging sensitivity image scaling for ListMode in "
			             "separate copy..."
			          << std::endl;
			// This is for the specific case of doing a list-mode reconstruction
			// from Python
			mp_copiedSensitivityImage =
			    std::make_unique<ImageOwned>(imageParams);
			mp_copiedSensitivityImage->allocate();
			mp_copiedSensitivityImage->copyFromImage(m_sensitivityImages.at(0));
			mp_copiedSensitivityImage->multWithScalar(
			    1.0f / (static_cast<float>(num_OSEM_subsets)));
		}
		else if (num_OSEM_subsets != 1)
		{
			std::cout << "Arranging sensitivity image scaling for ListMode..."
			          << std::endl;
			m_sensitivityImages[0]->multWithScalar(
			    1.0f / (static_cast<float>(num_OSEM_subsets)));
		}
	}

	getCorrector().setup();

	initializeForRecon();

	const int numDigitsInFilename = Util::numberOfDigits(num_MLEM_iterations);

	// MLEM iterations
	for (int iter = 0; iter < num_MLEM_iterations; iter++)
	{
		std::cout << "\n"
		          << "MLEM iteration " << iter + 1 << "/" << num_MLEM_iterations
		          << "..." << std::endl;
		// OSEM subsets
		for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
		{
			std::cout << "OSEM subset " << subsetId + 1 << "/"
			          << num_OSEM_subsets << "..." << std::endl;

			loadSubsetInternal(subsetId, true);

			// SET TMP VARIABLES TO 0
			getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO)
			    ->setValue(0.0);
			getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::PSF)
				->setValue(0.0);
			ImageBase* mlemImage_rp;
			if (flagImagePSF)
			{
				if (imgpsfmode == UNIFORM)
				{
					// PSF
					imagePsf->applyA(
				    	getMLEMImageBuffer(),
				    	getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::PSF));
					mlemImage_rp =
				    	getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::PSF);
					//mlemImage_rp->writeToFile(out_fname);
					//std::cout << "Output PSF result written to " << out_fname << ". Halting program." << std::endl;
    				//std::exit(0);  // Halts the program
				}
				else
				{
					// Variant PSF
					imageVarPsf->applyA(
				    	getMLEMImageBuffer(),
				    	getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::PSF));
					mlemImage_rp =
				    	getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::PSF);
					//mlemImage_rp->writeToFile(out_fname);
					//std::cout << "Output variant PSF result written to " << out_fname << ". Halting program." << std::endl;
    				//std::exit(0);  // Halts the program
				}
			}
			else
			{
				mlemImage_rp = getMLEMImageBuffer();
			}
			//std::cout << "ZTY: debug point 1." << std::endl;
			computeEMUpdateImage(*mlemImage_rp,
			                     *getMLEMImageTmpBuffer(
			                         TemporaryImageSpaceBufferType::EM_RATIO));
			//std::cout << "ZTY: debug point 2." << std::endl;
			// PSF
			//mlemImage_rp =getMLEMImageTmpBuffer(
			//	TemporaryImageSpaceBufferType::EM_RATIO);
			//mlemImage_rp->writeToFile(out_fname);
			//std::cout << "Output after BP result written to " << out_fname << ". Halting program." << std::endl;
			//std::exit(0);  // Halts the program
			if (flagImagePSF)
			{
				if (imgpsfmode == UNIFORM)
				{
					imagePsf->applyAH(getMLEMImageTmpBuffer(
				    	                TemporaryImageSpaceBufferType::EM_RATIO),
				    	              getMLEMImageTmpBuffer(
				        	            TemporaryImageSpaceBufferType::EM_RATIO));
					//mlemImage_rp =getMLEMImageTmpBuffer(
					//	TemporaryImageSpaceBufferType::EM_RATIO);
					//mlemImage_rp->writeToFile(out_fname);
					//std::cout << "Output Transposed PSF result written to " << out_fname << ". Halting program." << std::endl;
    				//std::exit(0);  // Halts the program
				}
				else
				{
					getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::PSF)
					->setValue(0.0);

					imageVarPsf->applyAH(getMLEMImageTmpBuffer(
						TemporaryImageSpaceBufferType::EM_RATIO),
					getMLEMImageTmpBuffer(
						TemporaryImageSpaceBufferType::PSF));
					//mlemImage_rp =getMLEMImageTmpBuffer(
					//	TemporaryImageSpaceBufferType::PSF);
					//mlemImage_rp->writeToFile(out_fname);
					//std::cout << "Output Transposed Var PSF result written to " << out_fname << ". Halting program." << std::endl;
					//std::exit(0);  // Halts the program
				}
			}

			// UPDATE
			if (imgpsfmode == VARIANT)
			{
				getMLEMImageBuffer()->updateEMThreshold(
					getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::PSF),
					getSensImageBuffer(), EPS_FLT);
			}
			else
			{
				getMLEMImageBuffer()->updateEMThreshold(
					getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO),
					getSensImageBuffer(), EPS_FLT);
			}

			
		}
		if (saveIterRanges.isIn(iter + 1))
		{
			std::string iteration_name =
			    Util::padZeros(iter + 1, numDigitsInFilename);
			std::string outIteration_fname = Util::addBeforeExtension(
			    saveIterPath, std::string("_iteration") + iteration_name);
			getMLEMImageBuffer()->writeToFile(outIteration_fname);
		}
		completeMLEMIteration();
	}

	endRecon();

	// Deallocate the copied sensitivity image if it was allocated
	mp_copiedSensitivityImage = nullptr;

	if (!out_fname.empty())
	{
		std::cout << "Saving image..." << std::endl;
		outImage->writeToFile(out_fname);
	}

	return std::move(outImage);
}

void OSEM::summary() const
{
	std::cout << "Number of iterations: " << num_MLEM_iterations << std::endl;
	std::cout << "Number of subsets: " << num_OSEM_subsets << std::endl;
	if (usingListModeInput)
	{
		std::cout << "Uses List-Mode data as input" << std::endl;
	}

	int numberOfSensImagesSet = 0;
	for (size_t i = 0; i < m_sensitivityImages.size(); i++)
	{
		if (m_sensitivityImages[i] != nullptr)
		{
			numberOfSensImagesSet++;
		}
	}
	std::cout << "Number of sensitivity images set: " << numberOfSensImagesSet
	          << std::endl;

	std::cout << "Hard threshold: " << hardThreshold << std::endl;
	if (projectorType == OperatorProjector::SIDDON)
	{
		std::cout << "Projector type: Siddon" << std::endl;
		std::cout << "Number of Siddon rays: " << numRays << std::endl;
	}
	else if (projectorType == OperatorProjector::DD)
	{
		std::cout << "Projector type: Distance-Driven" << std::endl;
	}

	std::cout << "Number of threads used: " << Globals::get_num_threads()
	          << std::endl;
	std::cout << "Scanner name: " << scanner.scannerName << std::endl;

	if (flagImagePSF)
	{
		if (imgpsfmode == UNIFORM)std::cout << "Uses Image-space PSF" << std::endl;
		else std::cout << "Uses Image-space variant PSF" << std::endl;
	}
	if (flagProjPSF)
	{
		std::cout << "Uses Projection-space PSF" << std::endl;
	}
	if (flagProjTOF)
	{
		std::cout << "Uses Time-of-flight with " << std::endl;
	}

	std::cout << "Saved iterations list: " << saveIterRanges << std::endl;
	if (!saveIterRanges.empty())
	{
		std::cout << "Saved image files prefix name: " << saveIterPath
		          << std::endl;
	}
}
