/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/OSEM.hpp"

#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionList.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/datastruct/projection/UniformHistogram.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"
#include "yrt-pet/operators/OperatorVarPsf.hpp"
#include "yrt-pet/operators/ProjectorDD.hpp"
#include "yrt-pet/operators/ProjectorSiddon.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_osem(pybind11::module& m)
{
	py::enum_<ImagePSFMode>(m, "ImagePSFMode")
	    .value("UNIFORM", ImagePSFMode::UNIFORM)
	    .value("VARIANT", ImagePSFMode::VARIANT)
	    .export_values();

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
	c.def("setNumRays", &OSEM::setNumRays, "num_rays"_a);
	c.def("addProjPSF", &OSEM::addProjPSF, "proj_psf_fname"_a);
	c.def("addImagePSF",
	      static_cast<void (OSEM::*)(const std::string& p_imagePsf_fname,
	                                 ImagePSFMode p_imagePSFMode)>(
	          &OSEM::addImagePSF),
	      "image_psf_fname"_a, "image_psf_mode"_a = ImagePSFMode::UNIFORM);
	c.def("setSaveIterRanges", &OSEM::setSaveIterRanges, "range_list"_a,
	      "path"_a);
	c.def("setListModeEnabled", &OSEM::setListModeEnabled, "enabled"_a);
	c.def("isListModeEnabled", &OSEM::isListModeEnabled);
	c.def("setProjector",
	      static_cast<void (OSEM::*)(const std::string& projectorName)>(
	          &OSEM::setProjector),
	      "projector_name"_a);
	c.def("setProjector",
	      static_cast<void (OSEM::*)(ProjectorType projectorType)>(
	          &OSEM::setProjector),
	      "projector_name"_a);
	c.def("hasImagePSF", &OSEM::hasImagePSF);
	c.def("setImageParams", &OSEM::setImageParams, "params"_a);
	c.def("getImageParams", &OSEM::getImageParams);
	c.def("getImageParamsForSensitivityImage",
	      &OSEM::getImageParamsForSensitivityImage);
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
	c.def("setProjectorUpdaterType", &OSEM::setProjectorUpdaterType,
	      py::arg("projector_updater_type"));
	c.def("setInitialEstimate", &OSEM::setInitialEstimate,
	      "initial_estimate"_a);
	c.def("setMaskImage", &OSEM::setMaskImage, "mask_image"_a);

	// New projParams getters and setters
	c.def("getProjectorParams", &OSEM::getProjectorParams,
	      py::return_value_policy::reference_internal);
	c.def("getProjectorUpdaterType", &OSEM::getProjectorUpdaterType);

	c.def_readwrite("num_MLEM_iterations", &OSEM::num_MLEM_iterations);
	c.def_readwrite("num_OSEM_subsets", &OSEM::num_OSEM_subsets);
	c.def_readwrite("hardThreshold", &OSEM::hardThreshold);
}
}  // namespace yrt
#endif

namespace yrt
{

OSEM::OSEM(const Scanner& pr_scanner)
    : num_MLEM_iterations(DEFAULT_NUM_ITERATIONS),
      num_OSEM_subsets(1),
      hardThreshold(DEFAULT_HARD_THRESHOLD),
      projectorParams(pr_scanner),
      scanner(pr_scanner),
      flagImagePSF(false),
      imagePsf(nullptr),
      usingListModeInput(false),
      needToMakeCopyOfSensImage(false),
      maskImage(nullptr),
      initialEstimate(nullptr),
      outImage(nullptr),
      m_current_OSEM_subset(-1),
      m_current_MLEM_iteration(-1),
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
	ImageParams sensImageParams;
	m_sensitivityImages.clear();

	for (size_t i = 0; i < sensImages.size(); i++)
	{
		auto sensImage = sensImages[i];

		ASSERT(sensImage != nullptr);
		ASSERT_MSG(sensImage->getParams().isValid(),
		           "Invalid image parameters");

		if (i == 0)
		{
			sensImageParams = sensImage->getParams();
		}
		else
		{
			ASSERT_MSG(sensImage->getParams().isSameAs(sensImageParams),
			           "Image parameters mismatch");
		}
		m_sensitivityImages.push_back(sensImage);
	}
	sensImageParams.nt = imageParams.nt;
	setImageParams(sensImageParams);
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

std::unique_ptr<Image> OSEM::reconstruct(const std::string& out_fname)
{
	ASSERT_MSG(mp_dataInput != nullptr, "Data input unspecified");
	ASSERT_MSG(!m_sensitivityImages.empty(), "Sensitivity image(s) not set");
	ASSERT_MSG(num_OSEM_subsets > 0, "Not enough OSEM subsets");
	ASSERT_MSG(num_MLEM_iterations > 0, "Not enough MLEM iterations");
	ASSERT_MSG(imageParams.isValid(), "Image parameters not valid/set");

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

	initializeForRecon();

	// MLEM iterations
	for (int iter = 0; iter < num_MLEM_iterations; iter++)
	{
		m_current_MLEM_iteration = iter;
		std::cout << "MLEM iteration " << iter + 1 << "/" << num_MLEM_iterations
		          << "..." << std::endl;

		iterate();

		if (saveIterRanges.isIn(iter + 1))
		{
			saveForCurrentIteration();
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
	if (projectorParams.projectorType == ProjectorType::SIDDON)
	{
		std::cout << "Projector type: Siddon" << std::endl;
		std::cout << "Number of Siddon rays: " << projectorParams.numRays
		          << std::endl;
	}
	else if (projectorParams.projectorType == ProjectorType::DD)
	{
		std::cout << "Projector type: Distance-Driven" << std::endl;
	}

	std::cout << "Number of threads used: " << globals::getNumThreads()
	          << std::endl;
	std::cout << "Scanner name: " << scanner.scannerName << std::endl;

	if (flagImagePSF)
	{
		if (dynamic_cast<OperatorVarPsf*>(imagePsf.get()) == nullptr)
		{
			std::cout << "Uses Image-space PSF" << std::endl;
		}
		else
		{
			std::cout << "Uses Image-space variant PSF" << std::endl;
		}
	}
	if (projectorParams.projPsf_fname.empty())
	{
		std::cout << "Uses Projection-space PSF" << std::endl;
	}
	if (projectorParams.hasTOF())
	{
		std::cout << "Uses Time-of-flight" << std::endl;
	}

	std::cout << "Saved iterations list: " << saveIterRanges << std::endl;
	if (!saveIterRanges.empty())
	{
		std::cout << "Saved image files prefix name: " << saveIterPath
		          << std::endl;
	}
}

void OSEM::setSensitivityHistogram(const Histogram* pp_sensitivityData)
{
	getCorrector().setSensitivityHistogram(pp_sensitivityData);
}

const Histogram* OSEM::getSensitivityHistogram() const
{
	return getCorrector().getSensitivityHistogram();
}

void OSEM::setGlobalScalingFactor(float globalScalingFactor)
{
	getCorrector().setGlobalScalingFactor(globalScalingFactor);
}

void OSEM::setInvertSensitivity(bool invert)
{
	getCorrector().setInvertSensitivity(invert);
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
	if (mp_dataInput != nullptr)
	{
		ASSERT_MSG_WARNING(mp_dataInput->hasTOF(),
		                   "Attempting to add TOF to the PET model, but no TOF "
		                   "information is available in the acquisition.");
	}
	projectorParams.addTOF(p_tofWidth_ps, p_tofNumStd);
}

void OSEM::setNumRays(int p_numRays)
{
	projectorParams.numRays = p_numRays;
}

void OSEM::addProjPSF(const std::string& pr_projPsf_fname)
{
	projectorParams.projPsf_fname = pr_projPsf_fname;
}

void OSEM::addImagePSF(const std::string& p_imagePsf_fname)
{
	addImagePSF(p_imagePsf_fname, ImagePSFMode::UNIFORM);
}

void OSEM::setSaveIterRanges(const util::RangeList& p_saveIterList,
                             const std::string& p_saveIterPath)
{
	saveIterRanges = p_saveIterList;
	saveIterPath = p_saveIterPath;
}

void OSEM::setListModeEnabled(bool enabled)
{
	usingListModeInput = enabled;
}

bool OSEM::isListModeEnabled() const
{
	return usingListModeInput;
}

void OSEM::setProjector(const std::string& projectorName)
{
	const ProjectorType projectorType = io::getProjector(projectorName);
	setProjector(projectorType);
}

void OSEM::setProjector(ProjectorType projectorType)
{
	projectorParams.projectorType = projectorType;
}

bool OSEM::hasImagePSF() const
{
	return flagImagePSF;
}

void OSEM::enableNeedToMakeCopyOfSensImage()
{
	needToMakeCopyOfSensImage = true;
}

void OSEM::setImageParams(const ImageParams& params)
{
	imageParams = params;
}

ImageParams OSEM::getImageParams() const
{
	return imageParams;
}

ImageParams OSEM::getImageParamsForSensitivityImage() const
{
	auto imageParamsSens = getImageParams();
	imageParamsSens.nt = 1;
	return imageParamsSens;
}

void OSEM::setRandomsHistogram(const Histogram* pp_randoms)
{
	getCorrector().setRandomsHistogram(pp_randoms);
}

void OSEM::setScatterHistogram(const Histogram* pp_scatter)
{
	getCorrector().setScatterHistogram(pp_scatter);
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

void OSEM::setProjectorUpdaterType(UpdaterType projectorUpdaterType)
{
	projectorParams.updaterType = projectorUpdaterType;
}

void OSEM::setInitialEstimate(const Image* p_initialEstimate)
{
	ASSERT(p_initialEstimate != nullptr);
	if (imageParams.isValid())
	{
		ASSERT_MSG(p_initialEstimate->getParams().isSameAsIgnoreFrames(imageParams),
				   "Image parameters mismatch for initial estimate image");
	}
	else
	{
		imageParams = p_initialEstimate->getParams();
	}
	initialEstimate = p_initialEstimate;
}

void OSEM::setMaskImage(const Image* p_maskImage)
{
	ASSERT(p_maskImage != nullptr);
	const ImageParams& imgParamsForSens = getImageParamsForSensitivityImage();
	ASSERT_MSG(imgParamsForSens.isValid(), "Image parameters not set for OSEM");
	ASSERT_MSG(p_maskImage->getParams().isSameAs(imgParamsForSens),
	           "Image parameters mismatch for mask image");
	maskImage = p_maskImage;
}

const ProjectorParams& OSEM::getProjectorParams() const
{
	return projectorParams;
}

UpdaterType OSEM::getProjectorUpdaterType() const
{
	return projectorParams.updaterType;
}

void OSEM::iterate()
{
	// OSEM subsets
	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		std::cout << "OSEM subset " << subsetId + 1 << "/" << num_OSEM_subsets
		          << "..." << std::endl;

		// Load the appropriate sensitivity image
		loadSubsetInternal(subsetId, true);

		// Reset mutiplicative image
		resetEMUpdateImage();

		// Compute multiplicative image
		computeEMUpdateImage();

		// Update MLEM image using multiplicative image (and sensitivity image)
		applyImageUpdate();

		// Prepare for next subset
		completeSubset();
	}
}

void OSEM::saveForCurrentIteration()
{
	const int numDigitsInFilename = util::numberOfDigits(num_MLEM_iterations);
	const std::string iteration_name =
	    util::padZeros(m_current_MLEM_iteration + 1, numDigitsInFilename);
	const std::string outIteration_fname = util::addBeforeExtension(
	    saveIterPath, std::string("_iteration") + iteration_name);
	getMLEMImageBuffer()->writeToFile(outIteration_fname);
}

void OSEM::setupForDynamicRecon()
{
	const size_t numDynamicFrames = mp_dataInput->getNumDynamicFrames();

	// This ensures that the image parameters used to allocate the
	//  reconstruction image(s) are of the correct size
	imageParams.nt = numDynamicFrames;
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

std::vector<Constraint*> OSEM::getConstraintsAsVectorOfPointers() const
{
	std::vector<Constraint*> constraints;

	for (auto& constraint : m_constraints)
	{
		constraints.emplace_back(constraint.get());
	}

	return constraints;
}

int OSEM::getCurrentOSEMSubset() const
{
	return m_current_OSEM_subset;
}

int OSEM::getCurrentMLEMIteration() const
{
	return m_current_MLEM_iteration;
}

const BinIterator* OSEM::getBinIterator(int subsetId) const
{
	return m_binIterators[subsetId].get();
}

void OSEM::loadSubsetInternal(int p_subsetId, bool p_forRecon)
{
	m_current_OSEM_subset = p_subsetId;

	loadCurrentSubset(p_forRecon);
}

void OSEM::initializeForSensImgGen()
{
	// Setup corrector
	getCorrector().setup();

	// Bin iterators
	initializeBinIterators(getCorrector().getSensImgGenProjData());

	// Constraints
	collectConstraints();

	// Projector
	setupProjectorForSensImgGen();

	// Allocate buffers
	allocateForSensImgGen();
}

void OSEM::initializeForRecon()
{
	// Adjustments for time series
	setupForDynamicRecon();

	initializeOutImageBuffer();
	initializeSensImageBuffer();

	// Setup corrector
	getCorrector().setup();

	// Bin iterators
	initializeBinIterators(getDataInput());

	// Constraints
	collectConstraints();

	// Projector
	setupProjectorForRecon();

	// Allocate buffers
	allocateForRecon();
}

void OSEM::generateSensitivityImagesCore(
    bool saveOnDisk, const std::string& out_fname, bool saveOnMemory,
    std::vector<std::unique_ptr<Image>>& sensImages)
{
	ASSERT_MSG(imageParams.isValid(), "Image parameters not valid/set");
	ASSERT_MSG(num_OSEM_subsets > 0, "Number of subsets must be larger than 0");
	ASSERT_MSG(
	    saveOnDisk || saveOnMemory,
	    "Need to save the sensitivity image either on the disk or the memory");

	// This is done to make sure we only make one sensitivity image if we're on
	// ListMode
	// YN: There must be a cleaner way to do this
	const int originalNumOSEMSubsets = num_OSEM_subsets;
	if (usingListModeInput)
	{
		num_OSEM_subsets = 1;
	}

	initializeForSensImgGen();

	sensImages.clear();

	const int numDigitsInFilename =
	    num_OSEM_subsets > 1 ? util::numberOfDigits(num_OSEM_subsets - 1) : 1;

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		std::cout << "OSEM subset " << subsetId + 1 << "/" << num_OSEM_subsets
		          << "..." << std::endl;

		// Load subset
		loadSubsetInternal(subsetId, false);

		auto generatedImage = generateSensitivityImageForCurrentSubset();

		if (saveOnDisk)
		{
			std::cout << "Saving image to disk..." << std::endl;
			std::string outFileName = out_fname;
			if (num_OSEM_subsets != 1)
			{
				outFileName = util::addBeforeExtension(
				    out_fname,
				    std::string("_subset") +
				        util::padZeros(subsetId, numDigitsInFilename));
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

void OSEM::initializeBinIterators(const ProjectionData* targetProjData)
{
	m_binIterators.clear();
	m_binIterators.reserve(num_OSEM_subsets);
	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		// Create and add Bin Iterator
		m_binIterators.push_back(
		    targetProjData->getBinIter(num_OSEM_subsets, subsetId));
	}
}

void OSEM::initializeOutImageBuffer()
{
	auto newImage = std::make_unique<ImageOwned>(imageParams);
	newImage->allocate();
	outImage = std::move(newImage);
}

void OSEM::initializeSensImageBuffer()
{
	if (usingListModeInput)
	{
		if (needToMakeCopyOfSensImage)
		{
			std::cout << "Arranging sensitivity image scaling for ListMode in "
			             "separate copy..."
			          << std::endl;
			// This is for the specific case of doing a list-mode reconstruction
			// from Python
			auto imageSensParams = m_sensitivityImages[0]->getParams();
			// imageSensParams.num_frames = 1;
			mp_copiedSensitivityImage =
			    std::make_unique<ImageOwned>(imageSensParams);
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
}

void OSEM::collectConstraints()
{
	m_constraints.clear();
	scanner.collectConstraints(m_constraints);
}

}  // namespace yrt
