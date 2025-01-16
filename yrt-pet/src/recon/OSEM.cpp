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
#include "motion/ImageWarperMatrix.hpp"
#include "operators/OperatorProjector.hpp"
#include "operators/OperatorProjectorDD.hpp"
#include "operators/OperatorProjectorSiddon.hpp"
#include "operators/OperatorPsf.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

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
	    py::arg("out_fname") = "", py::arg("saveToMemory") = true);

	c.def("validateSensImagesAmount", &OSEM::validateSensImagesAmount,
	      py::arg("amount"));

	c.def("setSensitivityImage", &OSEM::setSensitivityImage,
	      py::arg("sens_image"), py::arg("subset") = 0);
	c.def("setSensitivityImages",
	      static_cast<void (OSEM::*)(const pybind11::list& pySensImgList)>(
	          &OSEM::setSensitivityImages));

	c.def("reconstruct", &OSEM::reconstruct, py::arg("out_fname") = "");
	c.def("reconstructWithWarperMotion", &OSEM::reconstructWithWarperMotion,
	      py::arg("out_fname") = "");
	c.def("summary", &OSEM::summary);

	c.def("getSensitivityHistogram", &OSEM::getSensitivityHistogram);
	c.def("setSensitivityHistogram", &OSEM::setSensitivityHistogram,
	      py::arg("sens_his"));
	c.def("getDataInput", &OSEM::getDataInput);
	c.def("setDataInput", &OSEM::setDataInput, py::arg("proj_data"));
	c.def("addTOF", &OSEM::addTOF, py::arg("tof_width_ps"),
	      py::arg("tof_num_std"));
	c.def("addProjPSF", &OSEM::addProjPSF, py::arg("proj_psf_fname"));
	c.def("addImagePSF", &OSEM::addImagePSF, py::arg("image_psf_fname"));
	c.def("setSaveIter", &OSEM::setSaveIterRanges, py::arg("range_list"),
	      py::arg("path"));
	c.def("setListModeEnabled", &OSEM::setListModeEnabled, py::arg("enabled"));
	c.def("setProjector", &OSEM::setProjector, py::arg("projector_name"));
	c.def("setImageParams", &OSEM::setImageParams, py::arg("params"));
	c.def("isListModeEnabled", &OSEM::isListModeEnabled);

	c.def_readwrite("num_MLEM_iterations", &OSEM::num_MLEM_iterations);
	c.def_readwrite("num_OSEM_subsets", &OSEM::num_OSEM_subsets);
	c.def_readwrite("hardThreshold", &OSEM::hardThreshold);
	c.def_readwrite("numRays", &OSEM::numRays);
	c.def_readwrite("projectorType", &OSEM::projectorType);
	c.def_readwrite("maskImage", &OSEM::maskImage);
	c.def_readwrite("warper", &OSEM::warper);
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
      warper(nullptr),
      flagImagePSF(false),
      imageSpacePsf(nullptr),
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
		imageSpacePsf->applyAH(getSensImageBuffer(), getSensImageBuffer());
	}

	std::cout << "Applying threshold" << std::endl;
	getSensImageBuffer()->applyThreshold(getSensImageBuffer(), hardThreshold,
	                                     0.0, 0.0, 1.0, 0.0);
	std::cout << "Threshold applied" << std::endl;
}

void OSEM::generateSensitivityImagesCore(
    bool saveOnDisk, const std::string& out_fname, bool saveOnMemory,
    std::vector<std::unique_ptr<Image>>& sensImages)
{
	ASSERT_MSG(imageParams.isValid(), "Image parameters not valid/set");

	Corrector& corrector = getCorrector();

	// In case the user didn't specify a sensitivity histogram, use a uniform one
	bool usedUniformHistogramForSensitivityImageGeneration = false;
	if (corrector.getSensitivityHistogram() == nullptr)
	{
		mp_uniformHistogram = std::make_unique<UniformHistogram>(scanner);
		corrector.setSensitivityHistogram(mp_uniformHistogram.get());
		usedUniformHistogramForSensitivityImageGeneration = true;
	}

	// This is done to make sure we only make one sensitivity image if we're on
	// ListMode
	const int originalNumOSEMSubsets = num_OSEM_subsets;
	if (usingListModeInput)
	{
		num_OSEM_subsets = 1;
	}

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
			std::cout << "Image saved." << std::endl;
		}

		if (saveOnMemory)
		{
			sensImages.push_back(std::move(generatedImage));
		}
	}

	endSensImgGen();

	if (usedUniformHistogramForSensitivityImageGeneration)
	{
		corrector.setSensitivityHistogram(nullptr);
	}

	// Restore original value
	num_OSEM_subsets = originalNumOSEMSubsets;
}

bool OSEM::validateSensImagesAmount(int size) const
{
	if (usingListModeInput)
	{
		return size == 1;
	}
	return size == num_OSEM_subsets;
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
		m_sensitivityImages.resize(expectedSize);
	}

	ASSERT(sensImage != nullptr);
	ASSERT_MSG(sensImage->getParams().isValid(), "Invalid image parameters");

	const ImageParams currentImageParams = getImageParams();
	if (currentImageParams.isValid())
	{
		ASSERT_MSG(sensImage->getParams().isSameAs(currentImageParams),
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

void OSEM::addProjPSF(const std::string& p_projSpacePsf_fname)
{
	projSpacePsf_fname = p_projSpacePsf_fname;
	flagProjPSF = !projSpacePsf_fname.empty();
}

void OSEM::addImagePSF(const std::string& p_imageSpacePsf_fname)
{
	ASSERT_MSG(!p_imageSpacePsf_fname.empty(),
	           "Empty filename for Image-space PSF");
	imageSpacePsf = std::make_unique<OperatorPsf>(p_imageSpacePsf_fname);
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

	if (!validateSensImagesAmount(static_cast<int>(m_sensitivityImages.size())))
	{
		throw std::logic_error(
		    "The number of sensitivity image objects provided does "
		    "not match the number of subsets");
	}

	outImage = std::make_unique<ImageOwned>(imageParams);
	outImage->allocate();

	if (usingListModeInput)
	{
		std::cout << "Arranging sensitivity image scaling for ListMode"
		          << std::endl;
		if (needToMakeCopyOfSensImage)
		{
			// This is for the specific case of doing a list-mode reconstruction
			// from Python
			mp_copiedSensitivityImage =
			    std::make_unique<ImageOwned>(imageParams);
			mp_copiedSensitivityImage->allocate();
			mp_copiedSensitivityImage->copyFromImage(m_sensitivityImages.at(0));
			mp_copiedSensitivityImage->multWithScalar(
			    1.0f / (static_cast<float>(num_OSEM_subsets)));
		}
		else
		{
			m_sensitivityImages[0]->multWithScalar(
			    1.0f / (static_cast<float>(num_OSEM_subsets)));
		}
	}

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

			ImageBase* mlemImage_rp;
			if (flagImagePSF)
			{
				// PSF
				imageSpacePsf->applyA(
				    getMLEMImageBuffer(),
				    getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::PSF));
				mlemImage_rp =
				    getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::PSF);
			}
			else
			{
				mlemImage_rp = getMLEMImageBuffer();
			}

			computeEMUpdateImage(*mlemImage_rp,
			                     *getMLEMImageTmpBuffer(
			                         TemporaryImageSpaceBufferType::EM_RATIO));

			// PSF
			if (flagImagePSF)
			{
				imageSpacePsf->applyAH(
				    getMLEMImageTmpBuffer(
				        TemporaryImageSpaceBufferType::EM_RATIO),
				    getMLEMImageTmpBuffer(
				        TemporaryImageSpaceBufferType::EM_RATIO));
			}

			// UPDATE
			getMLEMImageBuffer()->updateEMThreshold(
			    getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO),
			    getSensImageBuffer(), 0.0);
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

std::unique_ptr<ImageOwned>
    OSEM::reconstructWithWarperMotion(const std::string& out_fname)
{
	ASSERT_MSG(
	    !IO::requiresGPU(projectorType),
	    "Error: The Reconstruction with an image warper only works on CPU");
	ASSERT_MSG(warper != nullptr, "Warper not defined");
	ASSERT_MSG(m_sensitivityImages.size() == 1,
	           "Exactly one sensitivity image is needed for MLEM "
	           "reconstruction with image warper");
	ASSERT_MSG(mp_dataInput != nullptr, "Data input unspecified");

	if (!imageParams.isValid())
	{
		imageParams = m_sensitivityImages.at(0)->getParams();
	}

	outImage = std::make_unique<ImageOwned>(imageParams);
	outImage->allocate();

	allocateForRecon();
	auto mlem_image_update_factor = std::make_unique<ImageOwned>(imageParams);
	mlem_image_update_factor->allocate();
	auto mlem_image_curr_frame = std::make_unique<ImageOwned>(imageParams);
	mlem_image_curr_frame->allocate();

	Image* sens_image = m_sensitivityImages.at(0);
	std::cout << "Computing global Warp-to-ref frame" << std::endl;
	warper->computeGlobalWarpToRefFrame(sens_image, !saveIterRanges.empty());
	std::cout << "Applying threshold" << std::endl;
	sens_image->applyThreshold(sens_image, hardThreshold, 0.0, 0.0, 1.0, 0.0);
	getMLEMImageBuffer()->applyThreshold(sens_image, 0.0, 0.0, 0.0, 0.0, 1.0);
	std::cout << "Threshold applied" << std::endl;

	std::vector<size_t> eventsPartitionOverMotionFrame(
	    warper->getNumberOfFrame() + 1);
	size_t currEventId = 0;
	int frameId = 0;
	float currFrameEndTime = warper->getFrameStartTime(frameId + 1);
	eventsPartitionOverMotionFrame[0] = 0;
	while (frameId < warper->getNumberOfFrame())
	{
		if (currEventId + 1 == getMLEMDataBuffer()->count())
		{
			eventsPartitionOverMotionFrame[frameId + 1] =
			    getMLEMDataBuffer()->count();
			break;
		}

		if (getMLEMDataBuffer()->getTimestamp(currEventId) >= currFrameEndTime)
		{
			eventsPartitionOverMotionFrame[frameId + 1] = currEventId - 1;
			frameId++;
			currFrameEndTime = warper->getFrameStartTime(frameId + 1);
		}
		else
		{
			currEventId++;
		}
	}
	warper->setRefImage(outImage.get());
	OperatorWarpRefImage warpImg(0);
	constexpr float UpdateEMThreshold = 1e-8f;

	const int numFrames = warper->getNumberOfFrame();

	getBinIterators().reserve(numFrames);

	// Subset operators
	for (frameId = 0; frameId < numFrames; frameId++)
	{
		getBinIterators().push_back(std::make_unique<BinIteratorRange>(
		    eventsPartitionOverMotionFrame[frameId],
		    eventsPartitionOverMotionFrame[frameId + 1] - 1));
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
	else
	{
		throw std::logic_error(
		    "Error during reconstruction: Unknown projector type");
	}

	const int num_digits_in_fname = Util::numberOfDigits(num_MLEM_iterations);

	/* MLEM iterations */
	for (int iter = 0; iter < num_MLEM_iterations; iter++)
	{
		std::cout << "\n"
		          << "MLEM iteration " << iter + 1 << "/" << num_MLEM_iterations
		          << "..." << std::endl;
		mlem_image_update_factor->setValue(0.0);
		warper->setRefImage(outImage.get());

		for (frameId = 0; frameId < numFrames; frameId++)
		{
			mp_projector->setBinIter(getBinIterators()[frameId].get());
			getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO)
			    ->setValue(0.0);

			warpImg.setFrameId(frameId);
			warpImg.applyA(warper, mlem_image_curr_frame.get());

			mp_projector->applyA(mlem_image_curr_frame.get(),
			                     getMLEMDataTmpBuffer());
			getMLEMDataTmpBuffer()->divideMeasurements(
			    getMLEMDataBuffer(), getBinIterators()[frameId].get());

			mp_projector->applyAH(
			    getMLEMDataTmpBuffer(),
			    getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO));

			warpImg.applyAH(
			    warper,
			    getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO));

			getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO)
			    ->addFirstImageToSecond(mlem_image_update_factor.get());
		}
		getMLEMImageBuffer()->updateEMThreshold(mlem_image_update_factor.get(),
		                                        sens_image, UpdateEMThreshold);

		if (saveIterRanges.isIn(iter + 1))
		{
			std::string iteration_name =
			    Util::padZeros(iter + 1, num_digits_in_fname);
			std::string out_fname = Util::addBeforeExtension(
			    saveIterPath, std::string("_iteration") + iteration_name);
			getMLEMImageBuffer()->writeToFile(out_fname);
		}
	}

	if (!out_fname.empty())
	{
		std::cout << "Saving image..." << std::endl;
		outImage->writeToFile(out_fname);
	}

	return std::move(outImage);
}

void OSEM::summary() const
{
	if (warper != nullptr)
	{
		std::cout << "Warning: This reconstruction uses deprecated "
		             "Warper-based MLEM. Not "
		             "all features will be enabled."
		          << std::endl;
	}
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
	else if (projectorType == OperatorProjector::DD_GPU)
	{
		std::cout << "Projector type: GPU Distance-Driven" << std::endl;
	}
	std::cout << "Number of threads used: " << Globals::get_num_threads()
	          << std::endl;
	std::cout << "Scanner name: " << scanner.scannerName << std::endl;

	if (flagImagePSF)
	{
		std::cout << "Uses Image-space PSF" << std::endl;
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
