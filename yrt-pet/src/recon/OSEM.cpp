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
#include "datastruct/projection/UniformHistogram.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "motion/ImageWarperMatrix.hpp"
#include "operators/OperatorProjector.hpp"
#include "operators/OperatorProjectorDD.hpp"
#include "operators/OperatorProjectorSiddon.hpp"
#include "operators/OperatorPsf.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ReconstructionUtils.hpp"
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

	c.def("getSensDataInput",
	      static_cast<ProjectionData* (OSEM::*)()>(&OSEM::getSensDataInput));
	c.def("setSensDataInput", &OSEM::setSensDataInput);
	c.def("getDataInput",
	      static_cast<ProjectionData* (OSEM::*)()>(&OSEM::getDataInput));
	c.def("setDataInput", &OSEM::setDataInput);
	c.def("addTOF", &OSEM::addTOF);
	c.def("addProjPSF", &OSEM::addProjPSF);
	c.def("addImagePSF", &OSEM::addImagePSF);
	c.def("setSaveSteps", &OSEM::setSaveSteps);
	c.def("setListModeEnabled", &OSEM::setListModeEnabled);
	c.def("setProjector", &OSEM::setProjector);
	c.def("setImageParams", &OSEM::setImageParams);
	c.def("isListModeEnabled", &OSEM::isListModeEnabled);

	c.def_readwrite("num_MLEM_iterations", &OSEM::num_MLEM_iterations);
	c.def_readwrite("num_OSEM_subsets", &OSEM::num_OSEM_subsets);
	c.def_readwrite("hardThreshold", &OSEM::hardThreshold);
	c.def_readwrite("numRays", &OSEM::numRays);
	c.def_readwrite("projectorType", &OSEM::projectorType);
	c.def_readwrite("maskImage", &OSEM::maskImage);
	c.def_readwrite("attenuationImage", &OSEM::attenuationImage);
	c.def_readwrite("attenuationImageForBackprojection",
	                &OSEM::attenuationImageForBackprojection);
	c.def_readwrite("addHis", &OSEM::addHis);
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
      attenuationImage(nullptr),
      attenuationImageForBackprojection(nullptr),
      addHis(nullptr),
      warper(nullptr),
      flagImagePSF(false),
      imageSpacePsf(nullptr),
      flagProjPSF(false),
      flagProjTOF(false),
      tofWidth_ps(0.0f),
      tofNumStd(0),
      saveSteps(0),
      usingListModeInput(false),
      needToMakeCopyOfSensImage(false),
      outImage(nullptr),
      sensDataInput(nullptr),
      dataInput(nullptr),
      copiedSensitivityImage(nullptr)
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

void OSEM::generateSensitivityImageForSubset(int subsetId)
{
	getSensImageBuffer()->setValue(0.0);

	// Backproject everything
	const int numBatches = getNumBatches(subsetId, false);
	for (int batchId = 0; batchId < numBatches; batchId++)
	{
		loadBatch(batchId, false);
		mp_projector->applyAH(getSensDataInputBuffer(), getSensImageBuffer());
	}

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

	// In case the user didn't specify a sensitivity data input
	std::unique_ptr<UniformHistogram> uniformHis = nullptr;
	const bool sensDataInputUnspecified = getSensDataInput() == nullptr;
	if (sensDataInputUnspecified)
	{
		uniformHis = std::make_unique<UniformHistogram>(scanner);
		setSensDataInput(uniformHis.get());
	}

	// This is done to make sure we only make one sensitivity image if we're on
	// ListMode
	const int realNumOSEMSubsets = num_OSEM_subsets;
	if (usingListModeInput)
	{
		num_OSEM_subsets = 1;
	}

	initializeForSensImgGen();

	sensImages.clear();

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		std::cout << "OSEM subset " << subsetId + 1 << "/" << num_OSEM_subsets
		          << "..." << std::endl;

		loadSubsetInternal(subsetId, false);

		generateSensitivityImageForSubset(subsetId);

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
				    std::string("_subset") + std::to_string(subsetId));
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

	if (sensDataInputUnspecified)
	{
		// To prevent a pointer to a deleted object
		setSensDataInput(nullptr);
	}

	// Restore original value
	if (usingListModeInput)
	{
		num_OSEM_subsets = realNumOSEMSubsets;
	}
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
	sensitivityImages.clear();

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
		sensitivityImages.push_back(sensImage);
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
	else if (subset < num_OSEM_subsets)
	{
		std::string errorMessage = "Subset index too high. The expected number "
		                           "of sensitivity images is ";
		errorMessage += std::to_string(num_OSEM_subsets) + ". Subset " +
		                std::to_string(subset) + " does not exist.";
		ASSERT_MSG(false, errorMessage.c_str());
	}
	const size_t expectedSize = usingListModeInput ? 1 : num_OSEM_subsets;
	if (sensitivityImages.size() != expectedSize)
	{
		sensitivityImages.resize(expectedSize);
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

	sensitivityImages[subset] = sensImage;
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

void OSEM::setSensDataInput(ProjectionData* p_sensDataInput)
{
	sensDataInput = p_sensDataInput;
}

void OSEM::setDataInput(ProjectionData* p_dataInput)
{
	dataInput = p_dataInput;
	if (dynamic_cast<const ListMode*>(dataInput))
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

void OSEM::addImagePSF(OperatorPsf* p_imageSpacePsf)
{
	imageSpacePsf = p_imageSpacePsf;
	flagImagePSF = imageSpacePsf != nullptr;
}

void OSEM::setSaveSteps(int p_saveSteps, const std::string& p_saveStepsPath)
{
	if (p_saveSteps > 0)
	{
		saveSteps = p_saveSteps;
		saveStepsPath = p_saveStepsPath;
	}
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

const Image* OSEM::getSensitivityImage(int subsetId) const
{
	if (copiedSensitivityImage != nullptr)
	{
		return copiedSensitivityImage.get();
	}
	return sensitivityImages.at(subsetId);
}

Image* OSEM::getSensitivityImage(int subsetId)
{
	if (copiedSensitivityImage != nullptr)
	{
		return copiedSensitivityImage.get();
	}
	return sensitivityImages.at(subsetId);
}

int OSEM::getNumBatches(int subsetId, bool forRecon) const
{
	(void)subsetId;
	(void)forRecon;
	return 1;
}

std::unique_ptr<ImageOwned> OSEM::reconstruct(const std::string& out_fname)
{
	ASSERT_MSG(dataInput != nullptr, "Data input unspecified");
	ASSERT_MSG(!sensitivityImages.empty(), "Sensitivity image(s) not set");
	ASSERT(imageParams.isValid());

	if (!validateSensImagesAmount(static_cast<int>(sensitivityImages.size())))
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
			copiedSensitivityImage = std::make_unique<ImageOwned>(imageParams);
			copiedSensitivityImage->allocate();
			copiedSensitivityImage->copyFromImage(sensitivityImages.at(0));
			copiedSensitivityImage->multWithScalar(
			    1.0 / (static_cast<double>(num_OSEM_subsets)));
		}
		else
		{
			sensitivityImages[0]->multWithScalar(
			    1.0 / (static_cast<double>(num_OSEM_subsets)));
		}
	}

	initializeForRecon();

	const int numDigitsInFilename =
	    Util::maxNumberOfDigits(num_MLEM_iterations);

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
			getMLEMImageTmpBuffer()->setValue(0.0);

			const int numBatches = getNumBatches(subsetId, true);

			// Data batches in case it doesn't fit in device memory
			for (int batchId = 0; batchId < numBatches; batchId++)
			{
				loadBatch(batchId, true);

				if (numBatches > 1)
				{
					std::cout << "Processing batch " << batchId + 1 << "/"
					          << numBatches << "..." << std::endl;
				}
				getMLEMDataTmpBuffer()->clearProjections(0.0);
				ImageBase* mlem_image_rp;
				if (flagImagePSF)
				{
					// PSF
					imageSpacePsf->applyA(getMLEMImageBuffer(),
					                      getMLEMImageTmpBuffer());
					mlem_image_rp = getMLEMImageTmpBuffer();
				}
				else
				{
					mlem_image_rp = getMLEMImageBuffer();
				}

				// PROJECTION OF IMAGE
				mp_projector->applyA(mlem_image_rp, getMLEMDataTmpBuffer());

				// DATA RATIO
				getMLEMDataTmpBuffer()->divideMeasurements(
				    getMLEMDataBuffer(), getBinIterators()[subsetId].get());

				if (flagImagePSF)
				{
					getMLEMImageTmpBuffer()->setValue(0.0);
				}
				// BACK PROJECTION OF RATIO
				mp_projector->applyAH(getMLEMDataTmpBuffer(),
				                      getMLEMImageTmpBuffer());
			}
			// PSF
			if (flagImagePSF)
			{
				imageSpacePsf->applyAH(getMLEMImageTmpBuffer(),
				                       getMLEMImageTmpBuffer());
			}

			// UPDATE
			getMLEMImageBuffer()->updateEMThreshold(getMLEMImageTmpBuffer(),
			                                        getSensImageBuffer(), 0.0);
		}
		if (saveSteps > 0 && ((iter + 1) % saveSteps) == 0)
		{
			std::string iteration_name =
			    Util::padZeros(iter + 1, numDigitsInFilename);
			std::string outIteration_fname = Util::addBeforeExtension(
			    saveStepsPath, std::string("_iteration") + iteration_name);
			getMLEMImageBuffer()->writeToFile(outIteration_fname);
		}
		completeMLEMIteration();
	}

	endRecon();

	// Deallocate the copied sensitivity image if it was allocated
	copiedSensitivityImage = nullptr;

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
	ASSERT(warper != nullptr);
	ASSERT_MSG(sensitivityImages.size() == 1,
	           "Exactly one sensitivity image is needed for MLEM "
	           "reconstruction with image warper");
	ASSERT_MSG(dataInput != nullptr, "Data input unspecified");

	if (!imageParams.isValid())
	{
		imageParams = sensitivityImages.at(0)->getParams();
	}

	outImage = std::make_unique<ImageOwned>(imageParams);
	outImage->allocate();

	allocateForRecon();
	auto mlem_image_update_factor = std::make_unique<ImageOwned>(imageParams);
	mlem_image_update_factor->allocate();
	auto mlem_image_curr_frame = std::make_unique<ImageOwned>(imageParams);
	mlem_image_curr_frame->allocate();

	Image* sens_image = sensitivityImages.at(0);
	std::cout << "Computing global Warp-to-ref frame" << std::endl;
	warper->computeGlobalWarpToRefFrame(sens_image, saveSteps > 0);
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
	constexpr double UpdateEMThreshold = 1e-8;

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
	if (attenuationImage != nullptr)
	{
		mp_projector->setAttenuationImage(attenuationImage);
	}

	const int num_digits_in_fname =
	    Util::maxNumberOfDigits(num_MLEM_iterations);

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
			getMLEMImageTmpBuffer()->setValue(0.0);

			warpImg.setFrameId(frameId);
			warpImg.applyA(warper, mlem_image_curr_frame.get());

			mp_projector->applyA(mlem_image_curr_frame.get(),
			                     getMLEMDataTmpBuffer());
			getMLEMDataTmpBuffer()->divideMeasurements(
			    getMLEMDataBuffer(), getBinIterators()[frameId].get());

			mp_projector->applyAH(getMLEMDataTmpBuffer(),
			                      getMLEMImageTmpBuffer());

			warpImg.applyAH(warper, getMLEMImageTmpBuffer());

			mlem_image_update_factor->addFirstImageToSecond(
			    getMLEMImageTmpBuffer());
		}
		getMLEMImageBuffer()->updateEMThreshold(mlem_image_update_factor.get(),
		                                        sens_image, UpdateEMThreshold);

		if (saveSteps > 0 && ((iter + 1) % saveSteps) == 0)
		{
			std::string iteration_name =
			    Util::padZeros(iter + 1, num_digits_in_fname);
			std::string out_fname = Util::addBeforeExtension(
			    saveStepsPath, std::string("_iteration") + iteration_name);
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
	for (size_t i = 0; i < sensitivityImages.size(); i++)
	{
		if (sensitivityImages[i] != nullptr)
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

	std::cout << "Save step mode: " << saveSteps << std::endl;
	if (saveSteps)
		std::cout << "Steps image files prefix name: " << saveStepsPath
		          << std::endl;
}
