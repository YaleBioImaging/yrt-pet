/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "../ReconstructionConfig.hpp"
#include "datastruct/IO.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ProgressDisplay.hpp"
#include "utils/ReconstructionUtils.hpp"
#include "utils/Utilities.hpp"

#include <cxxopts.hpp>
#include <iostream>

int main(int argc, char** argv)
{
	try
	{
		ArgumentRegistry registry{};

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          "", "0. Core", "s");

		registry.registerArgument(
		    "params",
		    "Image parameters file. Note: If sensitivity image(s) are "
		    "provided, "
		    "the image parameters will be determined from them",
		    false, "", "0. Core", "p");

		registry.registerArgument(
		    "sens_only",
		    "Only generate sensitivity image(s). Do not launch reconstruction",
		    false, false, "0. Core");

		registry.registerArgument("gpu", "Use GPU acceleration", false, false,
		                          "0. Core");

		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, -1, "0. Core");

		registry.registerArgument("out", "Output image filename", false, "",
		                          "0. Core", "o");

		registry.registerArgument(
		    "out_sens",
		    "Sensitivity image output filename (if it needs to be computed). "
		    "Leave blank to not save it",
		    false, "", "0. Core", "");

		// Sensitivity parameters
		registry.registerArgument(
		    "sens",
		    "Sensitivity image files (separated by a comma). Note: When the "
		    "input is a List-mode, one sensitivity image is required. When the "
		    "input is a histogram, one sensitivity image *per subset* is "
		    "required (Ordered by subset id)",
		    false, std::vector<std::string>{}, "1. Sensitivity", "");

		// Load configuration
		ReconstructionConfig config{registry};

		if (!config.loadFromCommandLine(argc, argv))
		{
			// "--help" requested. Quit
			return 0;
		}

		if (!config.validate())
		{
			std::cerr
			    << "Invalid configuration. Please check required parameters."
			    << std::endl;
			return -1;
		}

		if (config.getValue<bool>("sensOnly") &&
		    !config.getValue<bool>("mustMoveSens"))
		{
			ASSERT_MSG(
			    config.getValue<std::vector<std::string>>("sensImg_fnames")
			        .empty(),
			    "Logic error: Sensitivity image generation was requested while "
			    "pre-existing sensitivity images were provided");
		}

		std::cout << "Initializing scanner..." << std::endl;
		auto scanner = std::make_unique<Scanner>(
		    config.getValue<std::string>("scanner_fname"));
		auto projectorType =
		    IO::getProjector(config.getValue<std::string>("projector_name"));
		std::unique_ptr<OSEM> osem =
		    Util::createOSEM(*scanner, config.getValue<bool>("useGPU"));

		osem->num_MLEM_iterations = config.getValue<int>("numIterations");
		osem->num_OSEM_subsets = config.getValue<int>("numSubsets");
		osem->hardThreshold = config.getValue<float>("hardThreshold");
		osem->projectorType = projectorType;
		osem->numRays = config.getValue<int>("numRays");
		Globals::set_num_threads(config.getValue<int>("numThreads"));

		// To make sure the sensitivity image gets generated accordingly
		const bool useListMode =
		    !config.getValue<std::string>("input_format").empty() &&
		    IO::isFormatListMode(config.getValue<std::string>("input_format"));
		osem->setListModeEnabled(useListMode);

		// Total attenuation image
		std::unique_ptr<ImageOwned> attImg = nullptr;
		std::unique_ptr<ProjectionData> acfHisProjData = nullptr;
		if (!config.getValue<std::string>("acf_fname").empty())
		{
			std::cout << "Reading ACF histogram..." << std::endl;
			ASSERT_MSG(!config.getValue<std::string>("acf_format").empty(),
			           "Unspecified format for ACF histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("acf_format")),
			           "ACF has to be in a histogram format");

			acfHisProjData = IO::openProjectionData(
			    config.getValue<std::string>("acf_fname"),
			    config.getValue<std::string>("acf_format"), *scanner,
			    config.getPluginResults());

			const auto* acfHis =
			    dynamic_cast<const Histogram*>(acfHisProjData.get());
			ASSERT(acfHis != nullptr);

			osem->setACFHistogram(acfHis);
		}
		else if (!config.getValue<std::string>("attImg_fname").empty())
		{
			attImg = std::make_unique<ImageOwned>(
			    config.getValue<std::string>("attImg_fname"));
			osem->setAttenuationImage(attImg.get());
		}

		// Hardware attenuation image
		std::unique_ptr<ImageOwned> hardwareAttImg = nullptr;
		std::unique_ptr<ProjectionData> hardwareAcfHisProjData = nullptr;
		if (!config.getValue<std::string>("hardwareAcf_fname").empty())
		{
			std::cout << "Reading hardware ACF histogram..." << std::endl;
			ASSERT_MSG(
			    !config.getValue<std::string>("hardwareAcf_format").empty(),
			    "No format specified for hardware ACF histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("hardwareAcf_format")),
			           "Hardware ACF has to be in a histogram format");

			hardwareAcfHisProjData = IO::openProjectionData(
			    config.getValue<std::string>("hardwareAcf_fname"),
			    config.getValue<std::string>("hardwareAcf_format"), *scanner,
			    config.getPluginResults());

			const auto* hardwareAcfHis =
			    dynamic_cast<const Histogram*>(hardwareAcfHisProjData.get());
			ASSERT(hardwareAcfHis != nullptr);

			osem->setACFHistogram(hardwareAcfHis);
		}
		else if (!config.getValue<std::string>("hardwareAttImg_fname").empty())
		{
			hardwareAttImg = std::make_unique<ImageOwned>(
			    config.getValue<std::string>("hardwareAttImg_fname"));
			osem->setHardwareAttenuationImage(hardwareAttImg.get());
		}

		// Image-space PSF
		if (!config.getValue<std::string>("imagePsf_fname").empty())
		{
			osem->addImagePSF(config.getValue<std::string>("imagePsf_fname"));
		}

		// Projection-space PSF
		if (!config.getValue<std::string>("projPsf_fname").empty())
		{
			osem->addProjPSF(config.getValue<std::string>("projPsf_fname"));
		}

		// Sensitivity image(s)
		std::unique_ptr<ProjectionData> sensitivityProjData = nullptr;
		if (!config.getValue<std::string>("sensitivityData_fname").empty())
		{
			std::cout << "Reading sensitivity histogram..." << std::endl;
			ASSERT_MSG(
			    !config.getValue<std::string>("sensitivityData_format").empty(),
			    "No format specified for sensitivity histogram");
			ASSERT_MSG(!IO::isFormatListMode(config.getValue<std::string>(
			               "sensitivityData_format")),
			           "Sensitivity data has to be in a histogram format");

			sensitivityProjData = IO::openProjectionData(
			    config.getValue<std::string>("sensitivityData_fname"),
			    config.getValue<std::string>("sensitivityData_format"),
			    *scanner, config.getPluginResults());

			const auto* sensitivityHis =
			    dynamic_cast<const Histogram*>(sensitivityProjData.get());
			ASSERT(sensitivityHis != nullptr);

			osem->setSensitivityHistogram(sensitivityHis);
			osem->setInvertSensitivity(
			    config.getValue<bool>("invertSensitivity"));
		}
		osem->setGlobalScalingFactor(
		    config.getValue<float>("globalScalingFactor"));

		std::vector<std::unique_ptr<Image>> sensImages;
		bool sensImageAlreadyMoved = false;
		if (config.getValue<std::vector<std::string>>("sensImg_fnames").empty())
		{
			ASSERT_MSG(!config.getValue<std::string>("imgParams_fname").empty(),
			           "Image parameters file unspecified");
			ImageParams imgParams{
			    config.getValue<std::string>("imgParams_fname")};
			osem->setImageParams(imgParams);

			osem->generateSensitivityImages(
			    sensImages, config.getValue<std::string>("out_sensImg_fname"));
		}
		else if (osem->getExpectedSensImagesAmount() ==
		         static_cast<int>(
		             config.getValue<std::vector<std::string>>("sensImg_fnames")
		                 .size()))
		{
			std::cout << "Reading sensitivity images..." << std::endl;
			for (auto& sensImg_fname :
			     config.getValue<std::vector<std::string>>("sensImg_fnames"))
			{
				sensImages.push_back(
				    std::make_unique<ImageOwned>(sensImg_fname));
			}
			sensImageAlreadyMoved = !config.getValue<bool>("mustMoveSens");
		}
		else
		{
			std::cerr << "The number of sensitivity images given is "
			          << config
			                 .getValue<std::vector<std::string>>(
			                     "sensImg_fnames")
			                 .size()
			          << std::endl;
			std::cerr << "The expected number of sensitivity images is "
			          << osem->getExpectedSensImagesAmount() << std::endl;
			throw std::invalid_argument(
			    "The number of sensitivity images given "
			    "doesn't match the number of "
			    "subsets specified. Note: For ListMode formats, exactly one "
			    "sensitivity image is required.");
		}

		// No need to read data input if in sensOnly mode
		if (config.getValue<bool>("sensOnly") &&
		    config.getValue<std::string>("input_fname").empty())
		{
			std::cout << "Done." << std::endl;
			return 0;
		}

		// Projection Data Input file
		std::cout << "Reading input data..." << std::endl;
		std::unique_ptr<ProjectionData> dataInput;
		ASSERT_MSG(!config.getValue<std::string>("input_format").empty(),
		           "No format specified for Data input");
		dataInput =
		    IO::openProjectionData(config.getValue<std::string>("input_fname"),
		                           config.getValue<std::string>("input_format"),
		                           *scanner, config.getPluginResults());
		osem->setDataInput(dataInput.get());

		std::unique_ptr<ImageOwned> movedSensImage = nullptr;
		if (dataInput->hasMotion() && !sensImageAlreadyMoved)
		{
			ASSERT(sensImages.size() == 1);
			const Image* unmovedSensImage = sensImages[0].get();
			ASSERT(unmovedSensImage != nullptr);

			std::cout << "Moving sensitivity image..." << std::endl;
			movedSensImage = Util::timeAverageMoveSensitivityImage(
			    *dataInput, *unmovedSensImage);

			if (!config.getValue<std::string>("out_sensImg_fname").empty())
			{
				// Overwrite sensitivity image
				std::cout << "Saving sensitivity image..." << std::endl;
				movedSensImage->writeToFile(
				    config.getValue<std::string>("out_sensImg_fname"));
			}

			// Since this part is only for list-mode data, there is only one
			// sensitivity image
			osem->setSensitivityImage(movedSensImage.get());
		}
		else
		{
			std::cout
			    << "No motion in input file. No need to move sensitivity image."
			    << std::endl;
			osem->setSensitivityImages(sensImages);
		}

		if (config.getValue<bool>("sensOnly"))
		{
			std::cout << "Done." << std::endl;
			return 0;
		}

		if (config.getValue<float>("tofWidth_ps") > 0.f)
		{
			osem->addTOF(config.getValue<float>("tofWidth_ps"),
			             config.getValue<int>("tofNumStd"));
		}

		// Additive histograms
		std::unique_ptr<ProjectionData> randomsProjData = nullptr;
		if (!config.getValue<std::string>("randoms_fname").empty())
		{
			std::cout << "Reading randoms histogram..." << std::endl;
			ASSERT_MSG(!config.getValue<std::string>("randoms_format").empty(),
			           "No format specified for randoms histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("randoms_format")),
			           "Randoms must be specified in histogram format");

			randomsProjData = IO::openProjectionData(
			    config.getValue<std::string>("randoms_fname"),
			    config.getValue<std::string>("randoms_format"), *scanner,
			    config.getPluginResults());
			const auto* randomsHis =
			    dynamic_cast<const Histogram*>(randomsProjData.get());
			ASSERT_MSG(randomsHis != nullptr,
			           "The randoms histogram provided does not inherit from "
			           "Histogram.");
			osem->setRandomsHistogram(randomsHis);
		}
		std::unique_ptr<ProjectionData> scatterProjData = nullptr;
		if (!config.getValue<std::string>("scatter_fname").empty())
		{
			std::cout << "Reading scatter histogram..." << std::endl;
			ASSERT_MSG(!config.getValue<std::string>("scatter_format").empty(),
			           "No format specified for scatter histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("scatter_format")),
			           "Scatter must be specified in histogram format");

			scatterProjData = IO::openProjectionData(
			    config.getValue<std::string>("scatter_fname"),
			    config.getValue<std::string>("scatter_format"), *scanner,
			    config.getPluginResults());
			const auto* scatterHis =
			    dynamic_cast<const Histogram*>(scatterProjData.get());
			ASSERT_MSG(scatterHis != nullptr,
			           "The scatter histogram provided does not inherit from "
			           "Histogram.");
			osem->setScatterHistogram(scatterHis);
		}

		std::unique_ptr<ImageOwned> invivoAttImg = nullptr;
		if (!config.getValue<std::string>("invivoAttImg_fname").empty())
		{
			ASSERT_MSG_WARNING(dataInput->hasMotion(),
			                   "An in-vivo attenuation image was provided but "
			                   "the data input has no motion");
			invivoAttImg = std::make_unique<ImageOwned>(
			    config.getValue<std::string>("invivoAttImg_fname"));
			osem->setInVivoAttenuationImage(invivoAttImg.get());
		}
		std::unique_ptr<ProjectionData> inVivoAcfProjData = nullptr;
		if (!config.getValue<std::string>("invivoAcf_fname").empty())
		{
			std::cout << "Reading in-vivo ACF histogram..." << std::endl;
			ASSERT_MSG(
			    !config.getValue<std::string>("invivoAcf_format").empty(),
			    "No format specified for ACF histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("invivoAcf_format")),
			           "In-vivo ACF must be specified in histogram format");

			inVivoAcfProjData = IO::openProjectionData(
			    config.getValue<std::string>("invivoAcf_fname"),
			    config.getValue<std::string>("invivoAcf_format"), *scanner,
			    config.getPluginResults());
			const auto* inVivoAcfHis =
			    dynamic_cast<const Histogram*>(inVivoAcfProjData.get());
			ASSERT_MSG(
			    inVivoAcfHis != nullptr,
			    "The in-vivo ACF histogram provided does not inherit from "
			    "Histogram.");
			osem->setInVivoACFHistogram(inVivoAcfHis);
		}

		// Save steps
		ASSERT_MSG(config.getValue<int>("saveIterStep") >= 0,
		           "save_iter_step must be positive.");
		Util::RangeList ranges;
		if (config.getValue<int>("saveIterStep") > 0)
		{
			if (config.getValue<int>("saveIterStep") == 1)
			{
				ranges.insertSorted(0,
				                    config.getValue<int>("numIterations") - 1);
			}
			else
			{
				for (int it = 0; it < config.getValue<int>("numIterations");
				     it += config.getValue<int>("saveIterStep"))
				{
					ranges.insertSorted(it, it);
				}
			}
		}
		else if (!config.getValue<std::string>("saveIterRanges").empty())
		{
			ranges.readFromString(
			    config.getValue<std::string>("saveIterRanges"));
		}
		if (!ranges.empty())
		{
			osem->setSaveIterRanges(ranges,
			                        config.getValue<std::string>("out_fname"));
		}

		// Initial image estimate
		std::unique_ptr<ImageOwned> initialEstimate = nullptr;
		if (!config.getValue<std::string>("initialEstimate_fname").empty())
		{
			initialEstimate = std::make_unique<ImageOwned>(
			    config.getValue<std::string>("initialEstimate_fname"));
			osem->initialEstimate = initialEstimate.get();
		}

		std::cout << "Launching reconstruction..." << std::endl;
		osem->reconstruct(config.getValue<std::string>("out_fname"));

		std::cout << "Done." << std::endl;
		return 0;
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception& e)
	{
		Util::printExceptionMessage(e);
		return -1;
	}
}
