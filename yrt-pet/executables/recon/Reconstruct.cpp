/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../ArgumentReader.hpp"
#include "../PluginOptionsHelper.hpp"
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
		IO::ArgumentRegistry registry{};

		std::string coreGroup = "0. Core";
		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          "", coreGroup, "s");

		registry.registerArgument(
		    "params",
		    "Image parameters file. Note: If sensitivity image(s) are "
		    "provided, "
		    "the image parameters will be determined from them",
		    false, "", coreGroup, "p");

		registry.registerArgument(
		    "sens_only",
		    "Only generate sensitivity image(s). Do not launch reconstruction",
		    false, false, coreGroup);

#if BUILD_CUDA
		registry.registerArgument("gpu", "Use GPU acceleration", false, false,
		                          coreGroup);
#endif
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, -1, coreGroup);

		registry.registerArgument("out", "Output image filename", false, "",
		                          coreGroup, "o");

		registry.registerArgument(
		    "out_sens",
		    "Sensitivity image output filename (if it needs to be computed). "
		    "Leave blank to not save it",
		    false, "", coreGroup, "");

		// Sensitivity parameters
		std::string sensitivityGroup = "1. Sensitivity";
		registry.registerArgument(
		    "sens",
		    "Sensitivity image files (separated by a comma). Note: When the "
		    "input is a List-mode, one sensitivity image is required. When the "
		    "input is a histogram, one sensitivity image *per subset* is "
		    "required (Ordered by subset id)",
		    false, std::vector<std::string>{}, sensitivityGroup, "");
		registry.registerArgument("sensitivity", "Sensitivity histogram file",
		                          false, "", sensitivityGroup);
		registry.registerArgument(
		    "sensitivity_format",
		    "Sensitivity histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, "", sensitivityGroup);
		registry.registerArgument("invert_sensitivity",
		                          "Invert the sensitivity histogram values "
		                          "(sensitivity -> 1/sensitivity)",
		                          false, false, sensitivityGroup);
		registry.registerArgument(
		    "global_scale", "Global scaling factor to apply on the sensitivity",
		    false, 1.0f, sensitivityGroup);
		registry.registerArgument(
		    "move_sens", "Move the provided sensitivity image based on motion",
		    false, false, sensitivityGroup);

		// Input data parameters
		std::string inputGroup = "2. Input";
		registry.registerArgument("input", "Input file", false, "", inputGroup,
		                          "i");
		registry.registerArgument("format",
		                          "Input file format. Possible values: " +
		                              IO::possibleFormats(),
		                          false, "", inputGroup, "f");

		// Reconstruction parameters
		std::string reconstructionGroup = "3. Reconstruction";
		registry.registerArgument("num_iterations", "Number of MLEM Iterations",
		                          false, 10, reconstructionGroup);
		registry.registerArgument("num_subsets",
		                          "Number of OSEM subsets (Default: 1)", false,
		                          1, reconstructionGroup);
		registry.registerArgument("initial_estimate",
		                          "Initial image estimate for the MLEM", false,
		                          "", reconstructionGroup);
		registry.registerArgument("randoms",
		                          "Randoms estimate histogram filename", false,
		                          "", reconstructionGroup);
		registry.registerArgument(
		    "randoms_format",
		    "Randoms estimate histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, "", reconstructionGroup);
		registry.registerArgument("scatter",
		                          "Scatter estimate histogram filename", false,
		                          "", reconstructionGroup);
		registry.registerArgument(
		    "scatter_format",
		    "Scatter estimate histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, "", reconstructionGroup);
		registry.registerArgument("psf", "Image-space PSF kernel file", false,
		                          "", reconstructionGroup);
		registry.registerArgument("hard_threshold", "Hard Threshold", false,
		                          1.0f, reconstructionGroup);
		registry.registerArgument(
		    "save_iter_step",
		    "Increment into which to save MLEM iteration images", false, 0,
		    reconstructionGroup);
		registry.registerArgument(
		    "save_iter_ranges",
		    "List of iteration ranges to save MLEM iteration images", false, "",
		    reconstructionGroup);

		std::string attenuationGroup = "3.1 Attenuation correction";
		registry.registerArgument("att", "Total attenuation image filename",
		                          false, "", attenuationGroup);
		registry.registerArgument(
		    "acf", "Total attenuation correction factors histogram filename",
		    false, "", attenuationGroup);
		registry.registerArgument(
		    "acf_format",
		    "Total attenuation correction factors histogram format. Possible "
		    "values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, "", attenuationGroup);
		registry.registerArgument("att_invivo",
		                          "(Motion correction) In-vivo attenuation "
		                          "image filename",
		                          false, "", attenuationGroup);
		registry.registerArgument("acf_invivo",
		                          "(Motion correction) In-vivo attenuation "
		                          "correction factors histogram filename",
		                          false, "", attenuationGroup);
		registry.registerArgument(
		    "acf_invivo_format",
		    "(Motion correction) In-vivo attenuation correction factors "
		    "histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, "", attenuationGroup);
		registry.registerArgument(
		    "att_hardware",
		    "(Motion correction) Hardware attenuation image filename", false,
		    "", attenuationGroup);
		registry.registerArgument(
		    "acf_hardware",
		    "(Motion correction) Hardware attenuation correction factors",
		    false, "", attenuationGroup);
		registry.registerArgument(
		    "acf_hardware_format",
		    "(Motion correction) Hardware attenuation correction factors "
		    "histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, "", attenuationGroup);

		auto projectorGroup = "4. Projector";
		registry.registerArgument(
		    "projector",
		    "Projector to use, choices: Siddon (S), Distance-Driven (D). The "
		    "default projector is Siddon",
		    false, "S", projectorGroup);
		registry.registerArgument(
		    "num_rays", "Number of rays to use (for Siddon projector only)",
		    false, 1, projectorGroup);
		registry.registerArgument("proj_psf",
		                          "Projection-space PSF kernel file", false, "",
		                          projectorGroup);
		registry.registerArgument("tof_width_ps", "TOF Width in Picoseconds",
		                          false, 0.0f, projectorGroup);
		registry.registerArgument("tof_n_std",
		                          "Number of standard deviations to consider "
		                          "for TOF's Gaussian curve",
		                          false, 0, projectorGroup);

		PluginOptionsHelper::addOptionsFromPlugins(
		    registry, Plugin::InputFormatsChoice::ALL);

		// Load configuration
		IO::ArgumentReader config{registry};

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

		if (config.getValue<bool>("sens_only") &&
		    !config.getValue<bool>("move_sens"))
		{
			ASSERT_MSG(
			    config.getValue<std::vector<std::string>>("sens").empty(),
			    "Logic error: Sensitivity image generation was requested while "
			    "pre-existing sensitivity images were provided");
		}

		std::cout << "Initializing scanner..." << std::endl;
		auto scanner =
		    std::make_unique<Scanner>(config.getValue<std::string>("scanner"));
		auto projectorType =
		    IO::getProjector(config.getValue<std::string>("projector"));
		std::unique_ptr<OSEM> osem =
		    Util::createOSEM(*scanner, config.getValue<bool>("gpu"));

		osem->num_MLEM_iterations = config.getValue<int>("num_iterations");
		osem->num_OSEM_subsets = config.getValue<int>("num_subsets");
		osem->hardThreshold = config.getValue<float>("hard_threshold");
		osem->projectorType = projectorType;
		osem->numRays = config.getValue<int>("num_rays");
		Globals::set_num_threads(config.getValue<int>("num_threads"));

		// To make sure the sensitivity image gets generated accordingly
		const bool useListMode =
		    !config.getValue<std::string>("format").empty() &&
		    IO::isFormatListMode(config.getValue<std::string>("format"));
		osem->setListModeEnabled(useListMode);

		// Total attenuation image
		std::unique_ptr<ImageOwned> attImg = nullptr;
		std::unique_ptr<ProjectionData> acfHisProjData = nullptr;
		if (!config.getValue<std::string>("acf").empty())
		{
			std::cout << "Reading ACF histogram..." << std::endl;
			ASSERT_MSG(!config.getValue<std::string>("acf_format").empty(),
			           "Unspecified format for ACF histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("acf_format")),
			           "ACF has to be in a histogram format");

			acfHisProjData = IO::openProjectionData(
			    config.getValue<std::string>("acf"),
			    config.getValue<std::string>("acf_format"), *scanner,
			    config.getAllArguments());

			const auto* acfHis =
			    dynamic_cast<const Histogram*>(acfHisProjData.get());
			ASSERT(acfHis != nullptr);

			osem->setACFHistogram(acfHis);
		}
		else if (!config.getValue<std::string>("att").empty())
		{
			attImg = std::make_unique<ImageOwned>(
			    config.getValue<std::string>("att"));
			osem->setAttenuationImage(attImg.get());
		}

		// Hardware attenuation image
		std::unique_ptr<ImageOwned> hardwareAttImg = nullptr;
		std::unique_ptr<ProjectionData> hardwareAcfHisProjData = nullptr;
		if (!config.getValue<std::string>("acf_hardware").empty())
		{
			std::cout << "Reading hardware ACF histogram..." << std::endl;
			ASSERT_MSG(
			    !config.getValue<std::string>("acf_hardware_format").empty(),
			    "No format specified for hardware ACF histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("acf_hardware_format")),
			           "Hardware ACF has to be in a histogram format");

			hardwareAcfHisProjData = IO::openProjectionData(
			    config.getValue<std::string>("acf_hardware"),
			    config.getValue<std::string>("acf_hardware_format"), *scanner,
			    config.getAllArguments());

			const auto* hardwareAcfHis =
			    dynamic_cast<const Histogram*>(hardwareAcfHisProjData.get());
			ASSERT(hardwareAcfHis != nullptr);

			osem->setACFHistogram(hardwareAcfHis);
		}
		else if (!config.getValue<std::string>("att_hardware").empty())
		{
			hardwareAttImg = std::make_unique<ImageOwned>(
			    config.getValue<std::string>("att_hardware"));
			osem->setHardwareAttenuationImage(hardwareAttImg.get());
		}

		// Image-space PSF
		if (!config.getValue<std::string>("psf").empty())
		{
			osem->addImagePSF(config.getValue<std::string>("psf"));
		}

		// Projection-space PSF
		if (!config.getValue<std::string>("proj_psf").empty())
		{
			osem->addProjPSF(config.getValue<std::string>("proj_psf"));
		}

		// Sensitivity image(s)
		std::unique_ptr<ProjectionData> sensitivityProjData = nullptr;
		if (!config.getValue<std::string>("sensitivity").empty())
		{
			std::cout << "Reading sensitivity histogram..." << std::endl;
			ASSERT_MSG(
			    !config.getValue<std::string>("sensitivity_format").empty(),
			    "No format specified for sensitivity histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("sensitivity_format")),
			           "Sensitivity data has to be in a histogram format");

			sensitivityProjData = IO::openProjectionData(
			    config.getValue<std::string>("sensitivity"),
			    config.getValue<std::string>("sensitivity_format"), *scanner,
			    config.getAllArguments());

			const auto* sensitivityHis =
			    dynamic_cast<const Histogram*>(sensitivityProjData.get());
			ASSERT(sensitivityHis != nullptr);

			osem->setSensitivityHistogram(sensitivityHis);
			osem->setInvertSensitivity(
			    config.getValue<bool>("invert_sensitivity"));
		}
		osem->setGlobalScalingFactor(config.getValue<float>("global_scale"));

		std::vector<std::unique_ptr<Image>> sensImages;
		bool sensImageAlreadyMoved = false;
		if (config.getValue<std::vector<std::string>>("sens").empty())
		{
			ASSERT_MSG(!config.getValue<std::string>("params").empty(),
			           "Image parameters file unspecified");
			ImageParams imgParams{config.getValue<std::string>("params")};
			osem->setImageParams(imgParams);

			osem->generateSensitivityImages(
			    sensImages, config.getValue<std::string>("out_sens"));
		}
		else if (osem->getExpectedSensImagesAmount() ==
		         static_cast<int>(
		             config.getValue<std::vector<std::string>>("sens").size()))
		{
			std::cout << "Reading sensitivity images..." << std::endl;
			for (auto& sensImg_fname :
			     config.getValue<std::vector<std::string>>("sens"))
			{
				sensImages.push_back(
				    std::make_unique<ImageOwned>(sensImg_fname));
			}
			sensImageAlreadyMoved = !config.getValue<bool>("move_sens");
		}
		else
		{
			std::cerr
			    << "The number of sensitivity images given is "
			    << config.getValue<std::vector<std::string>>("sens").size()
			    << std::endl;
			std::cerr << "The expected number of sensitivity images is "
			          << osem->getExpectedSensImagesAmount() << std::endl;
			throw std::invalid_argument(
			    "The number of sensitivity images given "
			    "doesn't match the number of "
			    "subsets specified. Note: For ListMode formats, exactly one "
			    "sensitivity image is required.");
		}

		// No need to read data input if in sens_only mode
		if (config.getValue<bool>("sens_only") &&
		    config.getValue<std::string>("input").empty())
		{
			std::cout << "Done." << std::endl;
			return 0;
		}

		// Projection Data Input file
		std::cout << "Reading input data..." << std::endl;
		std::unique_ptr<ProjectionData> dataInput;
		ASSERT_MSG(!config.getValue<std::string>("format").empty(),
		           "No format specified for Data input");
		dataInput =
		    IO::openProjectionData(config.getValue<std::string>("input"),
		                           config.getValue<std::string>("format"),
		                           *scanner, config.getAllArguments());
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

			if (!config.getValue<std::string>("out_sens").empty())
			{
				// Overwrite sensitivity image
				std::cout << "Saving sensitivity image..." << std::endl;
				movedSensImage->writeToFile(
				    config.getValue<std::string>("out_sens"));
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

		if (config.getValue<bool>("sens_only"))
		{
			std::cout << "Done." << std::endl;
			return 0;
		}

		if (config.getValue<float>("tof_width_ps") > 0.f)
		{
			osem->addTOF(config.getValue<float>("tof_width_ps"),
			             config.getValue<int>("tof_n_std"));
		}

		// Additive histograms
		std::unique_ptr<ProjectionData> randomsProjData = nullptr;
		if (!config.getValue<std::string>("randoms").empty())
		{
			std::cout << "Reading randoms histogram..." << std::endl;
			ASSERT_MSG(!config.getValue<std::string>("randoms_format").empty(),
			           "No format specified for randoms histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("randoms_format")),
			           "Randoms must be specified in histogram format");

			randomsProjData = IO::openProjectionData(
			    config.getValue<std::string>("randoms"),
			    config.getValue<std::string>("randoms_format"), *scanner,
			    config.getAllArguments());
			const auto* randomsHis =
			    dynamic_cast<const Histogram*>(randomsProjData.get());
			ASSERT_MSG(randomsHis != nullptr,
			           "The randoms histogram provided does not inherit from "
			           "Histogram.");
			osem->setRandomsHistogram(randomsHis);
		}
		std::unique_ptr<ProjectionData> scatterProjData = nullptr;
		if (!config.getValue<std::string>("scatter").empty())
		{
			std::cout << "Reading scatter histogram..." << std::endl;
			ASSERT_MSG(!config.getValue<std::string>("scatter_format").empty(),
			           "No format specified for scatter histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("scatter_format")),
			           "Scatter must be specified in histogram format");

			scatterProjData = IO::openProjectionData(
			    config.getValue<std::string>("scatter"),
			    config.getValue<std::string>("scatter_format"), *scanner,
			    config.getAllArguments());
			const auto* scatterHis =
			    dynamic_cast<const Histogram*>(scatterProjData.get());
			ASSERT_MSG(scatterHis != nullptr,
			           "The scatter histogram provided does not inherit from "
			           "Histogram.");
			osem->setScatterHistogram(scatterHis);
		}

		std::unique_ptr<ImageOwned> invivoAttImg = nullptr;
		if (!config.getValue<std::string>("att_invivo").empty())
		{
			ASSERT_MSG_WARNING(dataInput->hasMotion(),
			                   "An in-vivo attenuation image was provided but "
			                   "the data input has no motion");
			invivoAttImg = std::make_unique<ImageOwned>(
			    config.getValue<std::string>("att_invivo"));
			osem->setInVivoAttenuationImage(invivoAttImg.get());
		}
		std::unique_ptr<ProjectionData> inVivoAcfProjData = nullptr;
		if (!config.getValue<std::string>("acf_invivo").empty())
		{
			std::cout << "Reading in-vivo ACF histogram..." << std::endl;
			ASSERT_MSG(
			    !config.getValue<std::string>("acf_invivo_format").empty(),
			    "No format specified for ACF histogram");
			ASSERT_MSG(!IO::isFormatListMode(
			               config.getValue<std::string>("acf_invivo_format")),
			           "In-vivo ACF must be specified in histogram format");

			inVivoAcfProjData = IO::openProjectionData(
			    config.getValue<std::string>("acf_invivo"),
			    config.getValue<std::string>("acf_invivo_format"), *scanner,
			    config.getAllArguments());
			const auto* inVivoAcfHis =
			    dynamic_cast<const Histogram*>(inVivoAcfProjData.get());
			ASSERT_MSG(
			    inVivoAcfHis != nullptr,
			    "The in-vivo ACF histogram provided does not inherit from "
			    "Histogram.");
			osem->setInVivoACFHistogram(inVivoAcfHis);
		}

		// Save steps
		ASSERT_MSG(config.getValue<int>("save_iter_step") >= 0,
		           "save_iter_step must be positive.");
		Util::RangeList ranges;
		if (config.getValue<int>("save_iter_step") > 0)
		{
			if (config.getValue<int>("save_iter_step") == 1)
			{
				ranges.insertSorted(0,
				                    config.getValue<int>("num_iterations") - 1);
			}
			else
			{
				for (int it = 0; it < config.getValue<int>("num_iterations");
				     it += config.getValue<int>("save_iter_step"))
				{
					ranges.insertSorted(it, it);
				}
			}
		}
		else if (!config.getValue<std::string>("save_iter_ranges").empty())
		{
			ranges.readFromString(
			    config.getValue<std::string>("save_iter_ranges"));
		}
		if (!ranges.empty())
		{
			osem->setSaveIterRanges(ranges,
			                        config.getValue<std::string>("out"));
		}

		// Initial image estimate
		std::unique_ptr<ImageOwned> initialEstimate = nullptr;
		if (!config.getValue<std::string>("initial_estimate").empty())
		{
			initialEstimate = std::make_unique<ImageOwned>(
			    config.getValue<std::string>("initial_estimate"));
			osem->initialEstimate = initialEstimate.get();
		}

		std::cout << "Launching reconstruction..." << std::endl;
		osem->reconstruct(config.getValue<std::string>("out"));

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
