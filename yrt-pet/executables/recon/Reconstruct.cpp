/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../ArgumentReader.hpp"
#include "../PluginOptionsHelper.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplay.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/utils/Timer.hpp"

#include <ctime>
#include <cxxopts.hpp>
#include <iostream>

using namespace yrt;

void printTimingStatistics(const util::Timer& ioTimer,
                           const util::Timer& sensTimer,
                           const util::Timer& reconTimer)
{
	std::cout << "I/O time: " << ioTimer.getElapsedSeconds() << "s"
	          << std::endl;
	std::cout << "Sensitivity generation time: "
	          << sensTimer.getElapsedSeconds() << "s" << std::endl;
	std::cout << "Reconstruction time: " << reconTimer.getElapsedSeconds()
	          << "s" << std::endl;
}

void addImagePSFtoReconIfNeeded(OSEM& osem, std::string psf_fname,
                                std::string varpsf_fname)
{
	if (!osem.hasImagePSF())
	{
		if (!psf_fname.empty())
		{
			ASSERT_MSG(varpsf_fname.empty(),
			           "Got two different image PSF inputs");
			osem.addImagePSF(psf_fname, ImagePSFMode::UNIFORM);
		}
		else if (!varpsf_fname.empty())
		{
			osem.addImagePSF(varpsf_fname, ImagePSFMode::VARIANT);
		}
	}
}

int main(int argc, char** argv)
{
	try
	{
		io::ArgumentRegistry registry{};

		std::string coreGroup = "0. Core";
		std::string sensitivityGroup = "1. Sensitivity";
		std::string inputGroup = "2. Input";
		std::string reconstructionGroup = "3. Reconstruction";
		std::string attenuationGroup = "3.1 Attenuation correction";
		std::string projectorGroup = "4. Projector";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "s");

		registry.registerArgument(
		    "params",
		    "Image parameters file. Note: If sensitivity image(s) are "
		    "provided, "
		    "the image parameters will be determined from them",
		    false, io::TypeOfArgument::STRING, "", coreGroup, "p");

		registry.registerArgument(
		    "sens_only",
		    "Only generate sensitivity image(s). Do not launch reconstruction",
		    false, io::TypeOfArgument::BOOL, false, coreGroup);

#if BUILD_CUDA
		registry.registerArgument("gpu", "Use GPU acceleration", false,
		                          io::TypeOfArgument::BOOL, false, coreGroup);
#endif
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, io::TypeOfArgument::INT, -1,
		                          coreGroup);

		registry.registerArgument("out", "Output image filename", false,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "o");

		registry.registerArgument(
		    "out_sens",
		    "Sensitivity image output filename (if it needs to be computed). "
		    "Leave blank to not save it",
		    false, io::TypeOfArgument::STRING, "", coreGroup);

		// Sensitivity parameters
		registry.registerArgument(
		    "sens",
		    "Sensitivity image files (separated by a comma). Note: When the "
		    "input is a List-mode, one sensitivity image is required. When the "
		    "input is a histogram, one sensitivity image *per subset* is "
		    "required (Ordered by subset id)",
		    false, io::TypeOfArgument::VECTOR_OF_STRINGS,
		    std::vector<std::string>{}, sensitivityGroup);
		registry.registerArgument("sensitivity", "Sensitivity histogram file",
		                          false, io::TypeOfArgument::STRING, "",
		                          sensitivityGroup);
		registry.registerArgument(
		    "sensitivity_format",
		    "Sensitivity histogram format. Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, io::TypeOfArgument::STRING, "", sensitivityGroup);
		registry.registerArgument("invert_sensitivity",
		                          "Invert the sensitivity histogram values "
		                          "(sensitivity -> 1/sensitivity)",
		                          false, io::TypeOfArgument::BOOL, false,
		                          sensitivityGroup);
		registry.registerArgument(
		    "global_scale", "Global scaling factor to apply on the sensitivity",
		    false, io::TypeOfArgument::FLOAT, 1.0f, sensitivityGroup);
		registry.registerArgument(
		    "move_sens", "Move the provided sensitivity image based on motion",
		    false, io::TypeOfArgument::BOOL, false, sensitivityGroup);

		// Input data parameters
		registry.registerArgument("input", "Input file", false,
		                          io::TypeOfArgument::STRING, "", inputGroup,
		                          "i");
		registry.registerArgument(
		    "format",
		    "Input file format. Possible values: " + io::possibleFormats(),
		    false, io::TypeOfArgument::STRING, "", inputGroup, "f");
		registry.registerArgument(
		    "lor_motion", "Motion CSV file for motion correction", false,
		    io::TypeOfArgument::STRING, "", inputGroup, "m");

		// Reconstruction parameters
		registry.registerArgument(
		    "num_iterations", "Number of MLEM iterations (Default: 10)", false,
		    io::TypeOfArgument::INT, 10, reconstructionGroup);
		registry.registerArgument(
		    "num_subsets", "Number of OSEM subsets (Default: 1)", false,
		    io::TypeOfArgument::INT, 1, reconstructionGroup);
		registry.registerArgument(
		    "initial_estimate", "Initial image estimate for the MLEM", false,
		    io::TypeOfArgument::STRING, "", reconstructionGroup);
		registry.registerArgument(
		    "randoms", "Randoms estimate histogram filename", false,
		    io::TypeOfArgument::STRING, "", reconstructionGroup);
		registry.registerArgument(
		    "randoms_format",
		    "Randoms estimate histogram format. Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, io::TypeOfArgument::STRING, "", reconstructionGroup);
		registry.registerArgument(
		    "scatter", "Scatter estimate histogram filename", false,
		    io::TypeOfArgument::STRING, "", reconstructionGroup);
		registry.registerArgument(
		    "scatter_format",
		    "Scatter estimate histogram format. Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, io::TypeOfArgument::STRING, "", reconstructionGroup);
		registry.registerArgument("psf", "Image-space PSF kernel file", false,
		                          io::TypeOfArgument::STRING, "",
		                          reconstructionGroup);
		registry.registerArgument(
		    "varpsf", "Image-space Variant PSF look-up table file", false,
		    io::TypeOfArgument::STRING, "", reconstructionGroup);
		registry.registerArgument("hard_threshold", "Hard Threshold", false,
		                          io::TypeOfArgument::FLOAT, 1.0f,
		                          reconstructionGroup);
		registry.registerArgument(
		    "save_iter_step",
		    "Increment into which to save MLEM iteration images", false,
		    io::TypeOfArgument::INT, 0, reconstructionGroup);
		registry.registerArgument(
		    "save_iter_ranges",
		    "List of iteration ranges to save MLEM iteration images", false,
		    io::TypeOfArgument::STRING, "", reconstructionGroup);

		registry.registerArgument("att", "Total attenuation image filename",
		                          false, io::TypeOfArgument::STRING, "",
		                          attenuationGroup);
		registry.registerArgument(
		    "acf", "Total attenuation correction factors histogram filename",
		    false, io::TypeOfArgument::STRING, "", attenuationGroup);
		registry.registerArgument(
		    "acf_format",
		    "Total attenuation correction factors histogram format. Possible "
		    "values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, io::TypeOfArgument::STRING, "", attenuationGroup);
		registry.registerArgument("att_invivo",
		                          "(Motion correction) In-vivo attenuation "
		                          "image filename",
		                          false, io::TypeOfArgument::STRING, "",
		                          attenuationGroup);
		registry.registerArgument("acf_invivo",
		                          "(Motion correction) In-vivo attenuation "
		                          "correction factors histogram filename",
		                          false, io::TypeOfArgument::STRING, "",
		                          attenuationGroup);
		registry.registerArgument(
		    "acf_invivo_format",
		    "(Motion correction) In-vivo attenuation correction factors "
		    "histogram format. Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, io::TypeOfArgument::STRING, "", attenuationGroup);
		registry.registerArgument(
		    "att_hardware",
		    "(Motion correction) Hardware attenuation image filename", false,
		    io::TypeOfArgument::STRING, "", attenuationGroup);
		registry.registerArgument(
		    "acf_hardware",
		    "(Motion correction) Hardware attenuation correction factors",
		    false, io::TypeOfArgument::STRING, "", attenuationGroup);
		registry.registerArgument(
		    "acf_hardware_format",
		    "(Motion correction) Hardware attenuation correction factors "
		    "histogram format. Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, io::TypeOfArgument::STRING, "", attenuationGroup);

		registry.registerArgument(
		    "projector",
		    "Projector to use, choices: Siddon (S), Distance-Driven (D). The "
		    "default projector is Siddon",
		    false, io::TypeOfArgument::STRING, "S", projectorGroup);
		registry.registerArgument(
		    "num_rays", "Number of rays to use (for Siddon projector only)",
		    false, io::TypeOfArgument::INT, 1, projectorGroup);
		registry.registerArgument(
		    "proj_psf",
		    "Projection-space PSF kernel file (for DD projector only)", false,
		    io::TypeOfArgument::STRING, "", projectorGroup);
		registry.registerArgument("tof_width_ps", "TOF width in picoseconds",
		                          false, io::TypeOfArgument::FLOAT, 0.0f,
		                          projectorGroup);
		registry.registerArgument("tof_n_std",
		                          "Number of standard deviations to consider "
		                          "for TOF's Gaussian curve. Default: 5",
		                          false, io::TypeOfArgument::INT, 5,
		                          projectorGroup);

		plugin::addOptionsFromPlugins(registry,
		                              plugin::InputFormatsChoice::ALL);

		// Load configuration
		io::ArgumentReader config{registry, "Reconstruction executable"};

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

		const bool sensOnly = config.getValue<bool>("sens_only");

		if (sensOnly && !config.getValue<bool>("move_sens"))
		{
			ASSERT_MSG(
			    config.getValue<std::vector<std::string>>("sens").empty(),
			    "Logic error: Sensitivity image generation was requested while "
			    "pre-existing sensitivity image(s) were provided.");
		}

		const auto dataInputFormat = config.getValue<std::string>("format");
		const auto dataInputFilename = config.getValue<std::string>("input");
		ASSERT_MSG(dataInputFilename.empty() == dataInputFormat.empty(),
		           "Input data must come with a specified format");

		if (!sensOnly)
		{
			ASSERT_MSG(!config.getValue<std::string>("out").empty(),
			           "Output reconstruction image filename unspecified.");
			ASSERT_MSG(!dataInputFilename.empty(), "Input data unspecified.");
			ASSERT_MSG(!dataInputFormat.empty(),
			           "No format specified for input data.");
		}

		util::Timer ioTimer, sensTimer, reconTimer;

		ioTimer.run();

		globals::setNumThreads(config.getValue<int>("num_threads"));
		std::cout << "Initializing scanner..." << std::endl;
		auto scanner =
		    std::make_unique<Scanner>(config.getValue<std::string>("scanner"));
		auto projectorType =
		    io::getProjector(config.getValue<std::string>("projector"));
		std::unique_ptr<OSEM> osem =
		    util::createOSEM(*scanner, config.getValue<bool>("gpu"));

		osem->num_MLEM_iterations = config.getValue<int>("num_iterations");
		osem->num_OSEM_subsets = config.getValue<int>("num_subsets");
		osem->hardThreshold = config.getValue<float>("hard_threshold");
		osem->projectorType = projectorType;
		osem->numRays = config.getValue<int>("num_rays");

		// To make sure the sensitivity image gets generated accordingly
		const bool useListMode =
		    !dataInputFormat.empty() && io::isFormatListMode(dataInputFormat);
		osem->setListModeEnabled(useListMode);

		// Total attenuation image
		std::unique_ptr<ImageOwned> attImg = nullptr;
		std::unique_ptr<ProjectionData> acfHisProjData = nullptr;
		if (!config.getValue<std::string>("acf").empty())
		{
			std::cout << "Reading ACF histogram..." << std::endl;
			ASSERT_MSG(!config.getValue<std::string>("acf_format").empty(),
			           "Unspecified format for ACF histogram.");
			ASSERT_MSG(!io::isFormatListMode(
			               config.getValue<std::string>("acf_format")),
			           "ACF has to be in a histogram format.");

			acfHisProjData = io::openProjectionData(
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
			std::cout << "Reading attenuation image..." << std::endl;
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
			ASSERT_MSG(!io::isFormatListMode(
			               config.getValue<std::string>("acf_hardware_format")),
			           "Hardware ACF has to be in a histogram format.");

			hardwareAcfHisProjData = io::openProjectionData(
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
			std::cout << "Reading hardware attenuation image..." << std::endl;
			hardwareAttImg = std::make_unique<ImageOwned>(
			    config.getValue<std::string>("att_hardware"));
			osem->setHardwareAttenuationImage(hardwareAttImg.get());
		}

		// Projection-space PSF
		if (!config.getValue<std::string>("proj_psf").empty())
		{
			osem->addProjPSF(config.getValue<std::string>("proj_psf"));
		}

		// Sensitivity histogram and global scale factor
		std::unique_ptr<ProjectionData> sensitivityProjData = nullptr;
		if (!config.getValue<std::string>("sensitivity").empty())
		{
			std::cout << "Reading sensitivity histogram..." << std::endl;
			ASSERT_MSG(
			    !config.getValue<std::string>("sensitivity_format").empty(),
			    "No format specified for sensitivity histogram.");
			ASSERT_MSG(!io::isFormatListMode(
			               config.getValue<std::string>("sensitivity_format")),
			           "Sensitivity data has to be in a histogram format.");

			sensitivityProjData = io::openProjectionData(
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

		// Sensitivity image(s)
		std::vector<std::unique_ptr<Image>> sensImages;
		bool needToMoveSensImage = true;
		if (config.getValue<std::vector<std::string>>("sens").empty())
		{
			ASSERT_MSG(!config.getValue<std::string>("params").empty(),
			           "Image parameters file unspecified");
			ImageParams imgParams{config.getValue<std::string>("params")};
			osem->setImageParams(imgParams);

			addImagePSFtoReconIfNeeded(*osem,
			                           config.getValue<std::string>("psf"),
			                           config.getValue<std::string>("varpsf"));

			ioTimer.pause();
			sensTimer.run();

			std::cout << "Generating sensitivity image(s)..." << std::endl;
			osem->generateSensitivityImages(
			    sensImages, config.getValue<std::string>("out_sens"));

			sensTimer.pause();
			ioTimer.run();
		}
		else if (osem->getExpectedSensImagesAmount() ==
		         static_cast<int>(
		             config.getValue<std::vector<std::string>>("sens").size()))
		{
			std::cout << "Reading sensitivity image(s)..." << std::endl;
			for (auto& sensImg_fname :
			     config.getValue<std::vector<std::string>>("sens"))
			{
				sensImages.push_back(
				    std::make_unique<ImageOwned>(sensImg_fname));
			}
			needToMoveSensImage = config.getValue<bool>("move_sens");
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

		const std::string lorMotion_fname =
		    config.getValue<std::string>("lor_motion");

		// No need to read data input if in sens_only mode
		if (sensOnly && lorMotion_fname.empty())
		{
			printTimingStatistics(ioTimer, sensTimer, reconTimer);
			std::cout << "Done." << std::endl;
			return 0;
		}

		std::unique_ptr<ProjectionData> dataInput;
		if (!dataInputFilename.empty())
		{
			// Projection data Input file
			std::cout << "Reading input data..." << std::endl;
			dataInput =
			    io::openProjectionData(dataInputFilename, dataInputFormat,
			                           *scanner, config.getAllArguments());
			osem->setDataInput(dataInput.get());
		}

		std::shared_ptr<LORMotion> lorMotion;
		std::unique_ptr<ImageOwned> movedSensImage = nullptr;
		if (!lorMotion_fname.empty())
		{
			lorMotion = std::make_shared<LORMotion>(lorMotion_fname);

			if (needToMoveSensImage)
			{
				ASSERT(sensImages.size() == 1);
				const Image* unmovedSensImage = sensImages[0].get();
				ASSERT(unmovedSensImage != nullptr);

				ioTimer.pause();
				sensTimer.run();

				std::cout << "Moving sensitivity image..." << std::endl;
				if (dataInput == nullptr)
				{
					// Time average move based on all the frames
					movedSensImage = util::timeAverageMoveImage(
					    *lorMotion, unmovedSensImage);
				}
				else
				{
					// Time average move based on the frames of the listmode
					const timestamp_t timeStart = dataInput->getTimestamp(0);
					const timestamp_t timeStop =
					    dataInput->getTimestamp(dataInput->count() - 1);
					movedSensImage = util::timeAverageMoveImage(
					    *lorMotion, unmovedSensImage, timeStart, timeStop);
				}

				sensTimer.pause();
				ioTimer.run();

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
				std::cout << "Sensitivity image(s) already provided. No need "
				             "for time-averaging."
				          << std::endl;
				osem->setSensitivityImages(sensImages);
			}
		}
		else
		{
			std::cout
			    << "No motion file given. No need to move sensitivity image."
			    << std::endl;
			osem->setSensitivityImages(sensImages);
		}

		if (sensOnly)
		{
			printTimingStatistics(ioTimer, sensTimer, reconTimer);
			std::cout << "Done." << std::endl;
			return 0;
		}

		addImagePSFtoReconIfNeeded(*osem, config.getValue<std::string>("psf"),
		                           config.getValue<std::string>("varpsf"));

		if (lorMotion != nullptr)
		{
			if (useListMode)
			{
				// Input data as listmode
				auto* dataInput_lm = dynamic_cast<ListMode*>(dataInput.get());
				ASSERT_MSG(dataInput_lm != nullptr,
				           "(Unexpected error) Input data has to be in "
				           "ListMode format to include motion correction");

				// Link input data to LOR motion (to allow event-by-event motion
				//  correction)
				dataInput_lm->addLORMotion(lorMotion);
			}
			else
			{
				std::cerr << "Warning: Event-by-event motion correction is not "
				             "available for Histogram data"
				          << std::endl;
			}
		}

		if (config.getValue<float>("tof_width_ps") > 0.f)
		{
			ASSERT_MSG(
			    dataInput->hasTOF(),
			    "The input file does not have TOF data, yet, TOF configuration "
			    "was still provided to the reconstruction options");
			osem->addTOF(config.getValue<float>("tof_width_ps"),
			             config.getValue<int>("tof_n_std"));
		}
		else
		{
			ASSERT_MSG_WARNING(
			    !dataInput->hasTOF(),
			    "The input file has TOF data, but the TOF width "
			    "was not provided to the reconstruction options");
		}

		// Additive histograms
		std::unique_ptr<ProjectionData> randomsProjData = nullptr;
		if (!config.getValue<std::string>("randoms").empty())
		{
			std::cout << "Reading randoms histogram..." << std::endl;
			ASSERT_MSG(!config.getValue<std::string>("randoms_format").empty(),
			           "No format specified for randoms histogram");
			ASSERT_MSG(!io::isFormatListMode(
			               config.getValue<std::string>("randoms_format")),
			           "Randoms must be specified in histogram format");

			randomsProjData = io::openProjectionData(
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
			ASSERT_MSG(!io::isFormatListMode(
			               config.getValue<std::string>("scatter_format")),
			           "Scatter must be specified in histogram format");

			scatterProjData = io::openProjectionData(
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
			std::cout << "Reading in-vivo attenuation image..." << std::endl;
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
			ASSERT_MSG(!io::isFormatListMode(
			               config.getValue<std::string>("acf_invivo_format")),
			           "In-vivo ACF must be specified in histogram format");

			inVivoAcfProjData = io::openProjectionData(
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
		util::RangeList ranges;
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

		ioTimer.pause();
		reconTimer.run();

		std::cout << "Launching reconstruction..." << std::endl;
		osem->reconstruct(config.getValue<std::string>("out"));

		reconTimer.pause();

		printTimingStatistics(ioTimer, sensTimer, reconTimer);

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
		util::printExceptionMessage(e);
		return -1;
	}
}
