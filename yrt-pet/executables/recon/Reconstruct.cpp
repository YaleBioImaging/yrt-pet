/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

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
		std::string scanner_fname;
		std::string imgParams_fname;
		std::string input_fname;
		std::string input_format;
		std::vector<std::string> sensImg_fnames;
		std::string initialEstimate_fname;
		std::string attImg_fname;
		std::string acf_fname;
		std::string acf_format;
		std::string invivoAttImg_fname;
		std::string invivoAcf_fname;
		std::string invivoAcf_format;
		std::string hardwareAttImg_fname;
		std::string hardwareAcf_fname;
		std::string hardwareAcf_format;
		std::string imageSpacePsf_fname;
		std::string projSpacePsf_fname;
		std::string randoms_fname;
		std::string randoms_format;
		std::string scatter_fname;
		std::string scatter_format;
		std::string projector_name = "S";
		std::string sensitivityData_fname;
		std::string sensitivityData_format;
		std::string out_fname;
		std::string out_sensImg_fname;
		int numIterations = 10;
		int numSubsets = 1;
		int numThreads = -1;
		int numRays = 1;
		float hardThreshold = 1.0f;
		float tofWidth_ps = 0.0f;
		float globalScalingFactor = 1.0f;
		int tofNumStd = 0;
		int saveIterStep = 0;
		std::string saveIterRanges;
		bool sensOnly = false;
		bool mustMoveSens = false;
		bool invertSensitivity = false;

		Plugin::OptionsResult pluginOptionsResults;  // For plugins' options

		// Parse command line arguments
		cxxopts::Options options(argv[0], "Reconstruction executable");
		options.positional_help("[optional args]").show_positional_help();

		auto coreGroup = options.add_options("0. Core");
		coreGroup("s,scanner", "Scanner parameters file",
		          cxxopts::value<std::string>(scanner_fname));
		coreGroup("p,params",
		          "Image parameters file."
		          "Note: If sensitivity image(s) are provided,"
		          "the image parameters will be determined from them.",
		          cxxopts::value<std::string>(imgParams_fname));
		coreGroup("sens_only",
		          "Only generate the sensitivity image(s)."
		          "Do not launch reconstruction",
		          cxxopts::value<bool>(sensOnly));
		coreGroup("num_threads", "Number of threads to use",
		          cxxopts::value<int>(numThreads));
		coreGroup("o,out", "Output image filename",
		          cxxopts::value<std::string>(out_fname));
		coreGroup("out_sens",
		          "Filename for the generated sensitivity image (if it needed "
		          "to be computed)."
		          "Leave blank to not save it",
		          cxxopts::value<std::string>(out_sensImg_fname));

		auto sensGroup = options.add_options("1. Sensitivity");
		sensGroup("sens",
		          "Sensitivity image files (separated by a comma). Note: When "
		          "the input is a List-mode, one sensitivity image is required."
		          "When the input is a histogram, one sensitivity image *per "
		          "subset* is required (Ordered by subset id)",
		          cxxopts::value<std::vector<std::string>>(sensImg_fnames));
		sensGroup("sensitivity", "Sensitivity histogram file",
		          cxxopts::value<std::string>(sensitivityData_fname));
		sensGroup(
		    "sensitivity_format",
		    "Sensitivity histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    cxxopts::value<std::string>(sensitivityData_format));
		sensGroup("invert_sensitivity",
		          "Invert the sensitivity histogram values (sensitivity -> "
		          "1/sensitivity)",
		          cxxopts::value<bool>(invertSensitivity));
		sensGroup("global_scale",
		          "Global scaling factor to apply on the sensitivity",
		          cxxopts::value<float>(globalScalingFactor));
		sensGroup("move_sens",
		          "Move the provided sensitivity image based on motion",
		          cxxopts::value<bool>(mustMoveSens));

		auto inputGroup = options.add_options("2. Input");
		inputGroup("i,input", "Input file",
		           cxxopts::value<std::string>(input_fname));
		inputGroup("f,format",
		           "Input file format. Possible values: " +
		               IO::possibleFormats(),
		           cxxopts::value<std::string>(input_format));

		auto reconGroup = options.add_options("3. Reconstruction");
		reconGroup("num_iterations", "Number of MLEM Iterations",
		           cxxopts::value<int>(numIterations));
		reconGroup("num_subsets", "Number of OSEM subsets (Default: 1)",
		           cxxopts::value<int>(numSubsets));
		reconGroup("initial_estimate", "Initial image estimate for the MLEM",
		           cxxopts::value<std::string>(initialEstimate_fname));
		reconGroup("randoms", "Randoms estimate histogram filename",
		           cxxopts::value<std::string>(randoms_fname));
		reconGroup(
		    "randoms_format",
		    "Randoms estimate histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    cxxopts::value<std::string>(randoms_format));
		reconGroup("scatter", "Scatter estimate histogram filename",
		           cxxopts::value<std::string>(scatter_fname));
		reconGroup(
		    "scatter_format",
		    "Scatter estimate histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    cxxopts::value<std::string>(scatter_format));
		reconGroup("psf", "Image-space PSF kernel file",
		           cxxopts::value<std::string>(imageSpacePsf_fname));
		reconGroup("hard_threshold", "Hard Threshold",
		           cxxopts::value<float>(hardThreshold));
		reconGroup("save_iter_step",
		           "Increment into which to save MLEM iteration images",
		           cxxopts::value<int>(saveIterStep));
		reconGroup("save_iter_ranges",
		           "List of iteration ranges to save MLEM iteration images",
		           cxxopts::value<std::string>(saveIterRanges));

		auto attenuationGroup =
		    options.add_options("3.1 Attenuation correction");
		attenuationGroup("att", "Total attenuation image filename",
		                 cxxopts::value<std::string>(attImg_fname));
		attenuationGroup(
		    "acf", "Total attenuation correction factors histogram filename",
		    cxxopts::value<std::string>(acf_fname));
		attenuationGroup(
		    "acf_format",
		    "Total attenuation correction factors histogram format. Possible "
		    "values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    cxxopts::value<std::string>(acf_format));
		attenuationGroup("att_invivo",
		                 "(Motion correction) In-vivo attenuation "
		                 "image filename",
		                 cxxopts::value<std::string>(invivoAttImg_fname));
		attenuationGroup("acf_invivo",
		                 "(Motion correction) In-vivo attenuation "
		                 "correction factors histogram filename",
		                 cxxopts::value<std::string>(invivoAcf_fname));
		attenuationGroup(
		    "acf_invivo_format",
		    "(Motion correction) In-vivo attenuation correction factors "
		    "histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    cxxopts::value<std::string>(invivoAcf_fname));
		attenuationGroup(
		    "att_hardware",
		    "(Motion correction) Hardware attenuation image filename",
		    cxxopts::value<std::string>(hardwareAttImg_fname));
		attenuationGroup(
		    "acf_hardware",
		    "(Motion correction) Hardware attenuation correction factors",
		    cxxopts::value<std::string>(hardwareAcf_fname));
		attenuationGroup(
		    "acf_hardware_format",
		    "(Motion correction) Hardware attenuation correction factors "
		    "histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    cxxopts::value<std::string>(hardwareAcf_format));

		auto projectorGroup = options.add_options("4. Projector");
		projectorGroup(
		    "projector",
		    "Projector to use, choices: Siddon (S), Distance-Driven (D)"
#if BUILD_CUDA
		    ", or GPU Distance-Driven (DD_GPU)"
#endif
		    ". The default projector is Siddon",
		    cxxopts::value<std::string>(projector_name));
		projectorGroup("num_rays",
		               "Number of rays to use (for Siddon projector only)",
		               cxxopts::value<int>(numRays));
		projectorGroup("proj_psf", "Projection-space PSF kernel file",
		               cxxopts::value<std::string>(projSpacePsf_fname));
		projectorGroup("tof_width_ps", "TOF Width in Picoseconds",
		               cxxopts::value<float>(tofWidth_ps));
		projectorGroup("tof_n_std",
		               "Number of standard deviations to consider for TOF's "
		               "Gaussian curve",
		               cxxopts::value<int>(tofNumStd));

		options.add_options()("h,help", "Print help");

		// Add plugin options
		PluginOptionsHelper::fillOptionsFromPlugins(options);

		const auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		std::vector<std::string> requiredParams = {"scanner"};
		std::vector<std::string> requiredParamsIfSensOnly = {"out_sens"};
		std::vector<std::string> requiredParamsIfRecon = {"input", "format",
		                                                  "out"};
		std::vector<std::string>& requiredParamsToAdd =
		    sensOnly ? requiredParamsIfSensOnly : requiredParamsIfRecon;
		requiredParams.insert(requiredParams.begin(),
		                      requiredParamsToAdd.begin(),
		                      requiredParamsToAdd.end());
		bool missing_args = false;
		for (auto& p : requiredParams)
		{
			if (result.count(p) == 0)
			{
				std::cerr << "Argument '" << p << "' missing" << std::endl;
				missing_args = true;
			}
		}
		if (missing_args)
		{
			std::cerr << options.help() << std::endl;
			return -1;
		}

		// Parse plugin options
		pluginOptionsResults =
		    PluginOptionsHelper::convertPluginResultsToMap(result);

		if (sensOnly && !mustMoveSens)
		{
			ASSERT_MSG(
			    sensImg_fnames.empty(),
			    "Logic error: Sensitivity image generation was requested while "
			    "pre-existing sensitivity images were provided");
		}


		auto scanner = std::make_unique<Scanner>(scanner_fname);
		auto projectorType = IO::getProjector(projector_name);
		std::unique_ptr<OSEM> osem =
		    Util::createOSEM(*scanner, IO::requiresGPU(projectorType));

		osem->num_MLEM_iterations = numIterations;
		osem->num_OSEM_subsets = numSubsets;
		osem->hardThreshold = hardThreshold;
		osem->projectorType = projectorType;
		osem->numRays = numRays;
		Globals::set_num_threads(numThreads);

		// To make sure the sensitivity image gets generated accordingly
		const bool useListMode =
		    !input_format.empty() && IO::isFormatListMode(input_format);
		osem->setListModeEnabled(useListMode);

		// Total attenuation image
		std::unique_ptr<ImageOwned> attImg = nullptr;
		std::unique_ptr<ProjectionData> acfHisProjData = nullptr;
		if (!acf_fname.empty())
		{
			std::cout << "Reading ACF histogram..." << std::endl;
			ASSERT_MSG(!acf_format.empty(),
			           "Unspecified format for ACF histogram");
			ASSERT_MSG(!IO::isFormatListMode(acf_format),
			           "ACF has to be in a histogram format");

			acfHisProjData = IO::openProjectionData(
			    acf_fname, acf_format, *scanner, pluginOptionsResults);

			const auto* acfHis =
			    dynamic_cast<const Histogram*>(acfHisProjData.get());
			ASSERT(acfHis != nullptr);

			osem->setACFHistogram(acfHis);
		}
		else if (!attImg_fname.empty())
		{
			attImg = std::make_unique<ImageOwned>(attImg_fname);
			osem->setAttenuationImage(attImg.get());
		}

		// Hardware attenuation image
		std::unique_ptr<ImageOwned> hardwareAttImg = nullptr;
		std::unique_ptr<ProjectionData> hardwareAcfHisProjData = nullptr;
		if (!hardwareAcf_fname.empty())
		{
			std::cout << "Reading hardware ACF histogram..." << std::endl;
			ASSERT_MSG(!hardwareAcf_format.empty(),
			           "No format specified for hardware ACF histogram");
			ASSERT_MSG(!IO::isFormatListMode(hardwareAcf_format),
			           "Hardware ACF has to be in a histogram format");

			hardwareAcfHisProjData =
			    IO::openProjectionData(hardwareAcf_fname, hardwareAcf_format,
			                           *scanner, pluginOptionsResults);

			const auto* hardwareAcfHis =
			    dynamic_cast<const Histogram*>(hardwareAcfHisProjData.get());
			ASSERT(hardwareAcfHis != nullptr);

			osem->setACFHistogram(hardwareAcfHis);
		}
		else if (!hardwareAttImg_fname.empty())
		{
			hardwareAttImg = std::make_unique<ImageOwned>(hardwareAttImg_fname);
			osem->setHardwareAttenuationImage(hardwareAttImg.get());
		}

		// Image-space PSF
		std::unique_ptr<OperatorPsf> imageSpacePsf;
		if (!imageSpacePsf_fname.empty())
		{
			osem->addImagePSF(imageSpacePsf_fname);
		}

		// Projection-space PSF
		if (!projSpacePsf_fname.empty())
		{
			osem->addProjPSF(projSpacePsf_fname);
		}

		// Sensitivity image(s)
		std::unique_ptr<ProjectionData> sensitivityProjData = nullptr;
		if (!sensitivityData_fname.empty())
		{
			std::cout << "Reading sensitivity histogram..." << std::endl;
			ASSERT_MSG(!sensitivityData_format.empty(),
			           "No format specified for sensitivity histogram");
			ASSERT_MSG(!IO::isFormatListMode(sensitivityData_format),
			           "Sensitivity data has to be in a histogram format");

			sensitivityProjData = IO::openProjectionData(
			    sensitivityData_fname, sensitivityData_format, *scanner,
			    pluginOptionsResults);

			const auto* sensitivityHis =
			    dynamic_cast<const Histogram*>(sensitivityProjData.get());
			ASSERT(sensitivityHis != nullptr);

			osem->setSensitivityHistogram(sensitivityHis);
			osem->setInvertSensitivity(invertSensitivity);
		}
		osem->setGlobalScalingFactor(globalScalingFactor);

		std::vector<std::unique_ptr<Image>> sensImages;
		bool sensImageAlreadyMoved = false;
		if (sensImg_fnames.empty())
		{
			ASSERT_MSG(!imgParams_fname.empty(),
			           "Image parameters file unspecified");
			ImageParams imgParams{imgParams_fname};
			osem->setImageParams(imgParams);

			osem->generateSensitivityImages(sensImages, out_sensImg_fname);
		}
		else if (osem->getExpectedSensImagesAmount() ==
		         static_cast<int>(sensImg_fnames.size()))
		{
			std::cout << "Reading sensitivity images..." << std::endl;
			for (auto& sensImg_fname : sensImg_fnames)
			{
				sensImages.push_back(
				    std::make_unique<ImageOwned>(sensImg_fname));
			}
			sensImageAlreadyMoved = !mustMoveSens;
		}
		else
		{
			std::cerr << "The number of sensitivity images given is "
			          << sensImg_fnames.size() << std::endl;
			std::cerr << "The expected number of sensitivity images is "
			          << osem->getExpectedSensImagesAmount() << std::endl;
			throw std::invalid_argument(
			    "The number of sensitivity images given "
			    "doesn't match the number of "
			    "subsets specified. Note: For ListMode formats, exactly one "
			    "sensitivity image is required.");
		}

		// No need to read data input if in sensOnly mode
		if (sensOnly && input_fname.empty())
		{
			std::cout << "Done." << std::endl;
			return 0;
		}

		// Projection Data Input file
		std::cout << "Reading input data..." << std::endl;
		std::unique_ptr<ProjectionData> dataInput;
		ASSERT_MSG(!input_format.empty(), "No format specified for Data input");
		dataInput = IO::openProjectionData(input_fname, input_format, *scanner,
		                                   pluginOptionsResults);
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

			if (!out_sensImg_fname.empty())
			{
				// Overwrite sensitivity image
				std::cout << "Saving sensitivity image..." << std::endl;
				movedSensImage->writeToFile(out_sensImg_fname);
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

		if (sensOnly)
		{
			std::cout << "Done." << std::endl;
			return 0;
		}

		if (tofWidth_ps > 0.f)
		{
			osem->addTOF(tofWidth_ps, tofNumStd);
		}

		// Additive histograms
		std::unique_ptr<ProjectionData> randomsProjData = nullptr;
		if (!randoms_fname.empty())
		{
			std::cout << "Reading randoms histogram..." << std::endl;
			ASSERT_MSG(!randoms_format.empty(),
			           "No format specified for randoms histogram");
			ASSERT_MSG(!IO::isFormatListMode(randoms_format),
			           "Randoms must be specified in histogram format");

			randomsProjData = IO::openProjectionData(
			    randoms_fname, randoms_format, *scanner, pluginOptionsResults);
			const auto* randomsHis =
			    dynamic_cast<const Histogram*>(randomsProjData.get());
			ASSERT_MSG(randomsHis != nullptr,
			           "The randoms histogram provided does not inherit from "
			           "Histogram.");
			osem->setRandomsHistogram(randomsHis);
		}
		std::unique_ptr<ProjectionData> scatterProjData = nullptr;
		if (!scatter_fname.empty())
		{
			std::cout << "Reading scatter histogram..." << std::endl;
			ASSERT_MSG(!scatter_format.empty(),
			           "No format specified for scatter histogram");
			ASSERT_MSG(!IO::isFormatListMode(scatter_format),
			           "Scatter must be specified in histogram format");

			scatterProjData = IO::openProjectionData(
			    scatter_fname, scatter_format, *scanner, pluginOptionsResults);
			const auto* scatterHis =
			    dynamic_cast<const Histogram*>(scatterProjData.get());
			ASSERT_MSG(scatterHis != nullptr,
			           "The scatter histogram provided does not inherit from "
			           "Histogram.");
			osem->setScatterHistogram(scatterHis);
		}

		std::unique_ptr<ImageOwned> invivoAttImg = nullptr;
		if (!invivoAttImg_fname.empty())
		{
			ASSERT_MSG_WARNING(dataInput->hasMotion(),
			                   "An in-vivo attenuation image was provided but "
			                   "the data input has no motion");
			invivoAttImg = std::make_unique<ImageOwned>(invivoAttImg_fname);
			osem->setInVivoAttenuationImage(invivoAttImg.get());
		}
		std::unique_ptr<ProjectionData> inVivoAcfProjData = nullptr;
		if (!invivoAcf_fname.empty())
		{
			std::cout << "Reading in-vivo ACF histogram..." << std::endl;
			ASSERT_MSG(!invivoAcf_format.empty(),
			           "No format specified for ACF histogram");
			ASSERT_MSG(!IO::isFormatListMode(invivoAcf_format),
			           "In-vivo ACF must be specified in histogram format");

			inVivoAcfProjData =
			    IO::openProjectionData(invivoAcf_fname, invivoAcf_format,
			                           *scanner, pluginOptionsResults);
			const auto* inVivoAcfHis =
			    dynamic_cast<const Histogram*>(inVivoAcfProjData.get());
			ASSERT_MSG(
			    inVivoAcfHis != nullptr,
			    "The in-vivo ACF histogram provided does not inherit from "
			    "Histogram.");
			osem->setInVivoACFHistogram(inVivoAcfHis);
		}

		// Save steps
		ASSERT_MSG(saveIterStep >= 0, "save_iter_step must be positive.");
		Util::RangeList ranges;
		if (saveIterStep > 0)
		{
			if (saveIterStep == 1)
			{
				ranges.insertSorted(0, numIterations - 1);
			}
			else
			{
				for (int it = 0; it < numIterations; it += saveIterStep)
				{
					ranges.insertSorted(it, it);
				}
			}
		}
		else if (!saveIterRanges.empty())
		{
			ranges.readFromString(saveIterRanges);
		}
		if (!ranges.empty())
		{
			osem->setSaveIterRanges(ranges, out_fname);
		}

		// Initial image estimate
		std::unique_ptr<ImageOwned> initialEstimate = nullptr;
		if (!initialEstimate_fname.empty())
		{
			initialEstimate =
			    std::make_unique<ImageOwned>(initialEstimate_fname);
			osem->initialEstimate = initialEstimate.get();
		}

		std::cout << "Launching reconstruction..." << std::endl;
		osem->reconstruct(out_fname);

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
