/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "datastruct/IO.hpp"
#include "datastruct/projection/UniformHistogram.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "motion/ImageWarperMatrix.hpp"
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
		std::string attImg_fname;
		std::string acf_fname;
		std::string acf_format;
		std::string invivoAttImg_fname;
		std::string invivoAcf_fname;
		std::string invivoAcf_format;
		std::string imageSpacePsf_fname;
		std::string projSpacePsf_fname;
		std::string randoms_fname;
		std::string randoms_format;
		std::string scatter_fname;
		std::string scatter_format;
		std::string projector_name = "S";
		// TODO NOW: Add backprojection executable
		// TODO NOW: Restore scatter estimation executable
		std::string sensitivityData_fname;
		std::string sensitivityData_format;
		std::string warpParamFile;  // For Warper
		std::string out_fname;
		std::string out_sensImg_fname;
		int numIterations = 10;
		int numSubsets = 1;
		int numThreads = -1;
		int numRays = 1;
		float hardThreshold = 1.0f;
		float tofWidth_ps = 0.0f;
		int tofNumStd = 0;
		int saveIterStep = 0;
		std::string saveIterRanges;
		bool sensOnly = false;

		Plugin::OptionsResult pluginOptionsResults;  // For plugins' options

		// Parse command line arguments
		cxxopts::Options options(argv[0], "Reconstruction executable");
		options.positional_help("[optional args]").show_positional_help();

		auto coreGroup = options.add_options("0. Core");
		coreGroup("s,scanner", "Scanner parameters file name",
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
		sensGroup("sensdata", "Sensitivity histogram file",
		          cxxopts::value<std::string>(sensitivityData_fname));
		sensGroup(
		    "sensdata_format",
		    "Sensitivity histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    cxxopts::value<std::string>(sensitivityData_format));
		sensGroup("att",
		          "Attenuation image filename (In case of motion correction, "
		          "Hardware attenuation image filename)",
		          cxxopts::value<std::string>(attImg_fname));
		sensGroup(
		    "acf",
		    "Attenuation correction factors histogram filename (In case "
		    "of motion correction, Hardware attenuation correction factors)",
		    cxxopts::value<std::string>(acf_fname));
		sensGroup(
		    "acf_format",
		    "Attenuation correction factors histogram format. Possible "
		    "values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    cxxopts::value<std::string>(acf_format));

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
		reconGroup("att_invivo",
		           "In case of motion correction only, in-vivo attenuation "
		           "image filename",
		           cxxopts::value<std::string>(invivoAttImg_fname));
		reconGroup("acf_invivo",
		           "In case of motion correction only, in-vivo attenuation "
		           "correction factors histogram filename",
		           cxxopts::value<std::string>(invivoAcf_fname));
		reconGroup(
		    "acf_invivo_format",
		    "In case of motion correction only, in-vivo attenuation "
		    "correction factors histogram format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    cxxopts::value<std::string>(invivoAcf_fname));

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

		auto otherGroup = options.add_options("Other");
		otherGroup("w,warper",
		           "Path to the warp parameters file (Specify this to use the "
		           "MLEM with image warper algorithm)",
		           cxxopts::value<std::string>(warpParamFile));

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

		if (sensOnly)
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

		// Attenuation image
		std::unique_ptr<ImageOwned> attImg = nullptr;
		if (!attImg_fname.empty())
		{
			attImg = std::make_unique<ImageOwned>(attImg_fname);
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
			ASSERT_MSG(!IO::isFormatListMode(sensitivityData_format),
			           "Sensitivity data has to be in a histogram format");

			sensitivityProjData = IO::openProjectionData(
			    sensitivityData_fname, sensitivityData_format, *scanner,
			    pluginOptionsResults);

			const auto* sensitivityData =
			    dynamic_cast<const Histogram*>(sensitivityProjData.get());
			ASSERT(sensitivityData != nullptr);

			osem->setSensitivityHistogram(sensitivityData);
		}

		std::vector<std::unique_ptr<Image>> sensImages;
		bool sensImageAlreadyMoved = false;
		if (sensImg_fnames.empty())
		{
			ASSERT_MSG(!imgParams_fname.empty(),
			           "Image parameters file unspecified");
			ImageParams imgParams{imgParams_fname};
			osem->setImageParams(imgParams);

			osem->attenuationImageForBackprojection = attImg.get();

			osem->generateSensitivityImages(sensImages, out_sensImg_fname);

			// Do not use this attenuation image for the reconstruction
			osem->attenuationImageForBackprojection = nullptr;
		}
		else if (osem->validateSensImagesAmount(
		             static_cast<int>(sensImg_fnames.size())))
		{
			std::cout << "Reading sensitivity images..." << std::endl;
			for (auto& sensImg_fname : sensImg_fnames)
			{
				sensImages.push_back(
				    std::make_unique<ImageOwned>(sensImg_fname));
			}
			sensImageAlreadyMoved = true;
			std::cout << "Done reading sensitivity images." << std::endl;
		}
		else
		{
			std::cerr << "The number of sensitivity images given is "
			          << sensImg_fnames.size() << std::endl;
			std::cerr << "The expected number of sensitivity images is "
			          << (useListMode ? 1 : numSubsets) << std::endl;
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
		dataInput = IO::openProjectionData(input_fname, input_format, *scanner,
		                                   pluginOptionsResults);
		std::cout << "Done reading input data." << std::endl;
		osem->setDataInput(dataInput.get());

		std::unique_ptr<ImageOwned> movedSensImage = nullptr;
		if (dataInput->hasMotion() && !sensImageAlreadyMoved)
		{
			ASSERT_MSG_WARNING(
			    !invivoAttImg_fname.empty() || sensOnly,
			    "The data input provided has motion information, but no "
			    "in-vivo attenuation was provided.");
			ASSERT(sensImages.size() == 1);
			const Image* unmovedSensImage = sensImages[0].get();
			ASSERT(unmovedSensImage != nullptr);

			movedSensImage =
			    std::make_unique<ImageOwned>(unmovedSensImage->getParams());
			movedSensImage->allocate();

			std::cout << "Moving sensitivity image..." << std::endl;
			int64_t numFrames = dataInput->getNumFrames();
			Util::ProgressDisplay progress{numFrames};
			const auto scanDuration =
			    static_cast<float>(dataInput->getScanDuration());
			for (frame_t frame = 0; frame < numFrames; frame++)
			{
				progress.progress(frame);
				transform_t transform = dataInput->getTransformOfFrame(frame);
				const float weight =
				    dataInput->getDurationOfFrame(frame) / scanDuration;
				unmovedSensImage->transformImage(transform, *movedSensImage,
				                                 weight);
			}

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
		std::cout << "Reading randoms histogram..." << std::endl;
		std::unique_ptr<ProjectionData> randomsProjData = nullptr;
		if (!randoms_fname.empty())
		{
			randomsProjData = IO::openProjectionData(randoms_fname, randoms_format,
			                                *scanner, pluginOptionsResults);
			const auto* randomsHis = dynamic_cast<const Histogram*>(randomsProjData.get());
			ASSERT_MSG(randomsHis != nullptr,
			           "The randoms histogram provided does not inherit from "
			           "Histogram.");
			osem->setRandomsHistogram(randomsHis);
		}
		std::cout << "Reading scatter histogram..." << std::endl;
		std::unique_ptr<ProjectionData> scatterProjData = nullptr;
		if (!scatter_fname.empty())
		{
			scatterProjData = IO::openProjectionData(scatter_fname, scatter_format,
											*scanner, pluginOptionsResults);
			const auto* scatterHis = dynamic_cast<const Histogram*>(scatterProjData.get());
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
		}
		osem->attenuationImageForForwardProjection = invivoAttImg.get();

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

		// Image Warper
		std::unique_ptr<ImageWarperTemplate> warper = nullptr;
		if (!warpParamFile.empty())
		{
			warper = std::make_unique<ImageWarperMatrix>();
			warper->setImageHyperParam(osem->getImageParams());
			warper->setFramesParamFromFile(warpParamFile);
			osem->warper = warper.get();
		}

		if (warper == nullptr)
		{
			std::cout << "Launching reconstruction..." << std::endl;
			osem->reconstruct(out_fname);
		}
		else
		{
			std::cout << "Launching reconstruction with image warper..."
			          << std::endl;
			osem->reconstructWithWarperMotion(out_fname);
		}

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
