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
		std::string invivoAttImg_fname;
		std::string imageSpacePsf_fname;
		std::string projSpacePsf_fname;
		std::string addHis_fname;
		std::string addHis_format = "H";
		std::string projector_name = "S";
		std::string sensData_fname;
		std::string sensData_format;
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
		int saveSteps = 0;
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
		sensGroup("sensdata", "Sensitivity data input file",
		          cxxopts::value<std::string>(sensData_fname));
		sensGroup("sensdata_format",
		          "Sensitivity data input file format. Possible values: " +
		              IO::possibleFormats(),
		          cxxopts::value<std::string>(sensData_format));
		sensGroup("att",
		          "Attenuation image filename (In case of motion correction, "
		          "Hardware attenuation image filename)",
		          cxxopts::value<std::string>(attImg_fname));

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
		reconGroup("add_his", "Additive corrections histogram",
		           cxxopts::value<std::string>(addHis_fname));
		reconGroup(
		    "add_his_format",
		    "Format of the additive corrections histogram. Default value: H",
		    cxxopts::value<std::string>(addHis_format));
		reconGroup("psf", "Image-space PSF kernel file",
		           cxxopts::value<std::string>(imageSpacePsf_fname));
		reconGroup("hard_threshold", "Hard Threshold",
		           cxxopts::value<float>(hardThreshold));
		reconGroup("save_steps",
		           "Increment into which to save MLEM iteration images",
		           cxxopts::value<int>(saveSteps));
		reconGroup("att_invivo",
		           "In case of motion correction only, in-vivo attenuation "
		           "image filename",
		           cxxopts::value<std::string>(invivoAttImg_fname));

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
		std::vector<std::unique_ptr<Image>> sensImages;
		if (sensImg_fnames.empty())
		{
			ASSERT_MSG(!imgParams_fname.empty(),
			           "Image parameters file unspecified");
			ImageParams imgParams{imgParams_fname};
			osem->setImageParams(imgParams);

			std::unique_ptr<ProjectionData> sensData = nullptr;
			if (!sensData_fname.empty())
			{
				sensData =
				    IO::openProjectionData(sensData_fname, sensData_format,
				                           *scanner, pluginOptionsResults);
			}

			osem->attenuationImageForBackprojection = attImg.get();
			osem->setSensDataInput(sensData.get());

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
		if (dataInput->hasMotion())
		{
			ASSERT_MSG_WARNING(
			    !invivoAttImg_fname.empty(),
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
			float scanDuration =
			    static_cast<float>(dataInput->getScanDuration());
			for (frame_t frame = 0; frame < numFrames; frame++)
			{
				progress.progress(frame);
				transform_t transform = dataInput->getTransformOfFrame(frame);
				float weight =
				    dataInput->getDurationOfFrame(frame) / scanDuration;
				unmovedSensImage->transformImage(transform, *movedSensImage,
				                                 weight);
			}

			if (!out_sensImg_fname.empty())
			{
				// Overwrite sensitivity image
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

		// Additive histogram
		std::cout << "Reading additive histogram..." << std::endl;
		std::unique_ptr<ProjectionData> addHis;
		if (!addHis_fname.empty())
		{
			addHis = IO::openProjectionData(addHis_fname, addHis_format,
			                                *scanner, pluginOptionsResults);
			osem->addHis = dynamic_cast<const Histogram*>(addHis.get());
			ASSERT_MSG(osem->addHis != nullptr,
			           "The additive histogram provided does not inherit from "
			           "Histogram.");
		}
		std::cout << "Done reading additive histogram." << std::endl;

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
		osem->setSaveSteps(saveSteps, out_fname);

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
