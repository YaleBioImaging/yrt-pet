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
		std::string attImgParams_fname;
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

		/* clang-format off */
		options.add_options()
		("s,scanner", "Scanner parameters file name", cxxopts::value<std::string>(scanner_fname))
		("p,params", "Image parameters file", cxxopts::value<std::string>(imgParams_fname))
		("i,input", "Input file", cxxopts::value<std::string>(input_fname))
		("f,format", "Input file format. Possible values: " + IO::possibleFormats(), cxxopts::value<std::string>(input_format))
		("sens", "Sensitivity image files (separated by a comma)", cxxopts::value<std::vector<std::string>>(sensImg_fnames))
		("att", "Attenuation image filename", cxxopts::value<std::string>(attImg_fname))
		("att_params", "Attenuation image parameters filename", cxxopts::value<std::string>(attImgParams_fname))
		("psf", "Image-space PSF kernel file", cxxopts::value<std::string>(imageSpacePsf_fname))
		("proj_psf", "Projection-space PSF kernel file", cxxopts::value<std::string>(projSpacePsf_fname))
		("add_his", "Histogram with additive corrections (scatter & randoms)", cxxopts::value<std::string>(addHis_fname))
		("add_his_format", "Format of the histogram with additive corrections. Default value: H", cxxopts::value<std::string>(addHis_format))
		("sensdata", "Sensitivity data input file", cxxopts::value<std::string>(sensData_fname))
		("sensdata_format", "Sensitivity data input file format. Possible values: " + IO::possibleFormats(), cxxopts::value<std::string>(sensData_format))
		("w,warper", "Path to the warp parameters file (Specify this to use the MLEM with image warper algorithm)", cxxopts::value<std::string>(warpParamFile))
		("projector", "Projector to use, choices: Siddon (S), Distance-Driven (D)"
		 #if BUILD_CUDA
		 ", or GPU Distance-Driven (DD_GPU)"
		 #endif
		 ". The default projector is Siddon", cxxopts::value<std::string>(projector_name))
		("num_rays", "Number of rays to use in the Siddon projector", cxxopts::value<int>(numRays))
		("tof_width_ps", "TOF Width in Picoseconds", cxxopts::value<float>(tofWidth_ps))
		("tof_n_std", "Number of standard deviations to consider for TOF's Gaussian curve", cxxopts::value<int>(tofNumStd))
		("o,out", "Output image filename", cxxopts::value<std::string>(out_fname))
		("num_iterations", "Number of MLEM Iterations", cxxopts::value<int>(numIterations))
		("num_threads", "Number of threads to use", cxxopts::value<int>(numThreads))
		("num_subsets","Number of OSEM subsets (Default: 1)", cxxopts::value<int>(numSubsets))
		("hard_threshold", "Hard Threshold", cxxopts::value<float>(hardThreshold))
		("save_steps", "Enable saving each MLEM iteration image (step)", cxxopts::value<int>(saveSteps))
		("sens_only", "Only generate the sensitivity image. Do not launch reconstruction", cxxopts::value<bool>(sensOnly))
		("out_sens", "Filename for the generated sensitivity image (if it needed to be computed). Leave blank to not save it", cxxopts::value<std::string>(out_sensImg_fname))
		("h,help", "Print help");
		/* clang-format on */

		// Add plugin options
		PluginOptionsHelper::fillOptionsFromPlugins(options);

		const auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		std::vector<std::string> requiredParams = {"scanner", "params"};
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

		auto scanner = std::make_unique<Scanner>(scanner_fname);
		auto projectorType = IO::getProjector(projector_name);

		std::unique_ptr<OSEM> osem =
		    Util::createOSEM(*scanner, IO::requiresGPU(projectorType));

		osem->num_MLEM_iterations = numIterations;
		osem->num_OSEM_subsets = numSubsets;
		osem->hardThreshold = hardThreshold;
		osem->setImageParams(ImageParams{imgParams_fname});
		osem->projectorType = projectorType;
		osem->numRays = numRays;
		Globals::set_num_threads(numThreads);

		// To make sure the sensitivity image gets generated accordingly
		const bool useListMode =
		    !input_format.empty() && IO::isFormatListMode(input_format);
		osem->setListModeEnabled(useListMode);

		// Attenuation image
		std::unique_ptr<ImageParams> att_img_params = nullptr;
		std::unique_ptr<ImageOwned> att_img = nullptr;
		if (!attImg_fname.empty())
		{
			if (attImgParams_fname.empty())
			{
				std::cerr << "If passing an attenuation image, the "
				             "corresponding image "
				          << "parameter file must also be provided"
				          << std::endl;
				return -1;
			}
			att_img_params = std::make_unique<ImageParams>(attImgParams_fname);
			att_img =
			    std::make_unique<ImageOwned>(*att_img_params, attImg_fname);
		}

		// Image-space PSF
		std::unique_ptr<OperatorPsf> imageSpacePsf;
		if (!imageSpacePsf_fname.empty())
		{
			imageSpacePsf = std::make_unique<OperatorPsf>(
			    osem->getImageParams(), imageSpacePsf_fname);
			osem->addImagePSF(imageSpacePsf.get());
		}

		// Projection-space PSF
		if (!projSpacePsf_fname.empty())
		{
			osem->addProjPSF(projSpacePsf_fname);
		}

		// Sensitivity image(s)
		std::vector<std::shared_ptr<Image>> sensImages;
		if (sensImg_fnames.empty())
		{
			std::unique_ptr<ProjectionData> sensData = nullptr;
			if (!sensData_fname.empty())
			{
				sensData =
				    IO::openProjectionData(sensData_fname, sensData_format,
				                           *scanner, pluginOptionsResults);
			}

			osem->attenuationImageForBackprojection = att_img.get();
			osem->setSensDataInput(sensData.get());

			osem->generateSensitivityImages(sensImages, out_sensImg_fname);

			// Do not use this attenuation image for the reconstruction
			osem->attenuationImageForBackprojection = nullptr;
		}
		else if (osem->validateSensImagesAmount(
		             static_cast<int>(sensImg_fnames.size())))
		{
			for (auto& sensImg_fname : sensImg_fnames)
			{
				sensImages.push_back(std::make_unique<ImageOwned>(
				    osem->getImageParams(), sensImg_fname));
			}
		}
		else
		{
			std::cout << "number of sensImages given: " << sensImg_fnames.size()
			          << std::endl;
			throw std::invalid_argument(
			    "The number of sensitivity images given "
			    "doesn't match the number of "
			    "subsets specified. Note: For ListMode format, only one "
			    "sensitivity image is required.");
		}

		if (sensOnly)
		{
			std::cout << "Done." << std::endl;
			return 0;
		}

		osem->setSensitivityImages(sensImages);

		// Projection Data Input file
		std::unique_ptr<ProjectionData> dataInput;
		dataInput = IO::openProjectionData(input_fname, input_format, *scanner,
		                                   pluginOptionsResults);
		osem->setDataInput(dataInput.get());
		if (tofWidth_ps > 0.f)
		{
			osem->addTOF(tofWidth_ps, tofNumStd);
		}

		// Additive histogram
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
