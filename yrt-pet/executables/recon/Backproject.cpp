/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "datastruct/IO.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "operators/OperatorProjector.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ReconstructionUtils.hpp"
#include "utils/Tools.hpp"

#include <cxxopts.hpp>
#include <iostream>

int main(int argc, char** argv)
{
	try
	{
		std::string scanner_fname;
		std::string outputImageParams_fname;
		std::string outputImage_fname;
		std::string input_fname;
		std::string input_format;
		std::string imagePsf_fname;
		std::string projPsf_fname;
		std::string projector_name = "S";
		int numThreads = -1;
		int numSubsets = 1;
		int subsetId = 0;
		int numRays = 1;
		bool useGPU = false;
		float tofWidth_ps = 0.0;
		int tofNumStd = 0;

		Plugin::OptionsResult pluginOptionsResults;  // For plugins' options

		// Parse command line arguments
		cxxopts::Options options(argv[0], "Backproject "
		                                  "projection data into an image.");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
		("s,scanner", "Scanner parameters file", cxxopts::value<std::string>(scanner_fname))
		("i,input", "Input data file", cxxopts::value<std::string>(input_fname))
		("f,format", "Input data format", cxxopts::value<std::string>(input_format))
		("o,out", "Output image filename", cxxopts::value<std::string>(outputImage_fname))
		("p,params", "Output image parameters filename", cxxopts::value<std::string>(outputImageParams_fname))
		("gpu", "Use GPU acceleration", cxxopts::value<bool>(useGPU))
		("psf", "Image-space PSF kernel file", cxxopts::value<std::string>(imagePsf_fname))
		("proj_psf", "Projection-space PSF kernel file", cxxopts::value<std::string>(projPsf_fname))
		("projector", "Projector to use, choices: Siddon (S), Distance-Driven (D)"
			". The default projector is Siddon", cxxopts::value<std::string>(projector_name))
		("tof_width_ps", "TOF Width in Picoseconds", cxxopts::value<float>(tofWidth_ps))
		("tof_n_std", "Number of standard deviations to consider for TOF's Gaussian curve", cxxopts::value<int>(tofNumStd))
		("num_rays", "Number of rays to use in the Siddon projector", cxxopts::value<int>(numRays))
		("num_threads", "Number of threads to use", cxxopts::value<int>(numThreads))
		("num_subsets", "Number of subsets to use (Default: 1)", cxxopts::value<int>(numSubsets))
		("subset_id", "Subset to backproject (Default: 0)", cxxopts::value<int>(subsetId))
		("h,help", "Print help");
		/* clang-format on */

		// Add plugin options
		PluginOptionsHelper::fillOptionsFromPlugins(options);

		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		std::vector<std::string> required_params = {"scanner", "input",
		                                            "format", "out"};
		bool missing_args = false;
		for (auto& p : required_params)
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
		Globals::set_num_threads(numThreads);

		// Output image
		std::cout << "Preparing output image..." << std::endl;
		ImageParams outputImageParams{outputImageParams_fname};
		auto outputImage = std::make_unique<ImageOwned>(outputImageParams);
		outputImage->allocate();

		// Input data
		std::cout << "Reading input data..." << std::endl;
		auto dataInput = IO::openProjectionData(input_fname, input_format,
		                                        *scanner, pluginOptionsResults);

		// Setup forward projection
		auto binIter = dataInput->getBinIter(numSubsets, subsetId);
		OperatorProjectorParams projParams(binIter.get(), *scanner, tofWidth_ps,
		                                   tofNumStd, projPsf_fname, numRays);

		auto projectorType = IO::getProjector(projector_name);

		Util::backProject(*outputImage, *dataInput, projParams, projectorType,
		                  useGPU);

		// Image-space PSF
		if (!imagePsf_fname.empty())
		{
			auto imagePsf = std::make_unique<OperatorPsf>(imagePsf_fname);
			std::cout << "Applying Image-space PSF..." << std::endl;
			imagePsf->applyAH(outputImage.get(), outputImage.get());
		}

		std::cout << "Writing image to file..." << std::endl;
		outputImage->writeToFile(outputImage_fname);
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
