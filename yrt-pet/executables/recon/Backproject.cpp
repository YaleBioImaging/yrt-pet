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
		IO::ArgumentRegistry registry{};

		const std::string coreGroup = "0. Core";
		const std::string inputGroup = "1. Input";
		const std::string projectorGroup = "2. Projector";
		const std::string imageGroup = "3. Image";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          IO::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument("input", "Input file", true,
		                          IO::TypeOfArgument::STRING, "", inputGroup,
		                          "i");
		registry.registerArgument(
		    "format",
		    "Input file format. Possible values: " + IO::possibleFormats(),
		    true, IO::TypeOfArgument::STRING, "", inputGroup, "f");

#if BUILD_CUDA
		registry.registerArgument("gpu", "Use GPU acceleration", false,
		                          IO::TypeOfArgument::BOOL, false, coreGroup);
#endif
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, IO::TypeOfArgument::INT, -1,
		                          coreGroup);

		registry.registerArgument("out", "Output image filename", true,
		                          IO::TypeOfArgument::STRING, "", imageGroup,
		                          "o");
		registry.registerArgument("params", "Image parameters file", true,
		                          IO::TypeOfArgument::STRING, "", imageGroup,
		                          "p");

		registry.registerArgument(
		    "projector",
		    "Projector to use, choices: Siddon (S), Distance-Driven (D). The "
		    "default projector is Siddon",
		    false, IO::TypeOfArgument::STRING, "S", projectorGroup);
		registry.registerArgument(
		    "psf",
		    "Image-space PSF kernel file (Applied after the backprojection)",
		    false, IO::TypeOfArgument::STRING, "", imageGroup);
		registry.registerArgument(
		    "proj_psf", "Projection-space PSF kernel file", false,
		    IO::TypeOfArgument::STRING, "", projectorGroup);
		registry.registerArgument(
		    "num_rays", "Number of rays to use (for Siddon projector only)",
		    false, IO::TypeOfArgument::INT, 1, projectorGroup);
		registry.registerArgument("tof_width_ps", "TOF Width in Picoseconds",
		                          false, IO::TypeOfArgument::FLOAT, 0.0f,
		                          projectorGroup);
		registry.registerArgument("tof_n_std",
		                          "Number of standard deviations to consider "
		                          "for TOF's Gaussian curve",
		                          false, IO::TypeOfArgument::INT, 0,
		                          projectorGroup);
		registry.registerArgument("num_subsets",
		                          "Number of OSEM subsets (Default: 1)", false,
		                          IO::TypeOfArgument::INT, 1, inputGroup);
		registry.registerArgument("subset_id",
		                          "Subset to backproject (Default: 0)", false,
		                          IO::TypeOfArgument::INT, 0, inputGroup);

		// Add plugin options
		PluginOptionsHelper::addOptionsFromPlugins(
		    registry, Plugin::InputFormatsChoice::ALL);

		// Load configuration
		IO::ArgumentReader config{registry,
		                          "Backproject projection data into an image"};

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

		const auto scanner =
		    std::make_unique<Scanner>(config.getValue<std::string>("scanner"));
		Globals::set_num_threads(config.getValue<int>("num_threads"));

		// Output image
		std::cout << "Preparing output image..." << std::endl;
		ImageParams outputImageParams{config.getValue<std::string>("params")};
		const auto outputImage =
		    std::make_unique<ImageOwned>(outputImageParams);
		outputImage->allocate();

		// Input data
		std::cout << "Reading input data..." << std::endl;
		const auto dataInput =
		    IO::openProjectionData(config.getValue<std::string>("input"),
		                           config.getValue<std::string>("format"),
		                           *scanner, config.getAllArguments());

		// Setup forward projection
		const auto binIter =
		    dataInput->getBinIter(config.getValue<int>("nunm_subsets"),
		                          config.getValue<int>("subset_id"));
		const OperatorProjectorParams projParams(
		    binIter.get(), *scanner, config.getValue<int>("tof_width_ps"),
		    config.getValue<int>("tof_n_std"),
		    config.getValue<std::string>("proj_psf"),
		    config.getValue<int>("num_rays"));

		const auto projectorType =
		    IO::getProjector(config.getValue<std::string>("projector"));

		Util::backProject(*outputImage, *dataInput, projParams, projectorType,
		                  config.getValue<bool>("gpu"));

		// Image-space PSF
		const std::string imagePsf_fname = config.getValue<std::string>("psf");
		if (!imagePsf_fname.empty())
		{
			const auto imagePsf = std::make_unique<OperatorPsf>(imagePsf_fname);
			std::cout << "Applying Image-space PSF..." << std::endl;
			imagePsf->applyAH(outputImage.get(), outputImage.get());
		}

		std::cout << "Writing image to file..." << std::endl;
		outputImage->writeToFile(config.getValue<std::string>("out"));
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
