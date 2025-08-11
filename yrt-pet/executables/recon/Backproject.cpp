/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#include <cxxopts.hpp>
#include <iostream>

using namespace yrt;

int main(int argc, char** argv)
{
	try
	{
		io::ArgumentRegistry registry{};

		const std::string coreGroup = "0. Core";
		const std::string inputGroup = "1. Input";
		const std::string projectorGroup = "2. Projector";
		const std::string outputGroup = "3. Output";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument("input", "Input file", true,
		                          io::TypeOfArgument::STRING, "", inputGroup,
		                          "i");
		registry.registerArgument(
		    "format",
		    "Input file format. Possible values: " + io::possibleFormats(),
		    true, io::TypeOfArgument::STRING, "", inputGroup, "f");
		registry.registerArgument(
		    "lor_motion", "Motion CSV file for motion correction", false,
		    io::TypeOfArgument::STRING, "", inputGroup, "m");

#if BUILD_CUDA
		registry.registerArgument("gpu", "Use GPU acceleration", false,
		                          io::TypeOfArgument::BOOL, false, coreGroup);
#endif
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, io::TypeOfArgument::INT, -1,
		                          coreGroup);

		registry.registerArgument("out", "Output image filename", true,
		                          io::TypeOfArgument::STRING, "", outputGroup,
		                          "o");
		registry.registerArgument("params", "Image parameters file", true,
		                          io::TypeOfArgument::STRING, "", outputGroup,
		                          "p");

		registry.registerArgument(
		    "projector",
		    "Projector to use, choices: Siddon (S), Distance-Driven (D). The "
		    "default projector is Siddon",
		    false, io::TypeOfArgument::STRING, "S", projectorGroup);
		registry.registerArgument(
		    "psf",
		    "Image-space PSF kernel file (Applied after the backprojection)",
		    false, io::TypeOfArgument::STRING, "", outputGroup);
		registry.registerArgument("varpsf",
		                          "Image-space Variant PSF look-up table file "
		                          "(Applied after the backprojection)",
		                          false, io::TypeOfArgument::STRING, "",
		                          outputGroup);
		registry.registerArgument(
		    "proj_psf",
		    "Projection-space PSF kernel file (for DD projector only)", false,
		    io::TypeOfArgument::STRING, "", projectorGroup);
		registry.registerArgument(
		    "num_rays", "Number of rays to use (for Siddon projector only)",
		    false, io::TypeOfArgument::INT, 1, projectorGroup);
		registry.registerArgument("tof_width_ps", "TOF Width in Picoseconds",
		                          false, io::TypeOfArgument::FLOAT, 0.0f,
		                          projectorGroup);
		registry.registerArgument("tof_n_std",
		                          "Number of standard deviations to consider "
		                          "for TOF's Gaussian curve",
		                          false, io::TypeOfArgument::INT, 0,
		                          projectorGroup);
		registry.registerArgument("num_subsets",
		                          "Number of OSEM subsets (Default: 1)", false,
		                          io::TypeOfArgument::INT, 1, inputGroup);
		registry.registerArgument("subset_id",
		                          "Subset to backproject (Default: 0)", false,
		                          io::TypeOfArgument::INT, 0, inputGroup);

		// Add plugin options
		plugin::addOptionsFromPlugins(registry,
		                              plugin::InputFormatsChoice::ALL);

		// Load configuration
		io::ArgumentReader config{registry,
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
		globals::setNumThreads(config.getValue<int>("num_threads"));

		// Output image
		std::cout << "Preparing output image..." << std::endl;
		ImageParams imgParams{config.getValue<std::string>("params")};
		const auto outputImage = std::make_unique<ImageOwned>(imgParams);
		outputImage->allocate();

		// Input data
		std::cout << "Reading input data..." << std::endl;
		const auto format = config.getValue<std::string>("format");
		const bool useListMode = io::isFormatListMode(format);
		const auto dataInput =
		    io::openProjectionData(config.getValue<std::string>("input"),
		                           format, *scanner, config.getAllArguments());

		const auto lorMotion_fname = config.getValue<std::string>("lor_motion");

		if (!lorMotion_fname.empty())
		{
			auto lorMotion = std::make_shared<LORMotion>(lorMotion_fname);

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

		// Setup forward projection
		const auto binIter =
		    dataInput->getBinIter(config.getValue<int>("num_subsets"),
		                          config.getValue<int>("subset_id"));
		auto projectorUpdaterType = OperatorProjectorParams::ProjectorUpdaterType::DEFAULT3D;
		if (config.hasValue("projector_updater_type")) {
			const auto s = config.getValue<std::string>("projector_updater_type");
			// map to enum
			if      (s == "DEFAULT3D") projectorUpdaterType = OperatorProjectorParams::ProjectorUpdaterType::DEFAULT3D;
			else if (s == "DEFAULT4D") projectorUpdaterType = OperatorProjectorParams::ProjectorUpdaterType::DEFAULT4D;
			else if (s == "LR")        projectorUpdaterType = OperatorProjectorParams::ProjectorUpdaterType::LR;
			else throw std::invalid_argument("Unknown projector_updater_type: " + s);
		}
		const OperatorProjectorParams projParams(
		    binIter.get(), *scanner, projectorUpdaterType,
		    config.getValue<float>("tof_width_ps"),
		    config.getValue<int>("tof_n_std"),
		    config.getValue<std::string>("proj_psf"),
		    config.getValue<int>("num_rays"));

		const auto projectorType =
		    io::getProjector(config.getValue<std::string>("projector"));

		util::backProject(*outputImage, *dataInput, projParams, projectorType,
		                  config.getValue<bool>("gpu"));

		// Image-space PSF
		auto imagePsf_fname = config.getValue<std::string>("psf");
		auto varPsf_fname = config.getValue<std::string>("varpsf");
		if (!imagePsf_fname.empty())
		{
			ASSERT_MSG(varPsf_fname.empty(),
			           "Got two different image PSF inputs");
			const auto imagePsf = std::make_unique<OperatorPsf>(imagePsf_fname);
			std::cout << "Applying uniform Image-space PSF..." << std::endl;
			imagePsf->applyAH(outputImage.get(), outputImage.get());
		}
		else if (!varPsf_fname.empty())
		{
			const auto imagePsf =
			    std::make_unique<OperatorVarPsf>(varPsf_fname, imgParams);
			std::cout << "Applying variant Image-space PSF..." << std::endl;
			auto tempBuffer = std::make_unique<ImageOwned>(imgParams);
			tempBuffer->allocate();
			tempBuffer->copyFromImage(outputImage.get());
			imagePsf->applyAH(tempBuffer.get(), outputImage.get());
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
		util::printExceptionMessage(e);
		return -1;
	}
}
