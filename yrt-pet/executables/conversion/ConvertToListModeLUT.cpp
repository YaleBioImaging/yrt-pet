/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "datastruct/IO.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "datastruct/projection/SparseHistogram.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ReconstructionUtils.hpp"

#include "omp.h"
#include <cxxopts.hpp>
#include <iostream>

int main(int argc, char** argv)
{
	try
	{
		IO::ArgumentRegistry registry{};

		std::string coreGroup = "0. Core";
		std::string inputGroup = "1. Input";
		std::string outputGroup = "2. Output";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          IO::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, IO::TypeOfArgument::INT, -1,
		                          coreGroup);
		registry.registerArgument("input", "Input listmdde file", true,
		                          IO::TypeOfArgument::STRING, "", inputGroup,
		                          "i");
		registry.registerArgument(
		    "format",
		    "Input listmode file format. Possible values: " +
		        IO::possibleFormats(Plugin::InputFormatsChoice::ONLYLISTMODES),
		    true, IO::TypeOfArgument::STRING, "", inputGroup, "f");

		registry.registerArgument("out", "Output listmode filename", true,
		                          IO::TypeOfArgument::STRING, "", outputGroup,
		                          "o");

		PluginOptionsHelper::addOptionsFromPlugins(
			registry, Plugin::InputFormatsChoice::ONLYLISTMODES);

		// Load configuration
		IO::ArgumentReader config{
		    registry, "Convert a list-mode input (of any format, including "
		              "plugin formats) to a ListModeLUT format"};

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

		auto scanner_fname = config.getValue<std::string>("scanner");
		auto input_fname = config.getValue<std::string>("input");
		auto input_format = config.getValue<std::string>("format");
		auto out_fname = config.getValue<std::string>("out");
		int numThreads = config.getValue<int>("num_threads");

		Globals::set_num_threads(numThreads);
		std::cout << "Initializing scanner..." << std::endl;
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		std::cout << "Reading input data..." << std::endl;
		std::unique_ptr<ProjectionData> dataInput = IO::openProjectionData(
		    input_fname, input_format, *scanner, config.getAllArguments());

		std::cout << "Generating output ListModeLUT..." << std::endl;
		auto lmOut =
		    std::make_unique<ListModeLUTOwned>(*scanner, dataInput->hasTOF());
		const size_t numEvents = dataInput->count();
		lmOut->allocate(numEvents);

		ListModeLUTOwned* lmOut_ptr = lmOut.get();
		const ProjectionData* dataInput_ptr = dataInput.get();
		const bool hasTOF = dataInput->hasTOF();
#pragma omp parallel for default(none), \
    firstprivate(lmOut_ptr, dataInput_ptr, numEvents, hasTOF)
		for (bin_t evId = 0; evId < numEvents; evId++)
		{
			lmOut_ptr->setTimestampOfEvent(evId,
			                               dataInput_ptr->getTimestamp(evId));
			auto [d1, d2] = dataInput_ptr->getDetectorPair(evId);
			lmOut_ptr->setDetectorIdsOfEvent(evId, d1, d2);
			if (hasTOF)
			{
				lmOut_ptr->setTOFValueOfEvent(evId,
				                              dataInput_ptr->getTOFValue(evId));
			}
		}

		std::cout << "Writing file..." << std::endl;
		lmOut->writeToFile(out_fname);

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
