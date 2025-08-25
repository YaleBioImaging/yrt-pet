/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/datastruct/projection/SparseHistogram.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
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

		std::string coreGroup = "0. Core";
		std::string inputGroup = "1. Input";
		std::string outputGroup = "2. Output";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, io::TypeOfArgument::INT, -1,
		                          coreGroup);
		registry.registerArgument("input", "Input listmode file", true,
		                          io::TypeOfArgument::STRING, "", inputGroup,
		                          "i");
		registry.registerArgument(
		    "format",
		    "Input listmode file format. Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYLISTMODES),
		    true, io::TypeOfArgument::STRING, "", inputGroup, "f");

		registry.registerArgument("out", "Output listmode filename", true,
		                          io::TypeOfArgument::STRING, "", outputGroup,
		                          "o");

		plugin::addOptionsFromPlugins(
		    registry, plugin::InputFormatsChoice::ONLYLISTMODES);

		// Load configuration
		io::ArgumentReader config{
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

		globals::setNumThreads(numThreads);
		std::cout << "Initializing scanner..." << std::endl;
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		std::cout << "Reading input data..." << std::endl;
		std::unique_ptr<ProjectionData> dataInput = io::openProjectionData(
		    input_fname, input_format, *scanner, config.getAllArguments());
		const bool hasTOF = dataInput->hasTOF();
		const bool hasRandoms = dataInput->hasRandomsEstimates();

		std::cout << "Generating output ListModeLUT..." << std::endl;
		auto lmOut =
		    std::make_unique<ListModeLUTOwned>(*scanner, hasTOF, hasRandoms);
		const size_t numEvents = dataInput->count();
		lmOut->allocate(numEvents);

		ListModeLUTOwned* lmOut_ptr = lmOut.get();
		const ProjectionData* dataInput_ptr = dataInput.get();

		util::parallelForChunked(
		    numEvents, globals::getNumThreads(),
		    [lmOut_ptr, dataInput_ptr, hasTOF, hasRandoms](size_t evId,
		                                                   size_t /*tid*/)
		    {
			    lmOut_ptr->setTimestampOfEvent(
			        evId, dataInput_ptr->getTimestamp(evId));
			    auto [d1, d2] = dataInput_ptr->getDetectorPair(evId);
			    lmOut_ptr->setDetectorIdsOfEvent(evId, d1, d2);
			    if (hasTOF)
			    {
				    lmOut_ptr->setTOFValueOfEvent(
				        evId, dataInput_ptr->getTOFValue(evId));
			    }
			    if (hasRandoms)
			    {
				    lmOut_ptr->setRandomsEstimateOfEvent(
				        evId, dataInput_ptr->getRandomsEstimate(evId));
			    }
		    });

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
		util::printExceptionMessage(e);
		return -1;
	}
}
