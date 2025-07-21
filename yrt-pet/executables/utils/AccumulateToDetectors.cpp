/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Array.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"

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

		// Core parameters
		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, io::TypeOfArgument::INT, -1,
		                          coreGroup);

		// Input data parameters
		registry.registerArgument("input", "Input file", true,
		                          io::TypeOfArgument::STRING, "", inputGroup,
		                          "i");
		registry.registerArgument(
		    "format",
		    "Input file format. Possible values: " + io::possibleFormats(),
		    true, io::TypeOfArgument::STRING, "", inputGroup, "f");

		// Output file
		registry.registerArgument("out", "Output map filename", true,
		                          io::TypeOfArgument::STRING, "", outputGroup,
		                          "o");

		plugin::addOptionsFromPlugins(registry,
		                              plugin::InputFormatsChoice::ALL);

		// Load configuration
		io::ArgumentReader config{
		    registry, "Accumulate a projection-space input into a "
		              "map of each detector used. Each value in the"
		              "map will represent a detector and the amount of"
		              "times it was used in the projection data. The"
		              "output file will be a RAWD file"};

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
		auto numThreads = config.getValue<int>("num_threads");


		globals::setNumThreads(numThreads);

		auto scanner = std::make_unique<Scanner>(scanner_fname);

		std::cout << "Reading input data..." << std::endl;

		std::unique_ptr<ProjectionData> dataInput = io::openProjectionData(
		    input_fname, input_format, *scanner, config.getAllArguments());

		auto map = std::make_unique<Array3D<float>>();
		map->allocate(scanner->numDOI, scanner->numRings, scanner->detsPerRing);
		map->fill(0.0f);

		const size_t numBins = dataInput->count();
		const size_t numDets = scanner->getNumDets();
		float* mapPtr = map->getRawPointer();
		ProjectionData* dataInputPtr = dataInput.get();

#pragma omp parallel for default(none) \
    firstprivate(numBins, mapPtr, dataInputPtr, numDets)
		for (bin_t bin = 0; bin < numBins; ++bin)
		{
			const det_pair_t detPair = dataInputPtr->getDetectorPair(bin);
			ASSERT_MSG(detPair.d1 < numDets && detPair.d2 < numDets,
			           "Invalid Detector Id");
#pragma omp atomic
			mapPtr[detPair.d1]++;
#pragma omp atomic
			mapPtr[detPair.d2]++;
		}

		map->writeToFile(out_fname);
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
