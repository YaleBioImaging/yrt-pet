/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "datastruct/IO.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/Array.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"

#include <cxxopts.hpp>
#include <iostream>


int main(int argc, char** argv)
{
	try
	{
		std::string scanner_fname;
		std::string input_fname;
		std::string input_format;
		std::string out_fname;
		int numThreads = -1;

		Plugin::OptionsResult pluginOptionsResults;  // For plugins' options

		// Parse command line arguments
		cxxopts::Options options(
		    argv[0], "Accumulate a projection-space input into a "
		             "map of each detector used. Each value in the"
		             "map will represent a detector and the amount of"
		             "times it was used in the projection data. The"
		             "output file will be a RAWD file");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
			("s,scanner", "Scanner parameters file",
		    cxxopts::value<std::string>(scanner_fname))
			("i,input", "Input file",
			cxxopts::value<std::string>(input_fname))
			("f,format", "Input file format. Possible values: " +
			IO::possibleFormats(),
		    cxxopts::value<std::string>(input_format))
			("o,out", "Output map filename",
		    cxxopts::value<std::string>(out_fname))
			("num_threads", "Number of threads to use",
		    cxxopts::value<int>(numThreads))
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

		std::vector<std::string> required_params = {"scanner", "input", "out",
		                                            "format"};
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

		Globals::set_num_threads(numThreads);

		auto scanner = std::make_unique<Scanner>(scanner_fname);

		std::cout << "Reading input data..." << std::endl;

		std::unique_ptr<ProjectionData> dataInput = IO::openProjectionData(
		    input_fname, input_format, *scanner, pluginOptionsResults);

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
		Util::printExceptionMessage(e);
		return -1;
	}
}
