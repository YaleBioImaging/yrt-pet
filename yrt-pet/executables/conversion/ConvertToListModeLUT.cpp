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
		std::string scanner_fname;
		std::string input_fname;
		std::string input_format;
		std::string out_fname;
		int numThreads = -1;

		Plugin::OptionsResult pluginOptionsResults;  // For plugins' options

		// Parse command line arguments
		cxxopts::Options options(argv[0],
		                         "Convert any input format to a histogram");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
		("s,scanner", "Scanner parameters file name", cxxopts::value<std::string>(scanner_fname))
		("i,input", "Input file", cxxopts::value<std::string>(input_fname))
		("f,format", "Input file format. Possible values: " +
			IO::possibleFormats(Plugin::InputFormatsChoice::ONLYLISTMODES),
			cxxopts::value<std::string>(input_format))
		("o,out", "Output listmode filename", cxxopts::value<std::string>(out_fname))
		("num_threads", "Number of threads to use", cxxopts::value<int>(numThreads))
		("h,help", "Print help");
		/* clang-format on */

		// Add plugin options
		PluginOptionsHelper::fillOptionsFromPlugins(
		    options, Plugin::InputFormatsChoice::ONLYLISTMODES);

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
			det_pair_t detPair = dataInput_ptr->getDetectorPair(evId);
			lmOut_ptr->setDetectorIdsOfEvent(evId, detPair.d1, detPair.d2);
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
