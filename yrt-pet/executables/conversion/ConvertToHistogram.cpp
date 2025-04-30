/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "datastruct/IO.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/projection/SparseHistogram.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ReconstructionUtils.hpp"

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
		registry.registerArgument("input", "Input projection data file", true,
		                          IO::TypeOfArgument::STRING, "", inputGroup,
		                          "i");
		registry.registerArgument(
		    "format",
		    "Input file format. Possible values: " + IO::possibleFormats(),
		    true, IO::TypeOfArgument::STRING, "", inputGroup, "f");

		registry.registerArgument("out", "Output histogram filename", true,
		                          IO::TypeOfArgument::STRING, "", outputGroup,
		                          "o");
		registry.registerArgument("sparse", "Convert to a sparse histogram",
		                          false, IO::TypeOfArgument::BOOL, false,
		                          outputGroup);

		PluginOptionsHelper::addOptionsFromPlugins(
			registry, Plugin::InputFormatsChoice::ALL);

		// Load configuration
		IO::ArgumentReader config{
		    registry,
		    "Convert any input format to a histogram (either fully 3D "
		    "dense histogram or sparse histogram)"};

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
		bool toSparseHistogram = config.getValue<bool>("sparse");
		int numThreads = config.getValue<int>("num_threads");

		Globals::set_num_threads(numThreads);
		std::cout << "Initializing scanner..." << std::endl;
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		std::cout << "Reading input data..." << std::endl;

		std::unique_ptr<ProjectionData> dataInput = IO::openProjectionData(
		    input_fname, input_format, *scanner, config.getAllArguments());

		if (toSparseHistogram)
		{
			std::cout << "Accumulating into sparse histogram..." << std::endl;
			auto sparseHisto =
			    std::make_unique<SparseHistogram>(*scanner, *dataInput);
			std::cout << "Saving sparse histogram..." << std::endl;
			sparseHisto->writeToFile(out_fname);
		}
		else
		{
			std::cout << "Preparing output Histogram3D..." << std::endl;
			auto histoOut = std::make_unique<Histogram3DOwned>(*scanner);
			histoOut->allocate();
			histoOut->clearProjections(0.0f);

			std::cout << "Accumulating into Histogram3D..." << std::endl;
			if (IO::isFormatListMode(input_format))
			{
				// ListMode input, use atomic to accumulate
				Util::convertToHistogram3D<true>(*dataInput, *histoOut);
			}
			else
			{
				// Histogram input, no need to use atomic to accumulate
				Util::convertToHistogram3D<false>(*dataInput, *histoOut);
			}

			std::cout << "Histogram3D generated.\nWriting file..." << std::endl;
			histoOut->writeToFile(out_fname);
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
