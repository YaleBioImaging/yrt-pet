/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/projection/SparseHistogram.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
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

		std::string coreGroup = "0. Core";
		std::string inputGroup = "1. Input";
		std::string outputGroup = "2. Output";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, io::TypeOfArgument::INT, -1,
		                          coreGroup);
		registry.registerArgument(
		    "input",
		    "Input histogram files (separated by commas, without spaces)", true,
		    io::TypeOfArgument::VECTOR_OF_STRINGS, "", inputGroup, "i");
		registry.registerArgument(
		    "format",
		    "Input file format. Possible values: " + io::possibleFormats(),
		    true, io::TypeOfArgument::STRING, "", inputGroup, "f");

		registry.registerArgument("out", "Output histogram filename", true,
		                          io::TypeOfArgument::STRING, "", outputGroup,
		                          "o");
		registry.registerArgument("sparse", "Convert to a sparse histogram",
		                          false, io::TypeOfArgument::BOOL, false,
		                          outputGroup);

		plugin::addOptionsFromPlugins(
		    registry, plugin::InputFormatsChoice::ONLYHISTOGRAMS);

		// Load configuration
		io::ArgumentReader config{
		    registry,
		    "Take several histograms (of any format, including plugin formats) "
		    "and accumulate them into a total histogram (either fully 3D "
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
		auto input_fnames = config.getValue<std::vector<std::string>>("input");
		auto input_format = config.getValue<std::string>("format");
		auto out_fname = config.getValue<std::string>("out");
		bool toSparseHistogram = config.getValue<bool>("sparse");
		int numThreads = config.getValue<int>("num_threads");

		globals::setNumThreads(numThreads);
		std::cout << "Initializing scanner..." << std::endl;
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		std::unique_ptr<Histogram> histoOut;
		if (toSparseHistogram)
		{
			histoOut = std::make_unique<SparseHistogram>(*scanner);
		}
		else
		{
			auto histo3DOut = std::make_unique<Histogram3DOwned>(*scanner);
			histo3DOut->allocate();
			histo3DOut->clearProjections(0.0f);
			histoOut = std::move(histo3DOut);
		}

		bool histo3DToHisto3D = input_format == "H" && !toSparseHistogram;

		for (const auto& input_fname : input_fnames)
		{
			std::cout << "Reading input data..." << std::endl;

			std::unique_ptr<ProjectionData> dataInput = io::openProjectionData(
			    input_fname, input_format, *scanner, config.getAllArguments());

			if (toSparseHistogram)
			{
				auto* sparseHisto =
				    reinterpret_cast<SparseHistogram*>(histoOut.get());
				std::cout << "Accumulating into sparse histogram..."
				          << std::endl;
				sparseHisto->accumulate<true>(*dataInput);
			}
			else
			{
				auto histo3DOut =
				    reinterpret_cast<Histogram3D*>(histoOut.get());
				if (histo3DToHisto3D)
				{
					std::cout << "Adding Histogram3D..." << std::endl;
					const auto* dataInputHisto3D =
					    dynamic_cast<const Histogram3D*>(dataInput.get());
					ASSERT(dataInputHisto3D != nullptr);

					histo3DOut->getData() += dataInputHisto3D->getData();
				}
				else
				{
					std::cout << "Accumulating Histogram into Histogram3D..."
					          << std::endl;
					util::convertToHistogram3D<false>(*dataInput, *histo3DOut);
				}
			}
		}

		if (toSparseHistogram)
		{
			const auto* sparseHisto =
			    reinterpret_cast<const SparseHistogram*>(histoOut.get());
			std::cout << "Saving output sparse histogram..." << std::endl;
			sparseHisto->writeToFile(out_fname);
		}
		else
		{
			const auto* histo3DOut =
			    reinterpret_cast<const Histogram3D*>(histoOut.get());
			std::cout << "Saving output Histogram3D..." << std::endl;
			histo3DOut->writeToFile(out_fname);
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
		util::printExceptionMessage(e);
		return -1;
	}
}
