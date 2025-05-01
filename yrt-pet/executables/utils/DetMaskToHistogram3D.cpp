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
#include <datastruct/projection/Histogram3D.hpp>
#include <iostream>


int main(int argc, char** argv)
{
	try
	{
		IO::ArgumentRegistry registry{};

		std::string coreGroup = "0. Core";
		std::string inputGroup = "1. Input";
		std::string outputGroup = "2. Output";

		// Core parameters
		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          IO::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, IO::TypeOfArgument::INT, -1,
		                          coreGroup);

		// Input data parameters
		registry.registerArgument("input", "Input detector mask", true,
		                          IO::TypeOfArgument::STRING, "", inputGroup,
		                          "i");

		// Output file
		registry.registerArgument("out", "Output Histogram3D filename", true,
		                          IO::TypeOfArgument::STRING, "", outputGroup,
		                          "o");

		// Load configuration
		IO::ArgumentReader config{registry,
		                          "Temportary executable - Delete me"};

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
		auto out_fname = config.getValue<std::string>("out");
		auto numThreads = config.getValue<int>("num_threads");


		Globals::set_num_threads(numThreads);

		auto scanner = std::make_unique<Scanner>(scanner_fname);

		std::cout << "Reading input data..." << std::endl;

		// Read detector mask

		std::ifstream fin(input_fname.c_str(), std::ios::in | std::ios::binary);
		if (!fin.good())
		{
			throw std::runtime_error("Error reading input file " + input_fname);
		}

		// first check that file has the right size:
		fin.seekg(0, std::ios::end);
		size_t end = fin.tellg();
		fin.seekg(0, std::ios::beg);
		size_t begin = fin.tellg();
		size_t fileSize = end - begin;

		size_t numElem = fileSize / sizeof(bool);

		if (numElem != scanner->getNumDets())
		{
			throw std::logic_error("Error: Input file has incorrect size");
		}

		auto detMask = std::make_unique<bool[]>(numElem);

		bool* detMask_ptr = detMask.get();

		fin.read(reinterpret_cast<char*>(detMask_ptr), numElem * sizeof(bool));


		// End - Read detector mask

		// Create histogram
		std::cout << "Creating Histogram3D..." << std::endl;

		auto histo = std::make_unique<Histogram3DOwned>(*scanner);
		histo->allocate();
		histo->clearProjections(1.0f);

		const size_t numBins = histo->count();
		Histogram3D* histo_ptr = histo.get();
		float* histoValues_ptr = histo->getData().getRawPointer();

		std::cout << "Populating Histogram3D..." << std::endl;

#pragma omp parallel for default(none) \
    firstprivate(numBins, histo_ptr, detMask_ptr, histoValues_ptr)
		for (bin_t bin = 0; bin < numBins; ++bin)
		{
			const det_pair_t detPair = histo_ptr->getDetectorPair(bin);

			if (!detMask_ptr[detPair.d1] || !detMask_ptr[detPair.d2])
			{
				histoValues_ptr[bin] = 0.0f;
			}
		}

		std::cout << "Saving Histogram3D..." << std::endl;
		histo->writeToFile(out_fname);
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
