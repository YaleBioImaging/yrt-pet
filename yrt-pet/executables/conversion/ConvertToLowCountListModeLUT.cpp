/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/datastruct/scanner/DetectorMask.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplay.hpp"
#include "yrt-pet/utils/Tools.hpp"

#include <cxxopts.hpp>
#include <iostream>
#include <random>
#include <vector>

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
		registry.registerArgument("detmask",
		                          "Detector mask (to disable a given set of "
		                          "detectors) [Ignored for now]",
		                          false, io::TypeOfArgument::STRING, "",
		                          inputGroup);

		registry.registerArgument("out",
		                          "Output prefix (files: prefix_0.lmDat, "
		                          "prefix_1.lmDat, ...)",
		                          true, io::TypeOfArgument::STRING, "",
		                          outputGroup, "o");
		registry.registerArgument(
		    "num_splits", "Number of low-dose sub-listmodes to generate", false,
		    io::TypeOfArgument::INT, 10, outputGroup, "N");
		registry.registerArgument("seed", "RNG seed", false,
		                          io::TypeOfArgument::INT, -1, outputGroup);

		plugin::addOptionsFromPlugins(
		    registry, plugin::InputFormatsChoice::ONLYLISTMODES);

		// Load configuration
		io::ArgumentReader config{
		    registry, "Generate N low-dose list-mode files from a "
		              "full-dose listmode using Poisson-thinning with p=1/N."};

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
		auto detmask_fname = config.getValue<std::string>("detmask");
		auto out_prefix = config.getValue<std::string>("out");
		int numThreads = config.getValue<int>("num_threads");
		int N = config.getValue<int>("num_splits");

		int seedArg = config.getValue<int>("seed");  // random seed
		uint64_t seed;
		if (seedArg < 0)
		{
			seed = std::random_device{}();
			std::cout << "Using random seed: " << seed << std::endl;
		}
		else
		{
			seed = static_cast<uint64_t>(seedArg);
		}

		globals::setNumThreads(numThreads);

		// Load scanner
		std::cout << "Initializing scanner..." << std::endl;
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		// Load input listmode
		std::cout << "Reading input data..." << std::endl;
		std::unique_ptr<ProjectionData> dataInput = io::openProjectionData(
		    input_fname, input_format, *scanner, config.getAllArguments());
		auto* lm = dynamic_cast<ListMode*>(dataInput.get());
		ASSERT_MSG(lm != nullptr, "The input file seems to not be list-mode");

		const size_t nEvents = lm->count();
		const bool hasTOF = lm->hasTOF();
		const bool hasRandoms = lm->hasRandomsEstimates();
		std::cout << "  " << nEvents << " events (TOF=" << hasTOF
		          << ", randoms=" << hasRandoms << ")" << std::endl;

		// TODO: Apply detector mask
		std::unique_ptr<DetectorMask> detmask = nullptr;
		if (!detmask_fname.empty())
		{
			std::cout << "Reading detector mask..." << std::endl;
			detmask = std::make_unique<DetectorMask>(detmask_fname);
		}

		// Pass 1: Assign each event randomly to a sub-listmode
		std::cout << "Splitting into " << N << " sub-listmodes..." << std::endl;
		std::mt19937_64 rng(seed);  // random seed
		std::uniform_int_distribution<int> lmDist(
		    0, N - 1);  // uniform distribution over [0, N-1]

		std::vector<uint8_t> eventToLmMap(nEvents);
		std::vector<size_t> lmSizes(N, 0);

		// For each event, randomly assign it to a sub-listmode and count
		// the number of events in each sub-listmode
		util::ProgressDisplay assignProgress(nEvents, 5);
		for (size_t i = 0; i < nEvents; ++i)
		{
			const int b = lmDist(rng);
			eventToLmMap[i] = static_cast<uint8_t>(b);
			++lmSizes[b];
			assignProgress.progress(i + 1);
		}

		std::cout << "  list-mode sizes:";
		for (int b = 0; b < N; ++b)
			std::cout << " [" << b << "]=" << lmSizes[b];
		std::cout << std::endl;

		// Allocate N output list-modes
		std::vector<std::unique_ptr<ListModeLUTOwned>> outputLUTs;
		outputLUTs.reserve(N);
		for (int b = 0; b < N; ++b)
		{
			auto lut = std::make_unique<ListModeLUTOwned>(*scanner, hasTOF,
			                                              hasRandoms);
			lut->allocate(lmSizes[b]);
			outputLUTs.push_back(std::move(lut));
		}

		// Pass 2: Copy events into their assigned LUT
		// Random rate: scaled by 1/N each sub-listmode
		const float randomsScale = 1.0f / static_cast<float>(N);
		std::cout << "Copying events..." << std::endl;
		util::ProgressDisplay copyProgress(nEvents, 5);
		std::vector<size_t> writeCursors(N, 0);
		for (size_t i = 0; i < nEvents; ++i)
		{
			const int b = eventToLmMap[i];
			const size_t writeIdx = writeCursors[b]++;
			auto& out = *outputLUTs[b];

			out.setTimestampOfEvent(writeIdx, lm->getTimestamp(i));
			out.setDetectorIdsOfEvent(writeIdx, lm->getDetector1(i),
			                          lm->getDetector2(i));

			if (hasTOF)
				out.setTOFValueOfEvent(writeIdx, lm->getTOFValue(i));

			if (hasRandoms)
				out.setRandomsEstimateOfEvent(
				    writeIdx, lm->getRandomsEstimate(i) * randomsScale);

			copyProgress.progress(i + 1);
		}

		// Write output files
		std::cout << "Writing output files..." << std::endl;
		const int numDigitsInFilename = N > 1 ? util::numberOfDigits(N - 1) : 1;
		for (int b = 0; b < N; ++b)
		{
			const std::string fname = util::addBeforeExtension(
			    out_prefix,
			    "_" + util::padZeros(b, numDigitsInFilename) + ".lmDat");
			outputLUTs[b]->writeToFile(fname);
			std::cout << "  " << fname << " (" << outputLUTs[b]->count()
			          << " events)" << std::endl;
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
