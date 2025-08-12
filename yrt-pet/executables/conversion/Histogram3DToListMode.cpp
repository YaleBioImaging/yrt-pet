/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
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
		std::string scanner_fname;
		std::string input_fname;
		std::string out_fname;
		size_t numEvents = 0;
		int numThreads = -1;

		// Parse command line arguments
		cxxopts::Options options(
		    argv[0],
		    "Convert a fully 3D dense histogram into list-mode (ListModeLUT)");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
		("s,scanner", "Scanner parameters file", cxxopts::value<std::string>(scanner_fname))
		("i,input", "Input histogram file", cxxopts::value<std::string>(input_fname))
		("o,out", "Output list-mode filename", cxxopts::value<std::string>(out_fname))
		("n,num", "Number of list-mode events", cxxopts::value<size_t>(numEvents))
		("num_threads", "Number of threads to use", cxxopts::value<int>(numThreads))
		("h,help", "Print help");
		/* clang-format on */

		const auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		std::vector<std::string> required_params = {"scanner", "input", "out"};
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

		globals::setNumThreads(numThreads);

		const auto scanner = std::make_unique<Scanner>(scanner_fname);
		const auto histo =
		    std::make_unique<Histogram3DOwned>(*scanner, input_fname);
		const auto lm = std::make_unique<ListModeLUTOwned>(*scanner);

		util::histogram3DToListModeLUT(histo.get(), lm.get(), numEvents);

		// TODO: We should shuffle the events afterwards

		lm->writeToFile(out_fname);

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
