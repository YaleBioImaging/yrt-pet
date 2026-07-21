/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/utils/Version.hpp"

#include <cstring>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace yrt;

static constexpr int PULSE_BYTES = 88;
static constexpr int COINC_BYTES = 176;
static constexpr int DET_PER_RING = 896;
static constexpr int N_RINGS = 112;
static constexpr int DET_PER_LAYER = DET_PER_RING * N_RINGS;

struct Pulse
{
	int32_t eventID;
	double sourcePosX, sourcePosY, sourcePosZ;
	double time;
	double posX, posY, posZ;
	int32_t volumeID[6];
	int32_t nPhantomCompton;

	void read(const char* buf)
	{
		int off = 0;
		std::memcpy(&eventID, buf + off, 4);
		off += 4;
		std::memcpy(&sourcePosX, buf + off, 8);
		off += 8;
		std::memcpy(&sourcePosY, buf + off, 8);
		off += 8;
		std::memcpy(&sourcePosZ, buf + off, 8);
		off += 8;
		std::memcpy(&time, buf + off, 8);
		off += 8;
		std::memcpy(&posX, buf + off, 8);
		off += 8;
		std::memcpy(&posY, buf + off, 8);
		off += 8;
		std::memcpy(&posZ, buf + off, 8);
		off += 8;
		for (int i = 0; i < 6; i++)
		{
			std::memcpy(&volumeID[i], buf + off, 4);
			off += 4;
		}
		std::memcpy(&nPhantomCompton, buf + off, 4);
	}

	uint32_t flat_id() const
	{
		int rsector = volumeID[1];
		int module = volumeID[2];
		int submodule = volumeID[3];
		int crystal = volumeID[4];
		int layer = volumeID[5];
		int sm_y = submodule % 2;
		int sm_z = submodule / 2;
		int cr_y = crystal % 8;
		int cr_z = crystal / 8;
		int ring = module * 8 + sm_z * 4 + cr_z;
		int det_in_ring = rsector * 16 + sm_y * 8 + cr_y;
		return layer * DET_PER_LAYER + ring * DET_PER_RING + det_in_ring;
	}
};

static std::vector<std::string> splitComma(const std::string& s)
{
	std::vector<std::string> parts;
	std::istringstream ss(s);
	std::string part;
	while (std::getline(ss, part, ','))
	{
		if (!part.empty())
			parts.push_back(part);
	}
	return parts;
}

static std::vector<std::string> readFileList(const std::string& fname)
{
	std::vector<std::string> paths;
	std::ifstream fin(fname);
	if (!fin)
	{
		std::cerr << "WARNING: cannot open file list " << fname << std::endl;
		return paths;
	}
	std::string line;
	while (std::getline(fin, line))
	{
		if (!line.empty())
			paths.push_back(line);
	}
	return paths;
}

int main(int argc, char** argv)
{
	try
	{
		std::string scanner_fname;
		std::string input_list;
		std::string input_file_fname;
		std::string out_fname;
		bool trues_only = false;
		int numThreads = -1;

		cxxopts::Options options(
		    argv[0],
		    "Convert Gate binary coincidence files (176 bytes each) into a "
		    "YRT-PET Histogram3D");

		/* clang-format off */
		options.add_options()
			("s,scanner", "Scanner parameters file",
			 cxxopts::value<std::string>(scanner_fname))
			("i,input", "Comma-separated list of Gate binary coincidence files",
			 cxxopts::value<std::string>(input_list))
			("input-file", "File containing list of input files (one per line)",
			 cxxopts::value<std::string>(input_file_fname))
			("o,out", "Output Histogram3D filename",
			 cxxopts::value<std::string>(out_fname))
			("trues-only", "Accumulate trues only (eventID match and no phantom "
			 "Compton scattering); default: all coincidences",
			 cxxopts::value<bool>(trues_only))
			("num_threads", "Number of threads to use",
			 cxxopts::value<int>(numThreads))
			("version", "Print version information")
			("h,help", "Print help");
		/* clang-format on */

		const auto result = options.parse(argc, argv);
		if (result.count("version"))
		{
			yrt::version::printVersion();
			return 0;
		}
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		bool missing_args = false;
		if (result.count("scanner") == 0)
		{
			std::cerr << "Argument 'scanner' missing" << std::endl;
			missing_args = true;
		}
		if (result.count("out") == 0)
		{
			std::cerr << "Argument 'out' missing" << std::endl;
			missing_args = true;
		}
		if (result.count("input") == 0 && result.count("input-file") == 0)
		{
			std::cerr << "Either 'input' or 'input-file' is required"
			          << std::endl;
			missing_args = true;
		}
		if (missing_args)
		{
			std::cerr << options.help() << std::endl;
			return -1;
		}

		globals::setNumThreads(numThreads);

		std::cout << "Initializing scanner..." << std::endl;
		auto scanner = std::make_unique<Scanner>(scanner_fname);
		const size_t num_dets = scanner->getExpectedNumDets();

		std::cout << "Number of detectors: " << num_dets << std::endl;

		std::cout << "Preparing output Histogram3D..." << std::endl;
		auto histo = std::make_unique<Histogram3DOwned>(*scanner);
		histo->allocate();
		histo->clearProjections(0.0f);

		std::vector<std::string> filenames;
		if (result.count("input"))
		{
			auto parts = splitComma(input_list);
			filenames.insert(filenames.end(), parts.begin(), parts.end());
		}
		if (result.count("input-file"))
		{
			auto parts = readFileList(input_file_fname);
			filenames.insert(filenames.end(), parts.begin(), parts.end());
		}
		std::cout << "Processing " << filenames.size() << " input files..."
		          << std::endl;

		char buf[COINC_BYTES];
		int64_t total_coinc = 0;
		int64_t accepted_coinc = 0;
		int64_t rejected_trues = 0;

		for (size_t fi = 0; fi < filenames.size(); fi++)
		{
			const auto& fname = filenames[fi];
			std::ifstream fin(fname, std::ios::binary);
			if (!fin)
			{
				std::cerr << "WARNING: cannot open " << fname << ", skipping"
				          << std::endl;
				continue;
			}

			int64_t n_file = 0;
			while (fin.read(buf, COINC_BYTES))
			{
				Pulse p1, p2;
				p1.read(buf);
				p2.read(buf + PULSE_BYTES);

				uint32_t d1 = p1.flat_id();
				uint32_t d2 = p2.flat_id();

				if (trues_only &&
				    (p1.eventID != p2.eventID || p1.nPhantomCompton > 0 ||
				     p2.nPhantomCompton > 0))
				{
					rejected_trues++;
					n_file++;
					continue;
				}

				if (d1 >= num_dets || d2 >= num_dets)
				{
					std::cerr << "WARNING: invalid detector IDs: " << d1 << ", "
					          << d2 << " in " << fname << std::endl;
					n_file++;
					continue;
				}

				auto bin = histo->getBinIdFromDetPair(d1, d2);
				histo->incrementProjection(bin, 1.0f);
				accepted_coinc++;
				n_file++;
			}
			fin.close();
			total_coinc += n_file;

			if ((fi + 1) % 10 == 0 || fi == filenames.size() - 1)
			{
				std::cout << "  Processed " << (fi + 1) << "/"
				          << filenames.size() << " files, " << n_file
				          << " coinc in current file, total accepted: "
				          << accepted_coinc << std::endl;
			}
		}

		std::cout << "\n===== SUMMARY =====" << std::endl;
		std::cout << "Total coincidences read: " << total_coinc << std::endl;
		std::cout << "Accepted into histogram: " << accepted_coinc << std::endl;
		if (trues_only)
		{
			std::cout << "Rejected (not trues):   " << rejected_trues
			          << std::endl;
		}
		std::cout << std::endl;

		std::cout << "Writing Histogram3D to " << out_fname << "..."
		          << std::endl;
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
		util::printExceptionMessage(e);
		return -1;
	}
}
