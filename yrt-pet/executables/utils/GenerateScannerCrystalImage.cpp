/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

/*******************************************************************************
 * Def.: Generate an image with a mask of the scanner crystals.
 ******************************************************************************/

#include "../ArgumentReader.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"
#include "yrt-pet/utils/Version.hpp"

#include <cxxopts.hpp>

#include <fstream>
#include <string>
#include <vector>

using namespace yrt;

int main(int argc, char* argv[])
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

		// Output file
		registry.registerArgument("out", "Output image filename", true,
		                          io::TypeOfArgument::STRING, "", outputGroup,
		                          "o");

		// Load configuration
		io::ArgumentReader config{registry,
		                          "Generate scanner crystal image. The"
		                          "output file will be a NIfTI file"};

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
		auto out_fname = config.getValue<std::string>("out");
		auto numThreads = config.getValue<int>("num_threads");
		auto spacingScaling = config.getValue<float>("spacing_scaling");

		globals::setNumThreads(numThreads);
		numThreads = globals::getNumThreads();

		auto scanner = std::make_unique<Scanner>(scanner_fname);
		float hsx = scanner->crystalDepth / 2;
		float hsy = scanner->crystalSize_trans / 2;
		float hsz = scanner->crystalSize_z / 2;
		Array2DOwned<float> lut;
		scanner->createLUT(lut);
		size_t numDets = scanner->getNumDets();
		float* lutPtr = lut.getRawPointer();

		// 1. Find extent of scanner
		float x_max = 0.f;
		float y_max = 0.f;
		float z_max = 0.f;
		for (size_t di = 0; di < numDets; di++)
		{
			if (std::abs(lut[di][0]) > x_max)
			{
				x_max = std::abs(lut[di][0]);
			}
			if (std::abs(lut[di][1]) > y_max)
			{
				y_max = std::abs(lut[di][1]);
			}
			if (std::abs(lut[di][2]) > z_max)
			{
				z_max = std::abs(lut[di][2]);
			}
		}

		float spacing =
		    std::min({scanner->crystalDepth, scanner->crystalSize_trans,
		              scanner->crystalSize_z}) /
		    spacingScaling;

		ssize_t n_sc_x = std::ceil(2.02f * x_max / spacing);
		ssize_t n_sc_y = std::ceil(2.02f * y_max / spacing);
		ssize_t n_sc_z = std::ceil(2.02f * z_max / spacing);

		float c_sc_x = -(static_cast<float>(n_sc_x) - 1.0) / 2.0;
		float c_sc_y = -(static_cast<float>(n_sc_y) - 1.0) / 2.0;
		float c_sc_z = -(static_cast<float>(n_sc_z) - 1.0) / 2.0;
		ImageParams imageParams(n_sc_x, n_sc_y, n_sc_z, spacing * n_sc_x,
		                        spacing * n_sc_y, spacing * n_sc_z);
		std::unique_ptr<ImageOwned> image;
		image = std::make_unique<ImageOwned>(imageParams);
		image->allocate();
		float* imagePtr = image->getRawPointer();

		util::ProgressDisplayMultiThread progressBar(
		    numThreads, n_sc_x * n_sc_y * n_sc_z, 5);

		util::parallelForChunked(
		    n_sc_x * n_sc_y * n_sc_z, numThreads,
		    [&progressBar, imagePtr, lutPtr, n_sc_x, n_sc_y, spacing, c_sc_x,
		     c_sc_y, c_sc_z, hsx, hsy, hsz, numDets](size_t pi, size_t threadId)
		    {
			    progressBar.incrementProgress(threadId, 1);
			    int zi = pi / (n_sc_x * n_sc_y);
			    ssize_t rest = pi - zi * n_sc_x * n_sc_y;
			    int yi = rest / n_sc_x;
			    int xi = rest % n_sc_x;
			    float z = (zi + c_sc_z) * spacing;
			    float y = (yi + c_sc_y) * spacing;
			    float x = (xi + c_sc_x) * spacing;
			    bool done = false;
			    size_t di = 0;
			    while (di < numDets && !done)
			    {
				    auto& [cx, cy, cz, nx, ny, nz] =
				        reinterpret_cast<float(&)[6]>(lutPtr[6 * di]);
				    float axis_u_x = nx;
				    float axis_u_y = ny;
				    float axis_u_z = 0.0;
				    float axis_v_x = -ny;
				    float axis_v_y = nx;
				    float axis_v_z = 0.0;
				    float axis_w_x = 0.0;
				    float axis_w_y = 0.0;
				    float axis_w_z = 1.0;
				    float dx = x - cx;
				    float dy = y - cy;
				    float dz = z - cz;
				    float proj_u =
				        dx * axis_u_x + dy * axis_u_y + dz * axis_u_z;
				    float proj_v =
				        dx * axis_v_x + dy * axis_v_y + dz * axis_v_z;
				    float proj_w =
				        dx * axis_w_x + dy * axis_w_y + dz * axis_w_z;
				    if ((std::abs(proj_u) <= hsx) &&
				        (std::abs(proj_v) <= hsy) && (std::abs(proj_w) <= hsz))
				    {
					    imagePtr[pi] = di;
					    done = true;
				    }
				    di++;
			    }
		    });

		image->writeToFile(out_fname);

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
