/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ExperimentalBackend.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Tools.hpp"
#include "yrt-pet/utils/Version.hpp"

#include <cxxopts.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{

struct CompareStats
{
	float maxAbsDiff = 0.0f;
	float maxRelDiff = 0.0f;
	std::size_t mismatchCount = 0;
};

std::size_t imageVoxelCount(const yrt::ImageParams& params)
{
	return static_cast<std::size_t>(params.nx) *
	       static_cast<std::size_t>(params.ny) *
	       static_cast<std::size_t>(params.nz) *
	       static_cast<std::size_t>(params.nt);
}

CompareStats compareImages(const yrt::Image& cpuImage,
                           const yrt::Image& metalImage, yrt::Image& diffImage,
                           float absTolerance, float relTolerance)
{
	CompareStats stats;
	const std::size_t count = imageVoxelCount(cpuImage.getParams());
	const float* cpuPtr = cpuImage.getRawPointer();
	const float* metalPtr = metalImage.getRawPointer();
	float* diffPtr = diffImage.getRawPointer();

	for (std::size_t i = 0; i < count; ++i)
	{
		const float diff = metalPtr[i] - cpuPtr[i];
		diffPtr[i] = diff;
		const float absDiff = std::fabs(diff);
		const float scale = std::max(1.0f, std::fabs(cpuPtr[i]));
		const float relDiff = absDiff / scale;
		stats.maxAbsDiff = std::max(stats.maxAbsDiff, absDiff);
		stats.maxRelDiff = std::max(stats.maxRelDiff, relDiff);
		if (absDiff > absTolerance + relTolerance * scale)
		{
			++stats.mismatchCount;
		}
	}

	return stats;
}

void writeImageIfRequested(const yrt::Image& image,
                           const std::string& filename)
{
	if (!filename.empty())
	{
		image.writeToFile(filename);
	}
}

}  // namespace

int main(int argc, char** argv)
{
	try
	{
		std::string inputFilename;
		std::string psfFilename;
		std::string cpuOutFilename;
		std::string metalOutFilename;
		std::string diffOutFilename;
		float absTolerance = 1.0e-4f;
		float relTolerance = 1.0e-4f;
		bool applyAdjoint = false;

		cxxopts::Options options(
		    argv[0],
		    "Compare CPU OperatorPsf with explicit opt-in Metal PSF helpers");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
		("i,input", "Input image file",
		 cxxopts::value<std::string>(inputFilename))
		("p,psf", "Uniform image-space PSF CSV file",
		 cxxopts::value<std::string>(psfFilename))
		("cpu-out", "Optional CPU PSF output image file",
		 cxxopts::value<std::string>(cpuOutFilename))
		("metal-out", "Optional Metal PSF output image file",
		 cxxopts::value<std::string>(metalOutFilename))
		("diff-out", "Optional signed difference image file (Metal - CPU)",
		 cxxopts::value<std::string>(diffOutFilename))
		("atol", "Absolute comparison tolerance",
		 cxxopts::value<float>(absTolerance)->default_value("1e-4"))
		("rtol", "Relative comparison tolerance",
		 cxxopts::value<float>(relTolerance)->default_value("1e-4"))
		("adjoint", "Compare adjoint PSF (AH) instead of forward PSF (A)",
		 cxxopts::value<bool>(applyAdjoint)->default_value("false"))
		("version", "Print version information")
		("h,help", "Print help");
		/* clang-format on */

		auto result = options.parse(argc, argv);
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

		std::vector<std::string> requiredParams = {"input", "psf"};
		bool missingArgs = false;
		for (const auto& param : requiredParams)
		{
			if (result.count(param) == 0)
			{
				std::cerr << "Argument '" << param << "' missing"
				          << std::endl;
				missingArgs = true;
			}
		}
		if (missingArgs)
		{
			std::cerr << options.help() << std::endl;
			return -1;
		}

		yrt::backend::metal::ExperimentalBackend backend;
		if (!backend.isAvailable())
		{
			std::cerr << "Metal PSF compare: FAIL "
			          << "(Metal device unavailable)\n";
			return 2;
		}
		if (!backend.isValid())
		{
			std::cerr << "Metal PSF compare: FAIL ("
			          << backend.errorMessage() << ")\n";
			return 1;
		}

		yrt::ImageOwned inputImage(inputFilename);
		const yrt::ImageParams& params = inputImage.getParams();
		yrt::ImageOwned cpuOutput(params);
		yrt::ImageOwned metalOutput(params);
		yrt::ImageOwned diffOutput(params);
		cpuOutput.allocate();
		metalOutput.allocate();
		diffOutput.allocate();

		yrt::OperatorPsf cpuPsf(psfFilename);
		if (applyAdjoint)
		{
			cpuPsf.applyAH(&inputImage, &cpuOutput);
			if (!backend.applyOperatorPsfAdjoint(inputImage, metalOutput,
			        psfFilename))
			{
				std::cerr << "Metal PSF compare: FAIL "
				          << "(Metal adjoint PSF call failed)\n";
				return 1;
			}
		}
		else
		{
			cpuPsf.applyA(&inputImage, &cpuOutput);
			if (!backend.applyOperatorPsfForward(inputImage, metalOutput,
			        psfFilename))
			{
				std::cerr << "Metal PSF compare: FAIL "
				          << "(Metal forward PSF call failed)\n";
				return 1;
			}
		}

		const CompareStats stats = compareImages(
		    cpuOutput, metalOutput, diffOutput, absTolerance, relTolerance);
		writeImageIfRequested(cpuOutput, cpuOutFilename);
		writeImageIfRequested(metalOutput, metalOutFilename);
		writeImageIfRequested(diffOutput, diffOutFilename);

		std::cout << "Metal PSF compare: max_abs_diff="
		          << stats.maxAbsDiff << " max_rel_diff="
		          << stats.maxRelDiff << " mismatches="
		          << stats.mismatchCount << '\n';
		if (stats.mismatchCount != 0)
		{
			std::cerr << "Metal PSF compare: FAIL\n";
			return 1;
		}

		std::cout << "Metal PSF compare: PASS\n";
		return 0;
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception& e)
	{
		yrt::util::printExceptionMessage(e);
		return -1;
	}
}
