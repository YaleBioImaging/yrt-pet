/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/IO.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "operators/OperatorProjector.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ReconstructionUtils.hpp"
#include "utils/Tools.hpp"

#include <cxxopts.hpp>
#include <iostream>

int main(int argc, char** argv)
{
	try
	{
		std::string scanner_fname;
		std::string inputImage_fname;
		std::string attImg_fname;
		std::string imageSpacePsf_fname;
		std::string projSpacePsf_fname;
		std::string outHis_fname;
		std::string projector_name = "S";
		int numThreads = -1;
		int numSubsets = 1;
		int subsetId = 0;
		int numRays = 1;
		bool convertToAcf = false;
		float tofWidth_ps = 0.0;
		int tofNumStd = 0;

		// Parse command line arguments
		cxxopts::Options options(argv[0],
		                         "Forward projection driver. Forward "
		                         "project an image into a Histogram3D");
		options.positional_help("[optional args]").show_positional_help();
		/* clang-format off */
		options.add_options()
		("s,scanner", "Scanner parameters file name", cxxopts::value<std::string>(scanner_fname))
		("i,input", "Input image file", cxxopts::value<std::string>(inputImage_fname))
		("att", "Attenuation image filename", cxxopts::value<std::string>(attImg_fname))
		("psf", "Image-space PSF kernel file", cxxopts::value<std::string>(imageSpacePsf_fname))
		("proj_psf", "Projection-space PSF kernel file", cxxopts::value<std::string>(projSpacePsf_fname))
		("projector", "Projector to use, choices: Siddon (S), Distance-Driven (D)"
		#if BUILD_CUDA
		", or GPU Distance-Driven (DD_GPU)"
		#endif
		". The default projector is Siddon", cxxopts::value<std::string>(projector_name))
		("acf", "Generate ACF histogram", cxxopts::value<bool>(convertToAcf))
		("tof_width_ps", "TOF Width in Picoseconds", cxxopts::value<float>(tofWidth_ps))
		("tof_n_std", "Number of standard deviations to consider for TOF's Gaussian curve", cxxopts::value<int>(tofNumStd))
		("num_rays", "Number of rays to use in the Siddon projector", cxxopts::value<int>(numRays))
		("o,out", "Output histogram filename", cxxopts::value<std::string>(outHis_fname))
		("num_threads", "Number of threads to use", cxxopts::value<int>(numThreads))
		("num_subsets", "Number of subsets to use (Default: 1)", cxxopts::value<int>(numSubsets))
		("subset_id", "Subset to project (Default: 0)", cxxopts::value<int>(subsetId))
		("h,help", "Print help");
		/* clang-format on */

		auto result = options.parse(argc, argv);
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

		auto scanner = std::make_unique<Scanner>(scanner_fname);
		Globals::set_num_threads(numThreads);

		// Input file
		auto inputImage = std::make_unique<ImageOwned>(inputImage_fname);

		// Attenuation image
		std::unique_ptr<ImageParams> attImgParams = nullptr;
		std::unique_ptr<ImageOwned> attImg = nullptr;
		if (!attImg_fname.empty())
		{
			attImg = std::make_unique<ImageOwned>(attImg_fname);
		}

		// Image-space PSF
		if (!imageSpacePsf_fname.empty())
		{
			auto imageSpacePsf =
			    std::make_unique<OperatorPsf>(imageSpacePsf_fname);
			std::cout << "Applying Image-space PSF..." << std::endl;
			imageSpacePsf->applyA(inputImage.get(), inputImage.get());
			std::cout << "Done applying Image-space PSF" << std::endl;
		}

		// Create histo here
		auto his = std::make_unique<Histogram3DOwned>(*scanner);
		his->allocate();


		// Start forward projection
		auto binIter = his->getBinIter(numSubsets, subsetId);
		OperatorProjectorParams projParams(binIter.get(), *scanner, tofWidth_ps,
		                                   tofNumStd, projSpacePsf_fname,
		                                   numRays);

		auto projectorType = IO::getProjector(projector_name);

		Util::forwProject(*inputImage, *his, projParams, projectorType,
		                  attImg.get(), nullptr);

		if (convertToAcf)
		{
			his->operationOnEachBinParallel(
			    [&his](bin_t bin) -> float
			    {
				    return Util::getAttenuationCoefficientFactor(
				        his->getProjectionValue(bin));
			    });
		}

		std::cout << "Writing histogram to file..." << std::endl;
		his->writeToFile(outHis_fname);
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
