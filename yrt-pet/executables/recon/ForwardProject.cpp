/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/IO.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/projection/SparseHistogram.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "operators/OperatorProjector.hpp"
#include "operators/OperatorProjectorDD.hpp"
#include "operators/OperatorProjectorSiddon.hpp"
#include "operators/SparseProjection.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ReconstructionUtils.hpp"

#include <cxxopts.hpp>
#include <iostream>

int main(int argc, char** argv)
{
	try
	{
		std::string scanner_fname;
		std::string inputImage_fname;
		std::string imagePsf_fname;
		std::string projPsf_fname;
		std::string outHis_fname;
		std::string projector_name = "S";
		int numThreads = -1;
		int numSubsets = 1;
		int subsetId = 0;
		int numRays = 1;
		bool useGPU = false;
		bool convertToAcf = false;
		bool toSparseHistogram = false;

		// Parse command line arguments
		cxxopts::Options options(argv[0],
		                         "Forward "
		                         "project an image into a Histogram3D");
		options.positional_help("[optional args]").show_positional_help();
		/* clang-format off */
		options.add_options()
		("s,scanner", "Scanner parameters file", cxxopts::value<std::string>(scanner_fname))
		("i,input", "Input image file", cxxopts::value<std::string>(inputImage_fname))
		("psf", "Image-space PSF kernel file", cxxopts::value<std::string>(imagePsf_fname))
		("proj_psf", "Projection-space PSF kernel file", cxxopts::value<std::string>(projPsf_fname))
		("projector", "Projector to use, choices: Siddon (S), Distance-Driven (D)."
			"The default projector is Siddon", cxxopts::value<std::string>(projector_name))
		("to_acf", "Generate ACF histogram", cxxopts::value<bool>(convertToAcf))
		("num_rays", "Number of rays to use in the Siddon projector", cxxopts::value<int>(numRays))
		("o,out", "Output histogram filename", cxxopts::value<std::string>(outHis_fname))
		("sparse", "Forward project to a sparse histogram", cxxopts::value<bool>(toSparseHistogram))
		("gpu", "Use GPU acceleration", cxxopts::value<bool>(useGPU))
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

		// Image-space PSF
		if (!imagePsf_fname.empty())
		{
			auto imagePsf = std::make_unique<OperatorPsf>(imagePsf_fname);
			std::cout << "Applying Image-space PSF..." << std::endl;
			imagePsf->applyA(inputImage.get(), inputImage.get());
		}

		auto projectorType = IO::getProjector(projector_name);

		if (!toSparseHistogram)
		{
			auto his = std::make_unique<Histogram3DOwned>(*scanner);
			his->allocate();

			// Setup forward projection
			auto binIter = his->getBinIter(numSubsets, subsetId);
			OperatorProjectorParams projParams(binIter.get(), *scanner, 0, 0,
			                                   projPsf_fname, numRays);

			Util::forwProject(*inputImage, *his, projParams, projectorType,
			                  useGPU);

			if (convertToAcf)
			{
				std::cout << "Computing attenuation coefficient factors..."
				          << std::endl;
				Util::convertProjectionValuesToACF(*his);
			}

			std::cout << "Writing histogram to file..." << std::endl;
			his->writeToFile(outHis_fname);
		}
		else
		{
			ASSERT_MSG(!useGPU,
			           "Forward projection to sparse histogram is currently "
			           "not supported on GPU");

			ASSERT_MSG(numSubsets == 1 && subsetId == 0,
			           "Forward projection to sparse histogram is currently "
			           "not supported for multiple subsets");

			std::unique_ptr<OperatorProjector> projector;
			if (projectorType == OperatorProjector::ProjectorType::SIDDON)
			{
				projector = std::make_unique<OperatorProjectorSiddon>(*scanner,
				                                                      numRays);
			}
			else
			{
				projector = std::make_unique<OperatorProjectorDD>(
				    *scanner, 0, -1, projPsf_fname);
			}

			const ImageParams& params = inputImage->getParams();
			auto sparseHistogram = std::make_unique<SparseHistogram>(*scanner);

			sparseHistogram->allocate(params.nx * params.ny);

			Util::forwProjectToSparseHistogram(*inputImage, *projector,
			                                   *sparseHistogram);

			if (convertToAcf)
			{
				std::cout << "Computing attenuation coefficient factors..."
				          << std::endl;
				Util::convertProjectionValuesToACF(*sparseHistogram);
			}

			sparseHistogram->writeToFile(outHis_fname);
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
