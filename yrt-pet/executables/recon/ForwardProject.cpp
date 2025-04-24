/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../ArgumentReader.hpp"

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
		IO::ArgumentRegistry registry{};

		std::string coreGroup = "0. Core";
		std::string inputGroup = "1. Input";
		std::string projectorGroup = "2. Projector";
		std::string outputGroup = "3. Output";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          IO::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
#if BUILD_CUDA
		registry.registerArgument("gpu", "Use GPU acceleration", false,
		                          IO::TypeOfArgument::BOOL, false, coreGroup);
#endif
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, IO::TypeOfArgument::INT, -1,
		                          coreGroup);

		registry.registerArgument("input", "Input image file", false,
		                          IO::TypeOfArgument::STRING, "", inputGroup,
		                          "i");
		registry.registerArgument("psf", "Image-space PSF kernel file", false,
		                          IO::TypeOfArgument::STRING, "", inputGroup);
		registry.registerArgument("num_subsets",
		                          "Number of OSEM subsets (Default: 1)", false,
		                          IO::TypeOfArgument::INT, 1, inputGroup);
		registry.registerArgument("subset_id",
		                          "Subset to backproject (Default: 0)", false,
		                          IO::TypeOfArgument::INT, 0, inputGroup);

		registry.registerArgument(
		    "projector",
		    "Projector to use, choices: Siddon (S), Distance-Driven (D). The "
		    "default projector is Siddon",
		    false, IO::TypeOfArgument::STRING, "S", projectorGroup);
		registry.registerArgument(
		    "proj_psf",
		    "Projection-space PSF kernel file (for DD projector only)", false,
		    IO::TypeOfArgument::STRING, "", projectorGroup);
		registry.registerArgument(
		    "num_rays", "Number of rays to use (for Siddon projector only)",
		    false, IO::TypeOfArgument::INT, 1, projectorGroup);

		registry.registerArgument("out", "Output histogram filename", false,
		                          IO::TypeOfArgument::STRING, "", outputGroup,
		                          "o");
		registry.registerArgument("to_acf", "Generate ACF histogram", false,
		                          IO::TypeOfArgument::BOOL, false, outputGroup);
		registry.registerArgument(
		    "sparse", "Forward project to a sparse histogram", false,
		    IO::TypeOfArgument::BOOL, false, outputGroup);

		// Load configuration
		IO::ArgumentReader config{
		    registry, "Forward project an image into a Histogram3D"};

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
		auto inputImage_fname = config.getValue<std::string>("input");
		auto imagePsf_fname = config.getValue<std::string>("psf");
		auto projPsf_fname = config.getValue<std::string>("proj_psf");
		auto outHis_fname = config.getValue<std::string>("out");
		auto projector_name = config.getValue<std::string>("projector");
		int numThreads = config.getValue<int>("num_threads");
		int numSubsets = config.getValue<int>("num_subsets");
		int subsetId = config.getValue<int>("subset_id");
		int numRays = config.getValue<int>("num_rays");
		bool useGPU = config.getValue<bool>("gpu");
		bool convertToAcf = config.getValue<bool>("to_acf");
		bool toSparseHistogram = config.getValue<bool>("sparse");

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
