/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/IO.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "geometry/Constants.hpp"
#include "scatter/ScatterEstimator.hpp"
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
		std::string promptsHis_fname;
		std::string randomsHis_fname;
		std::string sensitivityHis_fname;
		std::string acfHis_fname;
		std::string sourceImage_fname;
		std::string attImage_fname;
		std::string crystalMaterial_name = "LYSO";
		size_t nZ, nPhi, nR;
		std::string scatterOut_fname;
		std::string acfOutHis_fname;  // In case ACF needs to be calculated
		std::string saveIntermediary_dir;
		bool invertSensitivity = false;
		int numThreads = -1;
		int maskWidth = -1;
		float acfThreshold = Scatter::ScatterEstimator::DefaultACFThreshold;
		bool useGPU = false;
		int seed = Scatter::ScatterEstimator::DefaultSeed;

		// Parse command line arguments
		cxxopts::Options options(argv[0], "Scatter estimation executable");
		options.positional_help("[optional args]").show_positional_help();

		auto coreGroup = options.add_options("0. Core");
		coreGroup("s,scanner", "Scanner parameters file",
		          cxxopts::value(scanner_fname));
		coreGroup("save_intermediary",
		          "Directory where to save intermediary histograms (leave "
		          "blank to not save any)",
		          cxxopts::value(saveIntermediary_dir));
		coreGroup("num_threads", "Number of threads to use",
		          cxxopts::value(numThreads));
		coreGroup("seed", "Random number generator seed to use",
		          cxxopts::value(seed));
		coreGroup("o,out", "Output scatter estimate histogram filename",
		          cxxopts::value(scatterOut_fname));

		auto sssGroup = options.add_options("1. Single Scatter Simulation");
		sssGroup("att", "Attenuation image file",
		         cxxopts::value(attImage_fname));
		sssGroup("source", "Input source image",
		         cxxopts::value(sourceImage_fname));
		sssGroup("nZ", "Number of Z planes to consider for SSS",
		         cxxopts::value(nZ));
		sssGroup("nPhi", "Number of Phi angles to consider for SSS",
		         cxxopts::value(nPhi));
		sssGroup("nR", "Number of R distances to consider for SSS",
		         cxxopts::value(nR));
		sssGroup("crystal_mat", "Crystal material name (default: LYSO)",
		         cxxopts::value(crystalMaterial_name));

		auto tailFittingGroup = options.add_options("2. Tail fitting");
		tailFittingGroup("prompts", "Prompts histogram file",
		                 cxxopts::value(promptsHis_fname));
		tailFittingGroup("randoms", "Randoms histogram file (optional)",
		                 cxxopts::value(randomsHis_fname));
		tailFittingGroup("sensitivity", "Sensitivity histogram file (optional)",
		                 cxxopts::value(sensitivityHis_fname));
		tailFittingGroup(
		    "invert_sensitivity",
		    "Invert the sensitivity histogram values (sensitivity -> "
		    "1/sensitivity)",
		    cxxopts::value(invertSensitivity));
		tailFittingGroup("acf",
		                 "ACF histogram file (optional). Will be computed from "
		                 "attenuation image if not provided",
		                 cxxopts::value(acfHis_fname));
		tailFittingGroup("gpu",
		                 "Use GPU to compute the ACF histogram (if needed)",
		                 cxxopts::value(useGPU));
		tailFittingGroup(
		    "acf_threshold",
		    "Tail fitting ACF threshold for the scatter tails mask (Default: " +
		        std::to_string(acfThreshold) + ")",
		    cxxopts::value(acfThreshold));
		tailFittingGroup("mask_width",
		                 "Tail fitting mask width. By default, uses 1/10th of "
		                 "the histogram \'r\' dimension",
		                 cxxopts::value(maskWidth));

		options.add_options()("h,help", "Print help");

		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		std::vector<std::string> required_params = {
		    "scanner", "prompts", "source", "att", "out", "nZ", "nPhi", "nR"};

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

		if (useGPU)
		{
#if not BUILD_CUDA
			std::cerr << "YRT-PET needs to be built with CUDA "
			             "support in order to use GPU acceleration"
			          << std::endl;
			return -1;
#endif
		}

		Globals::set_num_threads(numThreads);
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		// Check if scanner parameters have been set properly for scatter
		// estimation
		if (scanner->collimatorRadius < 0.0f || scanner->fwhm < 0.0f ||
		    scanner->energyLLD < 0.0f)
		{
			std::cerr
			    << "The scanner parameters given need to have a value for "
			       "\'collimatorRadius\',\'fwhm\', and \'energyLLD\'."
			    << std::endl;
			return -1;
		}

		Scatter::CrystalMaterial crystalMaterial =
		    Scatter::getCrystalMaterialFromName(crystalMaterial_name);

		std::cout << "Reading prompts histogram..." << std::endl;
		auto promptsHis =
		    std::make_unique<Histogram3DOwned>(*scanner, promptsHis_fname);
		std::unique_ptr<Histogram3DOwned> randomsHis = nullptr;
		if (!randomsHis_fname.empty())
		{
			std::cout << "Reading randoms histogram..." << std::endl;
			randomsHis =
			    std::make_unique<Histogram3DOwned>(*scanner, randomsHis_fname);
		}
		std::unique_ptr<Histogram3DOwned> sensitivityHis = nullptr;
		if (!sensitivityHis_fname.empty())
		{
			std::cout << "Reading sensitivity histogram..." << std::endl;
			sensitivityHis = std::make_unique<Histogram3DOwned>(
			    *scanner, sensitivityHis_fname);
			if (invertSensitivity)
			{
				sensitivityHis->operationOnEachBinParallel(
				    [&sensitivityHis](bin_t bin)
				    {
					    const float sensitivity =
					        sensitivityHis->getProjectionValue(bin);
					    if (sensitivity > 1e-8)
					    {
						    return 1.0f / sensitivity;
					    }
					    return 0.0f;
				    });
			}
		}

		auto attImage = std::make_unique<ImageOwned>(attImage_fname);

		std::unique_ptr<Histogram3DOwned> acfHis = nullptr;
		if (acfHis_fname.empty())
		{
			std::cout << "ACF histogram not specified. Forward projecting "
			             "attenuation image..."
			          << std::endl;
			acfHis = std::make_unique<Histogram3DOwned>(*scanner);
			acfHis->allocate();

			auto projector = useGPU ? OperatorProjector::ProjectorType::DD_GPU :
			                          OperatorProjector::ProjectorType::SIDDON;

			Util::forwProject(*scanner, *attImage, *acfHis, projector);

			if (!acfOutHis_fname.empty())
			{
				acfHis->writeToFile(acfOutHis_fname);
			}
		}

		auto sourceImage = std::make_unique<ImageOwned>(sourceImage_fname);

		Scatter::ScatterEstimator scatterEstimator{*scanner,
		                                           *sourceImage,
		                                           *attImage,
		                                           promptsHis.get(),
		                                           randomsHis.get(),
		                                           acfHis.get(),
		                                           sensitivityHis.get(),
		                                           crystalMaterial,
		                                           seed,
		                                           maskWidth,
		                                           acfThreshold,
		                                           saveIntermediary_dir};

		auto scatterEstimate =
		    scatterEstimator.computeTailFittedScatterEstimate(nZ, nPhi, nR);

		scatterEstimate->writeToFile(scatterOut_fname);
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << std::endl;
		return -1;
	}
}