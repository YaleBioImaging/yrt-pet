/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../ArgumentReader.hpp"
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
		IO::ArgumentRegistry registry{};

		std::string coreGroup = "0. Core";
		std::string sssGroup = "1. Single Scatter Simulation";
		std::string tailFittingGroup = "2. Tail fitting";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          IO::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
#if BUILD_CUDA
		registry.registerArgument(
		    "gpu", "Use GPU to compute the ACF histogram (if needed)", false,
		    IO::TypeOfArgument::BOOL, false, coreGroup);
#endif
		registry.registerArgument(
		    "save_intermediary",
		    "Directory where to save intermediary histograms (leave "
		    "blank to not save any)",
		    false, IO::TypeOfArgument::STRING, "", coreGroup);
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, IO::TypeOfArgument::INT, -1,
		                          coreGroup);
		registry.registerArgument("seed", "Random number generator seed to use",
		                          false, IO::TypeOfArgument::INT,
		                          Scatter::ScatterEstimator::DefaultSeed,
		                          coreGroup);
		registry.registerArgument(
		    "out", "Output scatter estimate histogram filename", true,
		    IO::TypeOfArgument::STRING, "", coreGroup, "o");
		registry.registerArgument(
		    "out_acf",
		    "Output ACF histogram filename (if it needs to be calculated from "
		    "the attenuation image)",
		    false, IO::TypeOfArgument::STRING, "", coreGroup);

		registry.registerArgument("att", "Attenuation image file", true,
		                          IO::TypeOfArgument::STRING, "", sssGroup);
		registry.registerArgument("source", "Input source image", true,
		                          IO::TypeOfArgument::STRING, "", sssGroup);
		registry.registerArgument("n_z",
		                          "Number of Z planes to consider for SSS",
		                          true, IO::TypeOfArgument::INT, -1, sssGroup);
		registry.registerArgument("n_phi",
		                          "Number of Phi angles to consider for SSS",
		                          true, IO::TypeOfArgument::INT, -1, sssGroup);
		registry.registerArgument("n_r",
		                          "Number of R distances to consider for SSS",
		                          true, IO::TypeOfArgument::INT, -1, sssGroup);
		registry.registerArgument(
		    "crystal_mat", "Crystal material name (default: LYSO)", false,
		    IO::TypeOfArgument::STRING, "LYSO", sssGroup);

		registry.registerArgument("prompts", "Prompts histogram file", true,
		                          IO::TypeOfArgument::STRING, "",
		                          tailFittingGroup);
		registry.registerArgument(
		    "randoms", "Randoms histogram file (optional)", false,
		    IO::TypeOfArgument::STRING, "", tailFittingGroup);
		registry.registerArgument(
		    "sensitivity", "Sensitivity histogram file (optional)", false,
		    IO::TypeOfArgument::STRING, "", tailFittingGroup);
		registry.registerArgument(
		    "invert_sensitivity",
		    "Invert the sensitivity histogram values (sensitivity -> "
		    "1/sensitivity)",
		    false, IO::TypeOfArgument::BOOL, false, tailFittingGroup);
		registry.registerArgument(
		    "acf",
		    "ACF histogram file (optional). Will be computed from "
		    "attenuation image if not provided",
		    false, IO::TypeOfArgument::STRING, "", tailFittingGroup);
		registry.registerArgument(
		    "acf_threshold",
		    "Tail fitting ACF threshold for the scatter tails mask (Default: " +
		        std::to_string(Scatter::ScatterEstimator::DefaultACFThreshold) +
		        ")",
		    false, IO::TypeOfArgument::FLOAT,
		    Scatter::ScatterEstimator::DefaultACFThreshold, tailFittingGroup);
		registry.registerArgument(
		    "mask_width",
		    "Tail fitting mask width. By default, uses 1/10th of "
		    "the histogram \'r\' dimension",
		    false, IO::TypeOfArgument::INT, -1, tailFittingGroup);
		registry.registerArgument(
		    "no_denorm",
		    "Do not affect the scatter estimate by the sensitivity", false,
		    IO::TypeOfArgument::BOOL, false, tailFittingGroup);

		// Load configuration
		IO::ArgumentReader config{registry, "Scatter estimation executable"};

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
		auto promptsHis_fname = config.getValue<std::string>("prompts");
		auto randomsHis_fname = config.getValue<std::string>("randoms");
		auto sensitivityHis_fname = config.getValue<std::string>("sensitivity");
		auto acfHis_fname = config.getValue<std::string>("acf");
		auto sourceImage_fname = config.getValue<std::string>("source");
		auto attImage_fname = config.getValue<std::string>("att");
		auto crystalMaterial_name = config.getValue<std::string>("crystal_mat");
		size_t nZ = config.getValue<int>("n_z");
		size_t nPhi = config.getValue<int>("n_phi");
		size_t nR = config.getValue<int>("n_r");
		std::string scatterOut_fname = config.getValue<std::string>("out");
		std::string acfOutHis_fname = config.getValue<std::string>("out_acf");
		std::string saveIntermediary_dir =
		    config.getValue<std::string>("save_intermediary");
		bool invertSensitivity = config.getValue<bool>("invert_sensitivity");
		int numThreads = config.getValue<int>("num_threads");
		int maskWidth = config.getValue<int>("mask_width");
		float acfThreshold = config.getValue<float>("acf_threshold");
		bool useGPU = config.getValue<bool>("gpu");
		int seed = config.getValue<int>("seed");
		bool denormalize = !config.getValue<bool>("no_denorm");

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
		std::cout << "Initializing scanner..." << std::endl;
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

			Util::forwProject(*scanner, *attImage, *acfHis,
			                  OperatorProjector::ProjectorType::SIDDON, useGPU);

			Util::convertProjectionValuesToACF(*acfHis);

			if (!acfOutHis_fname.empty())
			{
				acfHis->writeToFile(acfOutHis_fname);
			}
		}
		else
		{
			acfHis = std::make_unique<Histogram3DOwned>(*scanner, acfHis_fname);
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
		    scatterEstimator.computeTailFittedScatterEstimate(nZ, nPhi, nR,
		                                                      denormalize);

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