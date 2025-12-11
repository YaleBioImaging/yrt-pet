/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../ArgumentReader.hpp"
#include "../PluginOptionsHelper.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/scatter/ScatterEstimator.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/utils/Tools.hpp"

#include <cxxopts.hpp>
#include <iostream>

using namespace yrt;

int main(int argc, char** argv)
{
	try
	{
		io::ArgumentRegistry registry{};

		std::string coreGroup = "0. Core";
		std::string sssGroup = "1. Single Scatter Simulation";
		std::string tailFittingGroup = "2. Tail fitting";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
#if BUILD_CUDA
		registry.registerArgument("gpu", "Use GPU acceleration", false,
		                          io::TypeOfArgument::BOOL, false, coreGroup);
#endif
		registry.registerArgument(
		    "save_intermediary",
		    "Directory where to save intermediary histograms (leave "
		    "blank to not save any)",
		    false, io::TypeOfArgument::STRING, "", coreGroup);
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, io::TypeOfArgument::INT, -1,
		                          coreGroup);
		registry.registerArgument("seed", "Random number generator seed to use",
		                          false, io::TypeOfArgument::INT,
		                          scatter::ScatterEstimator::DefaultSeed,
		                          coreGroup);
		registry.registerArgument("out", "Output scatter estimate filename",
		                          true, io::TypeOfArgument::STRING, "",
		                          coreGroup, "o");

		registry.registerArgument("source", "Input source image", true,
		                          io::TypeOfArgument::STRING, "", sssGroup);
		registry.registerArgument("att", "Attenuation image file", true,
		                          io::TypeOfArgument::STRING, "", sssGroup);
		registry.registerArgument(
		    "att_threshold",
		    "Tail fitting attenuation threshold for the scatter tails mask "
		    "(Default: " +
		        std::to_string(scatter::ScatterEstimator::DefaultAttThreshold) +
		        ")",
		    false, io::TypeOfArgument::FLOAT,
		    scatter::ScatterEstimator::DefaultAttThreshold, tailFittingGroup);

		// By default, no TOF
		registry.registerArgument("n_tof",
		                          "Number of TOF bins to consider for SSS",
		                          false, io::TypeOfArgument::INT, 1, sssGroup);
		registry.registerArgument(
		    "n_planes", "Number of axial planes to consider for SSS", false,
		    io::TypeOfArgument::INT,
		    static_cast<int>(ScatterSpace::RecommendedNumPlanes), sssGroup);
		registry.registerArgument(
		    "n_angles", "Number of angles to consider for SSS", false,
		    io::TypeOfArgument::INT,
		    static_cast<int>(ScatterSpace::RecommendedNumAngles), sssGroup);

		registry.registerArgument(
		    "crystal_mat", "Crystal material name (default: LYSO)", false,
		    io::TypeOfArgument::STRING, "LYSO", sssGroup);

		registry.registerArgument(
		    "prompts",
		    "Prompts file (input listmode or histogram). Possible values: " +
		        io::possibleFormats(),
		    true, io::TypeOfArgument::STRING, "", tailFittingGroup);
		registry.registerArgument("prompts_format", "Prompts format", true,
		                          io::TypeOfArgument::STRING, "",
		                          tailFittingGroup);
		registry.registerArgument("randoms",
		                          "Randoms histogram file (optional). Will "
		                          "override randoms gathered from prompts",
		                          false, io::TypeOfArgument::STRING, "",
		                          tailFittingGroup);
		registry.registerArgument(
		    "randoms_format",
		    "Randoms histogram format. Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, io::TypeOfArgument::STRING, "", tailFittingGroup);
		registry.registerArgument(
		    "sensitivity", "Sensitivity histogram file (optional)", false,
		    io::TypeOfArgument::STRING, "", tailFittingGroup);
		registry.registerArgument(
		    "sensitivity_format",
		    "Sensitivity histogram format. Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYHISTOGRAMS),
		    false, io::TypeOfArgument::STRING, "", tailFittingGroup);
		registry.registerArgument(
		    "mask_width",
		    "Tail fitting mask width (Default: " +
		        std::to_string(
		            scatter::ScatterEstimator::DefaultScatterTailsMaskWidth) +
		        ")",
		    false, io::TypeOfArgument::INT,
		    static_cast<int>(
		        scatter::ScatterEstimator::DefaultScatterTailsMaskWidth),
		    tailFittingGroup);

		plugin::addOptionsFromPlugins(registry,
		                              plugin::InputFormatsChoice::ALL);

		// Load configuration
		io::ArgumentReader config{registry, "Scatter estimation executable"};

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

		bool useGPU = config.getValue<bool>("gpu");
		auto scanner_fname = config.getValue<std::string>("scanner");
		auto prompts_fname = config.getValue<std::string>("prompts");
		auto prompts_format = config.getValue<std::string>("prompts_format");
		auto randomsHis_fname = config.getValue<std::string>("randoms");
		auto randomsHis_format = config.getValue<std::string>("randoms_format");
		auto sensitivityHis_fname = config.getValue<std::string>("sensitivity");
		auto sensitivityHis_format =
		    config.getValue<std::string>("sensitivity_format");
		auto sourceImage_fname = config.getValue<std::string>("source");
		auto attImage_fname = config.getValue<std::string>("att");
		auto attThreshold = config.getValue<float>("att_threshold");
		auto crystalMaterial_name = config.getValue<std::string>("crystal_mat");
		size_t numTOFBins = config.getValue<int>("n_tof");
		size_t numPlanes = config.getValue<int>("n_planes");
		size_t numAngles = config.getValue<int>("n_angles");
		std::string scatterOut_fname = config.getValue<std::string>("out");
		std::string saveIntermediary_dir =
		    config.getValue<std::string>("save_intermediary");
		int numThreads = config.getValue<int>("num_threads");
		int maskWidth = config.getValue<int>("mask_width");
		int seed = config.getValue<int>("seed");

		if (useGPU)
		{
#if not BUILD_CUDA
			std::cerr << "YRT-PET needs to be built with CUDA "
			             "support in order to use GPU acceleration"
			          << std::endl;
			return -1;
#endif
		}

		globals::setNumThreads(numThreads);
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

		scatter::CrystalMaterial crystalMaterial =
		    scatter::getCrystalMaterialFromName(crystalMaterial_name);

		std::cout << "Reading prompts..." << std::endl;
		auto prompts = io::openProjectionData(
		    prompts_fname, prompts_format, *scanner, config.getAllArguments());

		Histogram* randomsHis = nullptr;
		std::unique_ptr<ProjectionData> randomsHisProjData = nullptr;
		if (!randomsHis_fname.empty())
		{
			std::cout << "Reading randoms histogram..." << std::endl;
			ASSERT_MSG(!io::isFormatListMode(randomsHis_format),
			           "Randoms histogram format has to a histogram format");
			randomsHisProjData =
			    io::openProjectionData(randomsHis_fname, randomsHis_format,
			                           *scanner, config.getAllArguments());

			randomsHis = dynamic_cast<Histogram*>(randomsHisProjData.get());
			ASSERT(randomsHis != nullptr);
		}

		Histogram* sensitivityHis = nullptr;
		std::unique_ptr<ProjectionData> sensitivityHisProjData = nullptr;
		if (!sensitivityHis_fname.empty())
		{
			std::cout << "Reading sensitivity histogram..." << std::endl;
			sensitivityHisProjData = io::openProjectionData(
			    sensitivityHis_fname, sensitivityHis_format, *scanner,
			    config.getAllArguments());

			sensitivityHis =
			    dynamic_cast<Histogram*>(sensitivityHisProjData.get());
			ASSERT(sensitivityHis != nullptr);
		}

		auto attImage = std::make_unique<ImageOwned>(attImage_fname);
		auto sourceImage = std::make_unique<ImageOwned>(sourceImage_fname);

		scatter::ScatterEstimator scatterEstimator(
		    *scanner, *sourceImage, *attImage, *prompts, numTOFBins, numPlanes,
		    numAngles, randomsHis, sensitivityHis, crystalMaterial, seed,
		    maskWidth, attThreshold, saveIntermediary_dir);

		scatterEstimator.allocate();

		scatterEstimator.computeTailFittedScatterEstimate();

		scatterEstimator.getScatterEstimate().writeToFile(scatterOut_fname);

		std::cout << "Done." << std::endl;
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