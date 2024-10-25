/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/IO.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/projection/UniformHistogram.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "geometry/Constants.hpp"
#include "scatter/ScatterEstimator.hpp"
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
		std::string normHis_fname;
		std::string sensHis_fname;
		std::string randomsHis_fname;
		std::string imgParams_fname;
		std::string acfHis_fname;
		std::string attImg_fname;
		std::string crystalMaterial_name = "LYSO";
		size_t nZ, nPhi, nR;
		std::string outSensImg_fname;
		std::string outSensDataHis_fname;
		std::string histoOut_fname;
		std::string projector_name = "S";
		std::string sourceImage_fname;
		std::string scatterHistoIn_fname;
		std::string outAcfHis_fname;
		int numThreads = -1;
		int num_OSEM_subsets = 1;
		int num_MLEM_iterations = 3;
		int maskWidth = -1;
		float acfThreshold = 0.9523809f;  // 1/1.05
		bool saveIntermediary = false;
		bool noScatter = false;

		// Parse command line arguments
		cxxopts::Options options(argv[0],
		                         "Single-Scatter-Simulation and Scatter "
		                         "Correction histogram generation");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
		("s,scanner", "Scanner parameters file", cxxopts::value(scanner_fname))
		("prompts", "Prompts histogram file", cxxopts::value(promptsHis_fname))
		("norm", "Normalisation histogram file", cxxopts::value(normHis_fname))
		("randoms", "Randoms histogram file", cxxopts::value(randomsHis_fname))
		("sens_his", "Sensitivity histogram file (To use instead of a normalisation histogram)", cxxopts::value(sensHis_fname))
		("acf", "Attenuation coefficients factor", cxxopts::value(acfHis_fname))
		("p,params", "Source image parameters file", cxxopts::value(imgParams_fname))
		("att", "Attenuation image file", cxxopts::value(attImg_fname))
		("no_scatter", "Skip scatter estimation", cxxopts::value(noScatter))
		("crystal_mat", "Crystal material name (default: LYSO)", cxxopts::value(crystalMaterial_name))
		("nZ", "Number of Z planes to consider for SSS", cxxopts::value(nZ))
		("nPhi", "Number of Phi angles to consider for SSS", cxxopts::value(nPhi))
		("nR", "Number of R distances to consider for SSS", cxxopts::value(nR))
		("o,out", "Additive histogram output filename", cxxopts::value(histoOut_fname))
		("out_sens", "Generated sensitivity image output filename", cxxopts::value(outSensImg_fname))
		("out_sensdata", "Generated sensitivity data histogram output filename", cxxopts::value(outSensDataHis_fname))
		("out_acf", "Output ACF histogram filename (if generated from Attenuation image)", cxxopts::value(outAcfHis_fname))
		("source", "Non scatter-corrected source image (if available)", cxxopts::value(sourceImage_fname))
		("num_threads", "Number of threads", cxxopts::value(numThreads))
		("scatter_his", "Previously generated scatter estimate histogram (if available)", cxxopts::value(scatterHistoIn_fname))
		("save_intermediary", "Enable saving intermediary histograms", cxxopts::value(saveIntermediary))
		("mask_width", "Tail fitting mask width. By default, uses 1/10th of the \'r\' dimension", cxxopts::value(maskWidth))
		("acf_threshold", "Tail fitting ACF threshold (Default: " + std::to_string(acfThreshold) + ")", cxxopts::value(acfThreshold))
		("num_subsets", "Number of subsets to use for the MLEM part (Default: 1)", cxxopts::value(num_OSEM_subsets))
		("num_iterations", "Number of MLEM iterations to do to generate source image (if needed) (Default: 3)", cxxopts::value(num_MLEM_iterations))
		("projector", "Projector to use, choices: Siddon (S), Distance-Driven (DD)"
			 #if BUILD_CUDA
			 ", or GPU Distance-Driven (DD_GPU)"
			 #endif
			 ". The default projector is Siddon", cxxopts::value<std::string>(projector_name))
		("h,help", "Print help");
		/* clang-format on */

		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		std::vector<std::string> required_params = {
		    "scanner", "params", "randoms", "prompts", "att", "out"};

		if (!noScatter)
		{
			for (auto& p : {"nZ", "nPhi", "nR"})
			{
				required_params.emplace_back(p);
			}
		}

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

		bool isNorm;
		const std::string* normOrSensHis_fname = nullptr;
		bool uniformNorm = false;
		if (!normHis_fname.empty())
		{
			isNorm = true;
			normOrSensHis_fname = &normHis_fname;
		}
		else if (!sensHis_fname.empty())
		{
			isNorm = false;
			normOrSensHis_fname = &sensHis_fname;
		}
		else
		{
			std::cout << "No normalization or sensitivity provided, using "
			             "uniform histogram as normalization."
			          << std::endl;
			isNorm = true;
			uniformNorm = true;
		}

		Globals::set_num_threads(numThreads);
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		// Check if scanner parameters have been set properly for Scatter
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

		ImageParams imageParams(imgParams_fname);

		Scatter::CrystalMaterial crystalMaterial =
		    Scatter::getCrystalMaterialFromName(crystalMaterial_name);

		// Attenuation image
		auto attImg = std::make_unique<ImageOwned>(attImg_fname);

		std::cout << "Reading histograms..." << std::endl;
		auto promptsHis =
		    std::make_unique<Histogram3DOwned>(*scanner, promptsHis_fname);
		auto randomsHis =
		    std::make_unique<Histogram3DOwned>(*scanner, randomsHis_fname);

		std::unique_ptr<Histogram3D> normOrSensHis = nullptr;
		if (!uniformNorm)
		{
			normOrSensHis = std::make_unique<Histogram3DOwned>(
			    *scanner, *normOrSensHis_fname);
		}
		else
		{
			normOrSensHis = std::make_unique<UniformHistogram>(*scanner, 1.0f);
		}

		std::cout << "Done reading histograms." << std::endl;
		std::unique_ptr<Histogram3DOwned> acfHis = nullptr;
		if (!acfHis_fname.empty())
		{
			acfHis = std::make_unique<Histogram3DOwned>(*scanner, acfHis_fname);
		}
		else
		{
			std::cout << "ACF histogram not specified, forward-projecting "
			             "attenuation image..."
			          << std::endl;
			acfHis = std::make_unique<Histogram3DOwned>(*scanner);
			acfHis->allocate();
			Util::forwProject(*scanner, *attImg, *acfHis,
			                  IO::getProjector(projector_name));
			acfHis->operationOnEachBinParallel(
			    [&acfHis](bin_t bin) -> float
			    {
				    return Util::getAttenuationCoefficientFactor(
				        acfHis->getProjectionValue(bin));
			    });
			if (!outAcfHis_fname.empty())
			{
				acfHis->writeToFile(outAcfHis_fname);
			}
		}

		// Generate additive histogram
		std::cout << "Preparing additive histogram..." << std::endl;
		auto additiveHis = std::make_unique<Histogram3DOwned>(*scanner);
		additiveHis->allocate();
		if (isNorm)
		{
			additiveHis->operationOnEachBinParallel(
			    [&randomsHis, &acfHis, &normOrSensHis](bin_t bin) -> float
			    {
				    return randomsHis->getProjectionValue(bin) *
				           normOrSensHis->getProjectionValue(bin) /
				           (acfHis->getProjectionValue(bin) + EPS_FLT);
			    });
		}
		else
		{
			additiveHis->operationOnEachBinParallel(
			    [&randomsHis, &acfHis, &normOrSensHis](bin_t bin) -> float
			    {
				    return randomsHis->getProjectionValue(bin) /
				           (normOrSensHis->getProjectionValue(bin) *
				                acfHis->getProjectionValue(bin) +
				            EPS_FLT);
			    });
		}

		if (saveIntermediary)
		{
			additiveHis->writeToFile(
			    "intermediary_firstAdditiveCorrection.his");
		}
		if (noScatter)
		{
			additiveHis->writeToFile(histoOut_fname);
			std::cout << "Done." << std::endl;
			return 0;
		}

		std::shared_ptr<Image> sourceImg = nullptr;
		if (sourceImage_fname.empty())
		{
			// Generate histogram to use for the sensitivity image generation
			std::cout << "Preparing sensitivity histogram for the MLEM part..."
			          << std::endl;
			auto sensDataHis = std::make_unique<Histogram3DOwned>(*scanner);
			sensDataHis->allocate();
			if (isNorm)
			{
				sensDataHis->operationOnEachBinParallel(
				    [&acfHis, &normOrSensHis](bin_t bin) -> float
				    {
					    return acfHis->getProjectionValue(bin) /
					           (normOrSensHis->getProjectionValue(bin) +
					            EPS_FLT);
				    });
			}
			else
			{
				sensDataHis->operationOnEachBinParallel(
				    [&acfHis, &normOrSensHis](bin_t bin) -> float
				    {
					    return acfHis->getProjectionValue(bin) *
					           normOrSensHis->getProjectionValue(bin);
				    });
			}

			if (saveIntermediary)
			{
				sensDataHis->writeToFile("intermediary_sensData.his");
			}
			if (!outSensDataHis_fname.empty())
			{
				sensDataHis->writeToFile(outSensDataHis_fname);
			}

			std::vector<std::unique_ptr<Image>> sensImages;

			// Generate source Image
			auto projectorType = IO::getProjector(projector_name);
			auto osem =
			    Util::createOSEM(*scanner, IO::requiresGPU(projectorType));
			osem->num_MLEM_iterations = num_MLEM_iterations;
			osem->addHis = additiveHis.get();
			osem->setImageParams(imageParams);
			osem->num_OSEM_subsets = num_OSEM_subsets;
			osem->projectorType = projectorType;
			osem->setSensDataInput(sensDataHis.get());
			osem->setDataInput(promptsHis.get());
			osem->generateSensitivityImages(sensImages, outSensImg_fname);
			osem->setSensitivityImages(sensImages);
			sourceImg = osem->reconstruct(
			    saveIntermediary ? "intermediary_sourceImage.nii" : "");

			// Deallocate Sensitivity data histogram to save memory
			sensDataHis = nullptr;
		}
		else
		{
			if (!outSensImg_fname.empty())
			{
				std::cerr
				    << "Warning: The sensitivity image will not be generated "
				       "since a source image was already provided"
				    << std::endl;
			}
			sourceImg =
			    std::make_unique<ImageOwned>(imageParams, sourceImage_fname);
		}

		Scatter::ScatterEstimator sss(*scanner, *sourceImg, *attImg,
		                              promptsHis.get(), normOrSensHis.get(),
		                              randomsHis.get(), acfHis.get(),
		                              crystalMaterial, 13, isNorm, maskWidth,
		                              acfThreshold, saveIntermediary);

		if (!scatterHistoIn_fname.empty())
		{
			auto scatterHis = std::make_shared<Histogram3DOwned>(
			    *scanner, scatterHistoIn_fname);
			sss.setScatterHistogram(scatterHis);
		}

		sss.computeAdditiveScatterCorrection(nZ, nPhi, nR);

		std::cout << "Preparing final additive correction histogram..."
		          << std::endl;
		const Histogram3D* scatterHis = sss.getScatterHistogram();
		additiveHis->operationOnEachBinParallel(
		    [&additiveHis, &scatterHis](const bin_t bin) -> float
		    {
			    return additiveHis->getProjectionValue(bin) +
			           scatterHis->getProjectionValue(bin);
		    });

		std::cout << "Saving histogram file..." << std::endl;
		additiveHis->writeToFile(histoOut_fname);

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
		std::cerr << "Caught exception: " << e.what() << std::endl;
		return -1;
	}
}
