/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

/*******************************************************************************
 * Def.: Post-reconstruction correction of motion by warping images obtained by
 *       motion-divided independent reconstruction of a PET acquisition to a
 *		 frame of reference.
 ******************************************************************************/

#include "datastruct/image/Image.hpp"
#include "datastruct/projection/LORMotion.hpp"
#include "utils/Assert.hpp"

#include <cxxopts.hpp>

#include <fstream>
#include <string>
#include <vector>


int main(int argc, char* argv[])
{
	try
	{
		std::vector<std::string> images_fname;
		std::string lorMotion_fname;
		std::string out_fname;

		// Parse command line arguments
		cxxopts::Options options(
		    argv[0],
		    "Post-reconstruction motion correction "
		    "executable (alternative to event-by-event motion correction)");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
		("i,input",
			"Paths to each image (separated by commas). "
			"Specify one image per frame",
			cxxopts::value(images_fname))
		("lor_motion",
			"LOR motion file for motion correction",
			cxxopts::value(lorMotion_fname))
		("o,out",
			"Output image filename",
			cxxopts::value(out_fname))
		("help", "Print help");
		/* clang-format on */

		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		std::vector<std::string> required_params = {"input", "lor_motion",
		                                            "out"};
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

		// Load the LOR Motion file
		auto lorMotion = std::make_unique<LORMotion>(lorMotion_fname);
		size_t numFrames = lorMotion->getNumFrames();
		float totalDuration = lorMotion->getTotalDuration();

		ImageParams imageParams;
		std::unique_ptr<ImageOwned> finalImage;

		size_t numImages = images_fname.size();
		ASSERT_MSG(
		    numImages == numFrames,
		    "The number of images provided must match the number of frames "
		    "defined in the LOR Motion file (One image per frame).");

		for (frame_t i = 0; i < static_cast<frame_t>(numImages); i++)
		{
			std::cout << "Reading image for frame " << i << std::endl;
			auto currentImage = std::make_unique<ImageOwned>(images_fname[i]);

			ImageParams currentParams = currentImage->getParams();
			if (i == 0)
			{
				// Image parameters not set yet. Use the first image to get
				//  the parameters
				imageParams = currentParams;
				// Use the parameters to initialize the final image
				finalImage = std::make_unique<ImageOwned>(imageParams);
				finalImage->allocate();
			}
			else
			{
				ASSERT_MSG(imageParams.isSameAs(currentParams),
				           "Image parameters mismatch. Not all images provided "
				           "have the same properties");
			}
			ASSERT(finalImage != nullptr);

			transform_t currentTransform = lorMotion->getTransform(i);
			float currentFrameWeight =
			    lorMotion->getDuration(i) / totalDuration;

			// Transform and weight the image
			currentImage = currentImage->transformImage(currentTransform);
			currentImage->multWithScalar(currentFrameWeight);

			// Accumulate image
			currentImage->addFirstImageToSecond(finalImage.get());
		}

		finalImage->writeToFile(out_fname);

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
