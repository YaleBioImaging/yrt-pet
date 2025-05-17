/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "datastruct/image/Image.hpp"

#include <ctime>
#include <random>

void checkTwoImages(const Image& img1, const Image& img2)
{
	const ImageParams& params1 = img1.getParams();
	const ImageParams& params2 = img2.getParams();
	REQUIRE(params1.isSameAs(params2));

	const float* i1_ptr = img1.getRawPointer();
	const float* i2_ptr = img2.getRawPointer();
	const int numVoxels = params1.nx * params1.ny * params1.nz;
	for (int i = 0; i < numVoxels; i++)
	{
		CHECK(i1_ptr[i] == Approx(i2_ptr[i]));
	}
}

TEST_CASE("image-readwrite", "[image]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<int> imageSizeDistribution(25, 75);
	std::uniform_real_distribution<float> imageLengthDistribution(25.0f, 75.0f);
	std::uniform_real_distribution<float> imageDataDistribution(0.0f, 1.0f);
	std::uniform_real_distribution<float> imageOffsetDistribution(-10.0f,
	                                                              10.0f);

	std::string tmpImage_fname = "tmp.nii";
	std::string tmpCompressedImage_fname = "tmp.nii.gz";
	std::string tmpParams_fname = "tmp_params.json";

	int nx = imageSizeDistribution(engine);
	int ny = imageSizeDistribution(engine);
	int nz = imageSizeDistribution(engine);
	float length_x = imageLengthDistribution(engine);
	float length_y = imageLengthDistribution(engine);
	float length_z = imageLengthDistribution(engine);
	float off_x = imageOffsetDistribution(engine);
	float off_y = imageOffsetDistribution(engine);
	float off_z = imageOffsetDistribution(engine);

	ImageParams params1{nx,       ny,    nz,    length_x, length_y,
	                    length_z, off_x, off_y, off_z};
	ImageOwned img1{params1};
	img1.allocate();

	// Fill the image with random values
	float* imgData_ptr = img1.getRawPointer();
	int numVoxels = nx * ny * nz;
	for (int i = 0; i < numVoxels; i++)
	{
		imgData_ptr[i] = imageDataDistribution(engine);
	}

	img1.writeToFile(tmpImage_fname);
	img1.writeToFile(tmpCompressedImage_fname);

	ImageOwned img2{tmpImage_fname};
	ImageParams params2 = img2.getParams();
	REQUIRE(params2.isSameAs(params1));

	checkTwoImages(img1, img2);

	params1.serialize(tmpParams_fname);
	ImageParams params3{tmpParams_fname};
	REQUIRE(params1.isSameAs(params3));

	ImageOwned img3{params3, tmpImage_fname};
	REQUIRE(params1.isSameAs(img3.getParams()));
	checkTwoImages(img1, img3);

	ImageOwned img4{tmpCompressedImage_fname};
	REQUIRE(params1.isSameAs(img4.getParams()));
	checkTwoImages(img1, img4);
	REQUIRE(std::filesystem::file_size(tmpCompressedImage_fname) <
	        std::filesystem::file_size(tmpImage_fname));

	// Clear temporary files from disk
	std::remove(tmpImage_fname.c_str());
	std::remove(tmpParams_fname.c_str());
	std::remove(tmpCompressedImage_fname.c_str());
}
