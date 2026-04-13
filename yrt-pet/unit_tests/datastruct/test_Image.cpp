/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageBase.hpp"

#include <ctime>
#include <random>

namespace yrt
{
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
}  // namespace yrt

TEST_CASE("imageparams-fromParams", "[image]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<int> numVoxelsDistribution(1, 300);
	std::uniform_real_distribution<float> voxelSizeDistribution(0.1f, 5.0f);
	std::uniform_real_distribution<float> offsetDistribution(0.1f, 5.0f);

	constexpr int NumTrials = 5;

	for (int trial = 0; trial < NumTrials; trial++)
	{
		const int nx = numVoxelsDistribution(engine);
		const int ny = numVoxelsDistribution(engine);
		const int nz = numVoxelsDistribution(engine);
		const float vx = voxelSizeDistribution(engine);
		const float vy = voxelSizeDistribution(engine);
		const float vz = voxelSizeDistribution(engine);
		const float length_x = vx * nx;
		const float length_y = vy * ny;
		const float length_z = vz * nz;
		const float off_x = offsetDistribution(engine);
		const float off_y = offsetDistribution(engine);
		const float off_z = offsetDistribution(engine);

		yrt::ImageParams params(nx, ny, nz, length_x, length_y, length_z, off_x,
		                        off_y, off_z);
		// Convert offset to origin
		yrt::Vector3D origin = params.indexToPosition(0, 0, 0);

		auto paramsTest = yrt::ImageParams::fromParams(
		    params.nx, params.ny, params.nz, params.vx, params.vy, params.vz,
		    origin.x, origin.y, origin.z);

		CHECK(params.isSameAs(paramsTest));
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

	yrt::ImageParams params1{nx,       ny,    nz,    length_x, length_y,
	                         length_z, off_x, off_y, off_z};
	yrt::ImageOwned img1{params1};
	img1.allocate();

	// Fill the image with random values
	float* imgData_ptr = img1.getRawPointer();
	int numVoxels = nx * ny * nz;
	double sum = 0.0;
	for (int i = 0; i < numVoxels; i++)
	{
		imgData_ptr[i] = imageDataDistribution(engine);
		sum += imgData_ptr[i];
	}

	REQUIRE(img1.voxelSum() == static_cast<float>(sum));

	img1.writeToFile(tmpImage_fname);
	img1.writeToFile(tmpCompressedImage_fname);

	yrt::ImageOwned img2{tmpImage_fname};
	yrt::ImageParams params2 = img2.getParams();
	REQUIRE(params2.isSameAs(params1));

	checkTwoImages(img1, img2);

	params1.serialize(tmpParams_fname);
	yrt::ImageParams params3{tmpParams_fname};
	REQUIRE(params1.isSameAs(params3));

	yrt::ImageOwned img3{params3, tmpImage_fname};
	REQUIRE(params1.isSameAs(img3.getParams()));
	checkTwoImages(img1, img3);

	yrt::ImageOwned img4{tmpCompressedImage_fname};
	REQUIRE(params1.isSameAs(img4.getParams()));
	checkTwoImages(img1, img4);
	REQUIRE(std::filesystem::file_size(tmpCompressedImage_fname) <
	        std::filesystem::file_size(tmpImage_fname));

	// Clear temporary files from disk
	std::remove(tmpImage_fname.c_str());
	std::remove(tmpParams_fname.c_str());
	std::remove(tmpCompressedImage_fname.c_str());
}
