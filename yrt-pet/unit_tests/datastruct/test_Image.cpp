/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/geometry/TransformUtils.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#if BUILD_CUDA
#include "yrt-pet/utils/ReconstructionUtilsDevice.cuh"
#endif

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
void checkImageAllPositive(const Image& img)
{
	const ImageParams& params = img.getParams();

	const float* i_ptr = img.getRawPointer();
	const int numVoxels = params.nx * params.ny * params.nz;
	for (int i = 0; i < numVoxels; i++)
	{
		CHECK(i_ptr[i] >= 0);
	}
}
}  // namespace yrt

#if BUILD_CUDA

// This test validates that the image time-averaging is consistent between the
//  CPU and the GPU implementations
TEST_CASE("image-timeavg", "[image]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<int> numVoxelsDistribution(1, 300);
	std::uniform_real_distribution<float> voxelSizeDistribution(0.1f, 5.0f);

	std::uniform_int_distribution<size_t> numframesDistribution(10, 300);

	std::uniform_real_distribution<float> rollDistribution(-yrt::PI_FLT,
	                                                       yrt::PI_FLT);
	std::uniform_real_distribution<float> pitchDistribution(-yrt::PIHALF_FLT,
	                                                        yrt::PIHALF_FLT);
	std::uniform_real_distribution<float> yawDistribution(-yrt::PI_FLT,
	                                                      yrt::PI_FLT);
	std::uniform_real_distribution<float> translationDistribution(0.1f, 5.0f);
	std::uniform_int_distribution<yrt::timestamp_t> frameDurationDistribution(
	    5, 100);  // in ms

	constexpr int NumTrials = 5;

	for (int trial = 0; trial < NumTrials; trial++)
	{
		const int nxy = numVoxelsDistribution(engine);
		const int nz = numVoxelsDistribution(engine);
		const float vx = voxelSizeDistribution(engine);
		const float vy = voxelSizeDistribution(engine);
		const float vz = voxelSizeDistribution(engine);
		const float length_x = vx * nxy;
		const float length_y = vy * nxy;
		const float length_z = vz * nz;

		yrt::ImageParams params(nxy, nxy, nz, length_x, length_y, length_z);

		auto inputImage = yrt::util::test::makeImageWithRandomPrism(params);

		// Create transformations
		const size_t numframes = numframesDistribution(engine);
		yrt::LORMotion lorMotion(numframes);
		yrt::timestamp_t currentTimeStamp = 0;

		for (size_t frame_i = 0; frame_i < numframes; frame_i++)
		{
			// Set transformation
			yrt::Vector3D rotation{rollDistribution(engine),
			                       pitchDistribution(engine),
			                       yawDistribution(engine)};
			yrt::Vector3D translation{translationDistribution(engine),
			                          translationDistribution(engine),
			                          translationDistribution(engine)};
			yrt::transform_t transform =
			    yrt::util::fromRotationAndTranslationVectors(rotation,
			                                                 translation);
			lorMotion.setTransform(frame_i, transform);

			// Set timestamp
			lorMotion.setStartingTimestamp(frame_i, currentTimeStamp);
			currentTimeStamp += frameDurationDistribution(engine);
		}

		// Time-average using CPU
		auto outImage_cpu =
		    yrt::util::timeAverageMoveImage<false>(lorMotion, inputImage.get());

		// Time-average using GPU
		auto outImageDevice_gpu = yrt::util::timeAverageMoveImageDevice(
		    lorMotion, inputImage.get(), {nullptr, true});
		auto outImage_gpu = std::make_unique<yrt::ImageOwned>(params);
		outImage_gpu->allocate();
		outImageDevice_gpu->transferToHostMemory(outImage_gpu.get());

		yrt::checkImageAllPositive(*outImage_gpu);
		yrt::checkImageAllPositive(*outImage_cpu);

		CHECK(yrt::util::test::allclose(*outImage_cpu, *outImage_gpu, 0.001,
		                                0.001));
	}
}

#endif

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

	REQUIRE(img1.voxelSum() == static_cast<float>(sum);

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
