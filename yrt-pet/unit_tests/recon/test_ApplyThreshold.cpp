/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#if BUILD_CUDA

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageDevice.cuh"

#include <memory>

TEST_CASE("applythreshold", "[applythreshold]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<int> numVoxelsDistribution(10, 30);
	std::uniform_int_distribution<int> numFramesDistribution(1, 5);
	std::uniform_real_distribution<float> imageLengthDistribution(10.0f, 20.0f);
	std::uniform_real_distribution<float> thresholdDistribution(0.1f, 0.9f);
	std::uniform_real_distribution<float> valScaleDistribution(0.5f, 2.0f);
	std::uniform_real_distribution<float> valOffDistribution(-1.0f, 1.0f);

	constexpr int NumTrials = 3;

	for (int trial = 0; trial < NumTrials; trial++)
	{
		const ssize_t nx = numVoxelsDistribution(engine);
		const ssize_t ny = numVoxelsDistribution(engine);
		const ssize_t nz = numVoxelsDistribution(engine);
		const ssize_t nt = numFramesDistribution(engine);
		const float length_x = imageLengthDistribution(engine);
		const float length_y = imageLengthDistribution(engine);
		const float length_z = imageLengthDistribution(engine);

		yrt::ImageParams params(nx, ny, nz, length_x, length_y, length_z,
		                       0.0f, 0.0f, 0.0f, nt);

		auto mainImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);
		auto maskImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);

		auto mainImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(mainImage.get());
		auto maskImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(maskImage.get());

		const float threshold = thresholdDistribution(engine);
		const float val_le_scale = valScaleDistribution(engine);
		const float val_le_off = valOffDistribution(engine);
		const float val_gt_scale = valScaleDistribution(engine);
		const float val_gt_off = valOffDistribution(engine);

		mainImage->applyThreshold(maskImage.get(), threshold, val_le_scale,
		                         val_le_off, val_gt_scale, val_gt_off);

		mainImageDevice->applyThresholdDevice(
		    maskImageDevice.get(), threshold, val_le_scale, val_le_off,
		    val_gt_scale, val_gt_off, true);

		auto mainImage_gpu =
		    std::make_unique<yrt::ImageOwned>(mainImage->getParams());
		mainImage_gpu->allocate();
		mainImageDevice->transferToHostMemory(mainImage_gpu.get());

		REQUIRE(yrt::util::test::allclose(*mainImage, *mainImage_gpu, 0.001,
		                                0.001));
	}
}

TEST_CASE("applythreshold-broadcast", "[applythreshold]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<int> numVoxelsDistribution(10, 30);
	std::uniform_int_distribution<int> numFramesDistribution(3, 8);
	std::uniform_real_distribution<float> imageLengthDistribution(10.0f, 20.0f);
	std::uniform_real_distribution<float> thresholdDistribution(0.1f, 0.9f);
	std::uniform_real_distribution<float> valScaleDistribution(0.5f, 2.0f);
	std::uniform_real_distribution<float> valOffDistribution(-1.0f, 1.0f);

	constexpr int NumTrials = 3;

	for (int trial = 0; trial < NumTrials; trial++)
	{
		const ssize_t nx = numVoxelsDistribution(engine);
		const ssize_t ny = numVoxelsDistribution(engine);
		const ssize_t nz = numVoxelsDistribution(engine);
		const ssize_t nt = numFramesDistribution(engine);
		const float length_x = imageLengthDistribution(engine);
		const float length_y = imageLengthDistribution(engine);
		const float length_z = imageLengthDistribution(engine);

		yrt::ImageParams params(nx, ny, nz, length_x, length_y, length_z,
		                       0.0f, 0.0f, 0.0f, nt);

		// Mask image is 3D (single frame)
		yrt::ImageParams maskParams(nx, ny, nz, length_x, length_y,
		                            length_z);
		auto mainImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);
		auto maskImage =
		    yrt::util::test::makeImageWithRandomPrism(maskParams, &engine);

		auto mainImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(mainImage.get());
		auto maskImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(maskImage.get());

		const float threshold = thresholdDistribution(engine);
		const float val_le_scale = valScaleDistribution(engine);
		const float val_le_off = valOffDistribution(engine);
		const float val_gt_scale = valScaleDistribution(engine);
		const float val_gt_off = valOffDistribution(engine);

		mainImage->applyThresholdBroadcast(maskImage.get(), threshold,
		                                 val_le_scale, val_le_off,
		                                 val_gt_scale, val_gt_off);

		mainImageDevice->applyThresholdBroadcastDevice(
		    maskImageDevice.get(), threshold, val_le_scale, val_le_off,
		    val_gt_scale, val_gt_off, true);

		auto mainImage_gpu =
		    std::make_unique<yrt::ImageOwned>(mainImage->getParams());
		mainImage_gpu->allocate();
		mainImageDevice->transferToHostMemory(mainImage_gpu.get());

		REQUIRE(yrt::util::test::allclose(*mainImage, *mainImage_gpu, 0.001,
		                                0.001));
	}
}

#endif  // BUILD_CUDA
