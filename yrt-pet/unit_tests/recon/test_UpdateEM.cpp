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

TEST_CASE("updateem-static", "[updateem]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<int> numVoxelsDistribution(10, 50);
	std::uniform_real_distribution<float> imageLengthDistribution(10.0f, 20.0f);
	std::uniform_real_distribution<float> imageDataDistribution(0.1f, 10.0f);
	std::uniform_real_distribution<float> thresholdDistribution(0.01f, 0.5f);

	constexpr int NumTrials = 3;

	for (int trial = 0; trial < NumTrials; trial++)
	{
		const ssize_t nx = numVoxelsDistribution(engine);
		const ssize_t ny = numVoxelsDistribution(engine);
		const ssize_t nz = numVoxelsDistribution(engine);
		const float length_x = imageLengthDistribution(engine);
		const float length_y = imageLengthDistribution(engine);
		const float length_z = imageLengthDistribution(engine);

		yrt::ImageParams params(nx, ny, nz, length_x, length_y, length_z);

		auto mainImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);
		auto updateImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);
		auto sensImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);

		auto mainImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(mainImage.get());
		auto updateImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(updateImage.get());
		auto sensImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(sensImage.get());

		const float threshold = thresholdDistribution(engine);

		mainImage->updateEMThresholdStatic(updateImage.get(), sensImage.get(),
		                                   threshold);

		mainImageDevice->updateEMThresholdStaticDevice(
		    updateImageDevice.get(), sensImageDevice.get(), threshold, true);

		auto mainImage_gpu =
		    std::make_unique<yrt::ImageOwned>(mainImage->getParams());
		mainImage_gpu->allocate();
		mainImageDevice->transferToHostMemory(mainImage_gpu.get());

		REQUIRE(yrt::util::test::allclose(*mainImage, *mainImage_gpu, 0.001,
		                                  0.001));
	}
}

TEST_CASE("updateem-dynamic-with-scaling", "[updateem]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<int> numVoxelsDistribution(10, 30);
	std::uniform_int_distribution<int> numFramesDistribution(3, 8);
	std::uniform_real_distribution<float> imageLengthDistribution(10.0f, 20.0f);
	std::uniform_real_distribution<float> thresholdDistribution(0.01f, 0.5f);
	std::uniform_real_distribution<float> scalingDistribution(0.5f, 2.0f);

	constexpr int NumTrials = 4;

	for (int trial = 0; trial < NumTrials; trial++)
	{
		const ssize_t nx = numVoxelsDistribution(engine);
		const ssize_t ny = numVoxelsDistribution(engine);
		const ssize_t nz = numVoxelsDistribution(engine);
		const ssize_t nt = numFramesDistribution(engine);
		const float length_x = imageLengthDistribution(engine);
		const float length_y = imageLengthDistribution(engine);
		const float length_z = imageLengthDistribution(engine);

		yrt::ImageParams params(nx, ny, nz, length_x, length_y, length_z, 0.0f,
		                        0.0f, 0.0f, nt);

		auto mainImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);
		auto updateImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);

		yrt::ImageParams sensParams(nx, ny, nz, length_x, length_y, length_z);
		auto sensImage =
		    yrt::util::test::makeImageWithRandomPrism(sensParams, &engine);

		auto mainImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(mainImage.get());
		auto updateImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(updateImage.get());
		auto sensImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(sensImage.get());

		const float threshold = thresholdDistribution(engine);

		// One trial with no scalings
		if (trial == 0)
		{
			mainImage->updateEMThresholdDynamic(updateImage.get(),
			                                    sensImage.get(), threshold);

			mainImageDevice->updateEMThresholdDynamicWithScaling(
			    updateImageDevice.get(), sensImageDevice.get(), nullptr,
			    threshold, true);
		}
		else
		{
			std::vector<float> scaling(nt);
			for (int t = 0; t < nt; t++)
			{
				scaling[t] = scalingDistribution(engine);
			}

			yrt::DeviceArray<float> scalingDevice(scaling.size());
			scalingDevice.copyFromHost(scaling.data(), scaling.size(),
			                           {nullptr, true});

			mainImage->updateEMThresholdDynamic(
			    updateImage.get(), sensImage.get(), scaling, threshold);

			mainImageDevice->updateEMThresholdDynamicWithScaling(
			    updateImageDevice.get(), sensImageDevice.get(),
			    scalingDevice.getDevicePointer(), threshold, true);
		}

		auto mainImage_gpu =
		    std::make_unique<yrt::ImageOwned>(mainImage->getParams());
		mainImage_gpu->allocate();
		mainImageDevice->transferToHostMemory(mainImage_gpu.get());

		REQUIRE(yrt::util::test::allclose(*mainImage, *mainImage_gpu, 0.001,
		                                  0.001));
	}
}

TEST_CASE("updateem-dynamic-with-4d-sens", "[updateem]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<int> numVoxelsDistribution(10, 30);
	std::uniform_int_distribution<int> numFramesDistribution(3, 8);
	std::uniform_real_distribution<float> imageLengthDistribution(10.0f, 20.0f);
	std::uniform_real_distribution<float> thresholdDistribution(0.01f, 0.5f);

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

		yrt::ImageParams params(nx, ny, nz, length_x, length_y, length_z, 0.0f,
		                        0.0f, 0.0f, nt);

		auto mainImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);
		auto updateImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);
		auto sensImage =
		    yrt::util::test::makeImageWithRandomPrism(params, &engine);

		auto mainImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(mainImage.get());
		auto updateImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(updateImage.get());
		auto sensImageDevice =
		    std::make_unique<yrt::ImageDeviceOwned>(sensImage.get());

		const float threshold = thresholdDistribution(engine);

		mainImage->updateEMThresholdDynamic(updateImage.get(), sensImage.get(),
		                                    threshold);

		mainImageDevice->updateEMThresholdDynamicWith4DSens(
		    updateImageDevice.get(), sensImageDevice.get(), threshold, true);

		auto mainImage_gpu =
		    std::make_unique<yrt::ImageOwned>(mainImage->getParams());
		mainImage_gpu->allocate();
		mainImageDevice->transferToHostMemory(mainImage_gpu.get());

		REQUIRE(yrt::util::test::allclose(*mainImage, *mainImage_gpu, 0.001,
		                                  0.001));
	}
}

#endif  // BUILD_CUDA
