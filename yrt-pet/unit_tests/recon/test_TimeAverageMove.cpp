/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#if BUILD_CUDA

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/geometry/TransformUtils.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#include "yrt-pet/utils/ReconstructionUtilsDevice.cuh"

// This test validates that the image time-averaging is consistent between the
//  CPU and the GPU implementations (in 3D only)
TEST_CASE("image-timeavg-3d", "[timeavg]")
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

		yrt::util::test::checkImageAllPositive(*outImage_gpu);
		yrt::util::test::checkImageAllPositive(*outImage_cpu);

		CHECK(yrt::util::test::allclose(*outImage_cpu, *outImage_gpu, 0.001,
		                                0.001));
	}
}

#endif
