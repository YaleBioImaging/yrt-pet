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

	std::uniform_int_distribution<size_t> numFramesDistribution(10, 300);

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
		const size_t numMotionFrames = numFramesDistribution(engine);
		yrt::LORMotion lorMotion(numMotionFrames);
		yrt::timestamp_t currentTimeStamp = 0;

		for (size_t motionFrame = 0; motionFrame < numMotionFrames;
		     motionFrame++)
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
			lorMotion.setTransform(motionFrame, transform);

			// Set timestamp
			lorMotion.setStartingTimestamp(motionFrame, currentTimeStamp);
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

		REQUIRE(yrt::util::test::checkImageAllPositive(*outImage_gpu));
		REQUIRE(yrt::util::test::checkImageAllPositive(*outImage_cpu));

		CHECK(yrt::util::test::allclose(*outImage_cpu, *outImage_gpu, 0.001,
		                                0.001));
	}
}

// Validate that the image time-averaging with a dynamic framing is consistent
//  between the CPU and the GPU implementations (in 4D)
TEST_CASE("image-timeavg-4d", "[timeavg]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<int> numVoxelsDistribution(1, 300);
	std::uniform_real_distribution<float> voxelSizeDistribution(0.1f, 5.0f);

	std::uniform_real_distribution<float> rollDistribution(-yrt::PI_FLT,
	                                                       yrt::PI_FLT);
	std::uniform_real_distribution<float> pitchDistribution(-yrt::PIHALF_FLT,
	                                                        yrt::PIHALF_FLT);
	std::uniform_real_distribution<float> yawDistribution(-yrt::PI_FLT,
	                                                      yrt::PI_FLT);
	std::uniform_real_distribution<float> translationDistribution(0.1f, 5.0f);

	std::uniform_int_distribution<yrt::timestamp_t> scanDurationDistribution(
	    1'500, 30'000);  // in ms
	std::uniform_int_distribution<yrt::timestamp_t>
	    dynamicFrameDurationDistribution(500, 1'500);  // in ms

	// Say that every motion frame is 1/10 s
	constexpr yrt::timestamp_t motionFrameDuration = 100;  // in ms

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
		const yrt::timestamp_t scanDuration = scanDurationDistribution(engine);

		yrt::ImageParams params(nxy, nxy, nz, length_x, length_y, length_z);

		auto inputImage = yrt::util::test::makeImageWithRandomPrism(params);

		// Create transformations in motion frames
		const size_t numMotionFrames = scanDuration / motionFrameDuration;
		yrt::LORMotion lorMotion(numMotionFrames);
		yrt::timestamp_t currentTimeStamp = 0;

		for (size_t motionFrame = 0; motionFrame < numMotionFrames;
		     motionFrame++)
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
			lorMotion.setTransform(motionFrame, transform);

			// Set timestamp
			lorMotion.setStartingTimestamp(motionFrame, currentTimeStamp);
			currentTimeStamp += motionFrameDuration;
		}

		// Create dynamic framing
		std::vector<yrt::timestamp_t> dynamicFramingTimestamps;
		currentTimeStamp = 0;
		bool finished = false;
		while (!finished)
		{
			dynamicFramingTimestamps.push_back(currentTimeStamp);
			const yrt::timestamp_t dynamicFrameLength =
			    dynamicFrameDurationDistribution(engine);
			currentTimeStamp += dynamicFrameLength;
			finished = currentTimeStamp > scanDuration;
		}
		dynamicFramingTimestamps.push_back(currentTimeStamp);
		yrt::DynamicFraming dynamicFraming(dynamicFramingTimestamps);

		// Time-average using CPU
		auto outImage_cpu = yrt::util::timeAverageMoveImageDynamic<false>(
		    lorMotion, inputImage.get(), dynamicFraming);

		// Time-average using GPU
		auto outImageDevice_gpu = yrt::util::timeAverageMoveImageDynamicDevice(
		    lorMotion, inputImage.get(), dynamicFraming, {nullptr, true});
		auto outImage_gpu =
		    std::make_unique<yrt::ImageOwned>(outImageDevice_gpu->getParams());
		outImage_gpu->allocate();
		outImageDevice_gpu->transferToHostMemory(outImage_gpu.get());

		REQUIRE(yrt::util::test::checkImageAllPositive(*outImage_gpu));
		REQUIRE(yrt::util::test::checkImageAllPositive(*outImage_cpu));

		CHECK(yrt::util::test::allclose(*outImage_cpu, *outImage_gpu, 0.001,
		                                0.001));
	}
}

#endif
