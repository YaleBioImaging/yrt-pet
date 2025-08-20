/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/ReconstructionUtilsDevice.cuh"

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/image/ImageSpaceKernels.cuh"
#include "yrt-pet/geometry/TransformUtils.hpp"
#include "yrt-pet/operators/DeviceSynchronized.cuh"
#include "yrt-pet/utils/DeviceArray.cuh"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_reconstructionutilsdevice(pybind11::module& m)
{
	m.def(
	    "timeAverageMoveImageDevice",
	    [](const LORMotion& lorMotion, const ImageBase* unmovedImage)
	    {
		    return util::timeAverageMoveImageDevice(lorMotion, unmovedImage,
		                                            {nullptr, true});
	    },
	    "lorMotion"_a, "unmovedSensImage"_a,
	    "Blur a given image based on given motion information using the GPU");
	m.def(
	    "timeAverageMoveImageDevice",
	    [](const LORMotion& lorMotion, const ImageBase* unmovedImage,
	       timestamp_t timeStart, timestamp_t timeStop)
	    {
		    return util::timeAverageMoveImageDevice(
		        lorMotion, unmovedImage, timeStart, timeStop, {nullptr, true});
	    },
	    "lorMotion"_a, "unmovedSensImage"_a, "timeStart"_a, "timeStop"_a,
	    "Blur a given image based on given motion information using the GPU");
}
}  // namespace yrt
#endif

namespace yrt::util
{

std::unique_ptr<ImageDevice>
    timeAverageMoveImageDevice(const LORMotion& lorMotion,
                               const ImageBase* unmovedImage,
                               GPULaunchConfig launchConfig)
{
	auto [timeStart, timeStop] = getFullTimeRange(lorMotion);
	return timeAverageMoveImageDevice(lorMotion, unmovedImage, timeStart,
	                                  timeStop, launchConfig);
}

std::unique_ptr<ImageDevice> timeAverageMoveImageDevice(
    const LORMotion& lorMotion, const ImageBase* unmovedImage,
    timestamp_t timeStart, timestamp_t timeStop, GPULaunchConfig launchConfig)
{
	ASSERT_MSG(unmovedImage != nullptr, "Input image is null");
	ASSERT_MSG(timeStop > timeStart,
	           "Time stop must be greater than time start");

	const int64_t numFrames = lorMotion.getNumFrames();
	const auto scanDuration = static_cast<float>(timeStop - timeStart);

	// Check if image is already in device
	auto unmovedImageDevice_ptr =
	    dynamic_cast<const ImageDevice*>(unmovedImage);

	// Initialize here for scope
	std::unique_ptr<ImageDeviceOwned> unmovedImageDevice;

	// If image is on host, copy it to device
	if (unmovedImageDevice_ptr == nullptr)
	{
		const auto unmovedImageHost_ptr =
		    dynamic_cast<const Image*>(unmovedImage);
		ASSERT_MSG(unmovedImageHost_ptr != nullptr, "Unknown image type");

		unmovedImageDevice = std::make_unique<ImageDeviceOwned>(
		    unmovedImageHost_ptr->getParams());
		unmovedImageDevice->allocate();
		unmovedImageDevice->copyFromHostImage(unmovedImageHost_ptr, true);

		unmovedImageDevice_ptr = unmovedImageDevice.get();
	}
	ASSERT(unmovedImageDevice_ptr != nullptr);

	const ImageParams& params = unmovedImageDevice_ptr->getParams();
	const GPULaunchParams3D launchParams = initiateDeviceParameters(params);

	frame_t frameStart = -1;
	frame_t frameStop = numFrames;

	// Compute how many frames in total
	for (frame_t frame = 0; frame < numFrames; frame++)
	{
		const timestamp_t startingTimestamp =
		    lorMotion.getStartingTimestamp(frame);
		if (startingTimestamp >= timeStart)
		{
			if (frameStart < 0)
			{
				frameStart = frame;
			}
			if (startingTimestamp > timeStop)
			{
				frameStop = frame;
				break;
			}
		}
	}
	ASSERT(frameStop > frameStart);

	const size_t numFramesUsed = frameStop - frameStart;

	// Prepare host buffers
	PageLockedBuffer<transform_t> invTransforms{numFramesUsed};
	transform_t* invTransforms_ptr = invTransforms.getPointer();
	PageLockedBuffer<float> weights{numFramesUsed};
	float* weights_ptr = weights.getPointer();
	const LORMotion* lorMotion_ptr = &lorMotion;

	// Populate transforms and weights buffers
	util::parallel_for_chunked(
	    numFramesUsed, globals::numThreads(),
	    [frameStart, invTransforms_ptr, weights_ptr, scanDuration,
	     lorMotion_ptr](size_t frame_i, size_t /*tid*/)
	    {
		    const frame_t frame = frame_i + frameStart;
		    const transform_t transform = lorMotion_ptr->getTransform(frame);
		    const float weight =
		        lorMotion_ptr->getDuration(frame) / scanDuration;

		    invTransforms_ptr[frame_i] = invertTransform(transform);
		    weights_ptr[frame_i] = weight;
	    });

	// Transfer transforms and weights
	DeviceArray<transform_t> transformsDevice{numFramesUsed,
	                                          launchConfig.stream};
	transformsDevice.copyFromHost(invTransforms_ptr, numFramesUsed,
	                              {launchConfig.stream, false});

	DeviceArray<float> weightsDevice{numFramesUsed, launchConfig.stream};
	weightsDevice.copyFromHost(weights_ptr, numFramesUsed,
	                           {launchConfig.stream, false});

	// Prepare output image
	auto outImage = std::make_unique<ImageDeviceOwned>(params);
	outImage->allocate(true, true);

	if (launchConfig.stream != nullptr)
	{
		timeAverageMoveImage_kernel<true>
		    <<<launchParams.gridSize, launchParams.blockSize, 0,
		       *launchConfig.stream>>>(
		        unmovedImageDevice_ptr->getDevicePointer(),
		        outImage->getDevicePointer(), params.nx, params.ny, params.nz,
		        params.length_x, params.length_y, params.length_z, params.off_x,
		        params.off_y, params.off_z, transformsDevice.getDevicePointer(),
		        weightsDevice.getDevicePointer(), numFramesUsed);

		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		timeAverageMoveImage_kernel<true>
		    <<<launchParams.gridSize, launchParams.blockSize>>>(
		        unmovedImageDevice_ptr->getDevicePointer(),
		        outImage->getDevicePointer(), params.nx, params.ny, params.nz,
		        params.length_x, params.length_y, params.length_z, params.off_x,
		        params.off_y, params.off_z, transformsDevice.getDevicePointer(),
		        weightsDevice.getDevicePointer(), numFramesUsed);

		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	ASSERT(cudaCheckError());


	return outImage;
}

}  // namespace yrt::util
