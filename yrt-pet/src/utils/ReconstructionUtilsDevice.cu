/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/ReconstructionUtilsDevice.cuh"

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/image/ImageSpaceKernels.cuh"
#include "yrt-pet/datastruct/projection/DynamicFraming.hpp"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/geometry/TransformUtils.hpp"
#include "yrt-pet/operators/DeviceSynchronized.cuh"
#include "yrt-pet/operators/OperatorProjectorDD_GPU.cuh"
#include "yrt-pet/operators/OperatorProjectorSiddon_GPU.cuh"
#include "yrt-pet/recon/LREM_GPU.cuh"
#include "yrt-pet/recon/OSEM_GPU.cuh"
#include "yrt-pet/utils/DeviceArray.cuh"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
	m.def(
	    "createOperatorProjectorDevice",
	    [](const ProjectorParams& projParams, const BinIterator& binIter,
	       const std::vector<Constraint*>& constraints)
	    {
		    return util::createOperatorProjectorDevice(
		        projParams, binIter, constraints, nullptr, nullptr);
	    },
	    "proj_params"_a, "bin_iter"_a,
	    "constraints"_a = std::vector<Constraint*>());
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

void timeAverageMoveImageDevice(const LORMotion& lorMotion,
                                const ImageBase* unmovedImage,
                                ImageDevice* outImage, frame_t outDynamicFrame,
                                GPULaunchConfig launchConfig)
{
	auto [timeStart, timeStop] = getFullTimeRange(lorMotion);
	timeAverageMoveImageDevice(lorMotion, unmovedImage, outImage, timeStart,
	                           timeStop, outDynamicFrame, launchConfig);
}

std::unique_ptr<ImageDevice> timeAverageMoveImageDevice(
    const LORMotion& lorMotion, const ImageBase* unmovedImage,
    timestamp_t timeStart, timestamp_t timeStop, GPULaunchConfig launchConfig)
{
	const ImageParams& params = unmovedImage->getParams();

	// Prepare output image
	auto outImage = std::make_unique<ImageDeviceOwned>(params);
	outImage->allocate(true, true);

	timeAverageMoveImageDevice(lorMotion, unmovedImage, outImage.get(),
	                           timeStart, timeStop, 0, launchConfig);

	return outImage;
}

void timeAverageMoveImageDevice(const LORMotion& lorMotion,
                                const ImageBase* unmovedImage,
                                ImageDevice* outImage, timestamp_t timeStart,
                                timestamp_t timeStop, frame_t outDynamicFrame,
                                GPULaunchConfig launchConfig)
{
	ASSERT_MSG(unmovedImage != nullptr, "Input image is null");
	ASSERT_MSG(outImage != nullptr, "Output image is null");
	ASSERT_MSG(outImage->isMemoryValid(), "Output image is not allocated");
	ASSERT_MSG(timeStop > timeStart,
	           "Time stop must be greater than time start");

	const int64_t numMotionFrames = lorMotion.getNumFrames();
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

		unmovedImageDevice =
		    std::make_unique<ImageDeviceOwned>(unmovedImageHost_ptr);

		unmovedImageDevice_ptr = unmovedImageDevice.get();
	}
	ASSERT(unmovedImageDevice_ptr != nullptr);

	const ImageParams& params = unmovedImageDevice_ptr->getParams();
	const GPULaunchParams3D launchParams = initiateDeviceParameters(params);

	frame_t frameStart = -1;
	frame_t frameStop = numMotionFrames;

	// Compute how many frames in total
	for (frame_t frame = 0; frame < numMotionFrames; frame++)
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
	util::parallelForChunked(
	    numFramesUsed, globals::getNumThreads(),
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

	if (launchConfig.stream != nullptr)
	{
		timeAverageMoveImage_kernel<true>
		    <<<launchParams.gridSize, launchParams.blockSize, 0,
		       *launchConfig.stream>>>(
		        unmovedImageDevice_ptr->getDevicePointer(),
		        outImage->getDevicePointer(), outDynamicFrame, params.nx,
		        params.ny, params.nz, params.length_x, params.length_y,
		        params.length_z, params.off_x, params.off_y, params.off_z,
		        transformsDevice.getDevicePointer(),
		        weightsDevice.getDevicePointer(), numFramesUsed);
	}
	else
	{
		timeAverageMoveImage_kernel<true>
		    <<<launchParams.gridSize, launchParams.blockSize>>>(
		        unmovedImageDevice_ptr->getDevicePointer(),
		        outImage->getDevicePointer(), outDynamicFrame, params.nx,
		        params.ny, params.nz, params.length_x, params.length_y,
		        params.length_z, params.off_x, params.off_y, params.off_z,
		        transformsDevice.getDevicePointer(),
		        weightsDevice.getDevicePointer(), numFramesUsed);
	}
	synchronizeIfNeeded(launchConfig);
	ASSERT(cudaCheckError());
}

std::unique_ptr<ImageDevice> timeAverageMoveImageDynamicDevice(
    const LORMotion& lorMotion, const ImageBase* unmovedImage,
    const DynamicFraming& dynamicFraming, GPULaunchConfig launchConfig)
{
	ASSERT_MSG(unmovedImage != nullptr, "Null input image given");

	const frame_t numDynamicFrames =
	    static_cast<frame_t>(dynamicFraming.getNumFrames());

	// Create the output image on the device
	ImageParams params = unmovedImage->getParams();
	params.nt = numDynamicFrames;
	auto outImage = std::make_unique<ImageDeviceOwned>(params);
	outImage->allocate();

	timeAverageMoveImageDynamicDevice(lorMotion, unmovedImage, outImage.get(),
	                                  dynamicFraming,
	                                  {launchConfig.stream, true});

	return outImage;
}

void timeAverageMoveImageDynamicDevice(const LORMotion& lorMotion,
                                       const ImageBase* unmovedImage,
                                       ImageDevice* outImage,
                                       const DynamicFraming& dynamicFraming,
                                       GPULaunchConfig launchConfig)
{
	ASSERT_MSG(unmovedImage != nullptr, "Null input image given");

	const frame_t numDynamicFrames =
	    static_cast<frame_t>(dynamicFraming.getNumFrames());

	ASSERT_MSG(outImage->getNumFrames() != numDynamicFrames,
	           "Output image does not have the same number of frames as the "
	           "given dynamic framing.");

	// Make sure the input image is in the device
	//  (otherwise create it and copy it there)
	std::unique_ptr<ImageDeviceOwned> unmovedImageDevice;
	auto unmovedImageDevice_ptr =
	    dynamic_cast<const ImageDevice*>(unmovedImage);
	if (unmovedImageDevice_ptr == nullptr)
	{
		const auto unmovedImageHost_ptr =
		    dynamic_cast<const Image*>(unmovedImage);
		ASSERT_MSG(unmovedImageHost_ptr != nullptr, "Unknown image type");

		unmovedImageDevice =
		    std::make_unique<ImageDeviceOwned>(unmovedImageHost_ptr);

		unmovedImageDevice_ptr = unmovedImageDevice.get();
	}
	ASSERT(unmovedImageDevice_ptr != nullptr);


	for (frame_t dynamicFrame = 0; dynamicFrame < numDynamicFrames;
	     dynamicFrame++)
	{
		const timestamp_t dynamicFrameStart =
		    dynamicFraming.getStartingTimestamp(dynamicFrame);
		const timestamp_t dynamicFrameStop =
		    dynamicFraming.getStoppingTimestamp(dynamicFrame);

		timeAverageMoveImageDevice(lorMotion, unmovedImageDevice_ptr, outImage,
		                           dynamicFrameStart, dynamicFrameStop,
		                           dynamicFrame, {launchConfig.stream, false});
	}

	synchronizeIfNeeded(launchConfig);
}

std::unique_ptr<OSEM> createOSEM_GPU(const Scanner& scanner, bool isLowRank)
{
	std::unique_ptr<OSEM> osem;
	if (!isLowRank)
	{
		osem = std::make_unique<OSEM_GPU>(scanner);
	}
	else
	{
		osem = std::make_unique<LREM_GPU>(scanner);
	}
	return osem;
}

std::unique_ptr<OperatorProjectorBase> createOperatorProjectorDevice(
    const ProjectorParams& projParams, const BinIterator& binIter,
    const std::vector<Constraint*>& constraintsPtr,
    const cudaStream_t* mainStream, const cudaStream_t* auxStream)
{
	const ProjectorType projType = projParams.projectorType;
	if (projType == ProjectorType::SIDDON)
	{
#ifdef BUILD_CUDA
		return std::make_unique<OperatorProjectorSiddon_GPU>(
		    projParams, &binIter, constraintsPtr, mainStream, auxStream);
#else
		throw std::runtime_error("Siddon GPU projector not supported because "
		                         "project was not compiled with CUDA");
#endif
	}
	else if (projType == ProjectorType::DD)
	{
#ifdef BUILD_CUDA
		return std::make_unique<OperatorProjectorDD_GPU>(
		    projParams, &binIter, constraintsPtr, mainStream, auxStream);
#else
		throw std::runtime_error(
		    "Distance-driven GPU projector not supported because "
		    "project was not compiled with CUDA");
#endif
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}
}

}  // namespace yrt::util
