/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/MotionOps.hpp"

#include "yrt-pet/backends/metal/ImageSpaceKernels.hpp"
#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/geometry/TransformUtils.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace yrt::backend::metal
{
namespace
{

bool fitsUint32(std::int64_t value)
{
	return value > 0 &&
	       static_cast<std::uint64_t>(value) <=
	           std::numeric_limits<std::uint32_t>::max();
}

bool hasUsableImageMemory(const Image& image)
{
	const ImageParams& params = image.getParams();
	return image.isMemoryValid() && fitsUint32(params.nx) &&
	       fitsUint32(params.ny) && fitsUint32(params.nz) &&
	       fitsUint32(params.nt);
}

ImageShape makeShape(const ImageParams& params)
{
	return {static_cast<std::uint32_t>(params.nx),
	    static_cast<std::uint32_t>(params.ny),
	    static_cast<std::uint32_t>(params.nz),
	    static_cast<std::uint32_t>(params.nt)};
}

std::size_t byteCount(const ImageShape& shape)
{
	return sizeof(float) * shape.voxelCount();
}

bool sameGeometryIgnoreFrames(const ImageParams& lhs, const ImageParams& rhs)
{
	return lhs.isSameAsIgnoreFrames(rhs);
}

bool selectMotionFrames(const LORMotion& lorMotion, timestamp_t timeStart,
    timestamp_t timeStop, std::vector<transform_t>& inverseTransforms,
    std::vector<float>& weights)
{
	if (timeStop <= timeStart)
	{
		return false;
	}

	const auto numMotionFrames =
	    static_cast<frame_t>(lorMotion.getNumFrames());
	frame_t frameStart = -1;
	frame_t frameStop = numMotionFrames;
	for (frame_t frame = 0; frame < numMotionFrames; ++frame)
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

	if (frameStart < 0 || frameStop <= frameStart)
	{
		return false;
	}

	const auto scanDuration = static_cast<float>(timeStop - timeStart);
	const auto frameCount =
	    static_cast<std::size_t>(frameStop - frameStart);
	inverseTransforms.resize(frameCount);
	weights.resize(frameCount);
	for (std::size_t frameIndex = 0; frameIndex < frameCount; ++frameIndex)
	{
		const frame_t frame = static_cast<frame_t>(frameIndex) + frameStart;
		inverseTransforms[frameIndex] =
		    util::invertTransform(lorMotion.getTransform(frame));
		weights[frameIndex] = lorMotion.getDuration(frame) / scanDuration;
	}

	return true;
}

}  // namespace

std::unique_ptr<ImageOwned> timeAverageMoveImage(
    const Context& context, const LORMotion& lorMotion,
    const Image& unmovedImage, timestamp_t timeStart, timestamp_t timeStop)
{
	const ImageParams& params = unmovedImage.getParams();
	auto outImage = std::make_unique<ImageOwned>(params);
	outImage->allocate();
	outImage->fill(0.0f);
	if (!timeAverageMoveImage(context, lorMotion, unmovedImage, *outImage,
	        timeStart, timeStop, 0))
	{
		return nullptr;
	}
	return outImage;
}

bool timeAverageMoveImage(const Context& context, const LORMotion& lorMotion,
    const Image& unmovedImage, Image& outImage, timestamp_t timeStart,
    timestamp_t timeStop, frame_t outDynamicFrame)
{
	if (!context.isValid() || !hasUsableImageMemory(unmovedImage) ||
	    !hasUsableImageMemory(outImage) || unmovedImage.getParams().nt != 1 ||
	    outDynamicFrame < 0 ||
	    outDynamicFrame >= outImage.getParams().nt ||
	    !sameGeometryIgnoreFrames(
	        unmovedImage.getParams(), outImage.getParams()))
	{
		return false;
	}

	std::vector<transform_t> inverseTransforms;
	std::vector<float> weights;
	if (!selectMotionFrames(lorMotion, timeStart, timeStop, inverseTransforms,
	        weights))
	{
		return false;
	}

	const ImageShape inputShape = makeShape(unmovedImage.getParams());
	const ImageShape outputShape = makeShape(outImage.getParams());
	Buffer inputBuffer = Buffer::copyFromHost(context.device(),
	    unmovedImage.getRawPointer(), byteCount(inputShape));
	Buffer outputBuffer = Buffer::copyFromHost(context.device(),
	    outImage.getRawPointer(), byteCount(outputShape));
	Buffer transformBuffer = Buffer::copyFromHost(context.device(),
	    inverseTransforms.data(), sizeof(transform_t) * inverseTransforms.size());
	Buffer weightBuffer = Buffer::copyFromHost(context.device(), weights.data(),
	    sizeof(float) * weights.size());
	if (!inputBuffer.isValid() || !outputBuffer.isValid() ||
	    !transformBuffer.isValid() || !weightBuffer.isValid())
	{
		return false;
	}

	if (!launchImageTimeAverageMove3D(context.device(), context.library(),
	        context.commandQueue(), inputBuffer, outputBuffer, transformBuffer,
	        weightBuffer, outputShape, unmovedImage.getParams().length_x,
	        unmovedImage.getParams().length_y,
	        unmovedImage.getParams().length_z, unmovedImage.getParams().off_x,
	        unmovedImage.getParams().off_y, unmovedImage.getParams().off_z,
	        static_cast<std::uint32_t>(outDynamicFrame),
	        static_cast<std::uint32_t>(inverseTransforms.size())))
	{
		return false;
	}

	return outputBuffer.copyToHost(outImage.getRawPointer(),
	    byteCount(outputShape));
}

}  // namespace yrt::backend::metal
