/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ImageOps.hpp"

#include "yrt-pet/backends/metal/ImageSpaceKernels.hpp"
#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"

#include <cstdint>
#include <limits>

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

bool is3D(const Image& image)
{
	return image.getParams().nt == 1;
}

bool sameDimensions(const Image& lhs, const Image& rhs)
{
	return lhs.getParams().isSameDimensionsAs(rhs.getParams());
}

bool sameFrameCount(const Image& lhs, const Image& rhs)
{
	return lhs.getParams().isSameNumFramesAs(rhs.getParams());
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

Buffer copyImageToBuffer(const Context& context, const Image& image,
                         const ImageShape& shape)
{
	return Buffer::copyFromHost(context.device(), image.getRawPointer(),
	                            byteCount(shape));
}

bool copyBufferToImage(const Buffer& buffer, Image& image,
                       const ImageShape& shape)
{
	return buffer.copyToHost(image.getRawPointer(), byteCount(shape));
}

ImageThresholdParams makeThresholdParams(float threshold, float valLeScale,
                                         float valLeOffset, float valGtScale,
                                         float valGtOffset)
{
	return {threshold, valLeScale, valLeOffset, valGtScale, valGtOffset};
}

}  // namespace

bool fill(const Context& context, Image& image, float value)
{
	if (!context.isValid() || !hasUsableImageMemory(image))
	{
		return false;
	}

	const ImageShape shape = makeShape(image.getParams());
	Buffer imageBuffer = Buffer::allocate(context.device(), byteCount(shape));
	if (!imageBuffer.isValid() ||
	    !launchImageFill(context.device(), context.library(),
	        context.commandQueue(), imageBuffer, shape, value))
	{
		return false;
	}
	return copyBufferToImage(imageBuffer, image, shape);
}

bool multiplyByScalar(const Context& context, Image& image, float scalar)
{
	if (!context.isValid() || !hasUsableImageMemory(image))
	{
		return false;
	}

	const ImageShape shape = makeShape(image.getParams());
	Buffer imageBuffer = copyImageToBuffer(context, image, shape);
	if (!imageBuffer.isValid() ||
	    !launchImageMultiplyScalar(context.device(), context.library(),
	        context.commandQueue(), imageBuffer, shape, scalar))
	{
		return false;
	}
	return copyBufferToImage(imageBuffer, image, shape);
}

bool add3DTo3D(const Context& context, const Image& input3D, Image& output3D)
{
	if (!context.isValid() || !hasUsableImageMemory(input3D) ||
	    !hasUsableImageMemory(output3D) || !is3D(input3D) ||
	    !is3D(output3D) || !sameDimensions(input3D, output3D))
	{
		return false;
	}

	const ImageShape shape = makeShape(output3D.getParams());
	Buffer inputBuffer = copyImageToBuffer(context, input3D, shape);
	Buffer outputBuffer = copyImageToBuffer(context, output3D, shape);
	if (!inputBuffer.isValid() || !outputBuffer.isValid() ||
	    !launchImageAdd3DTo3D(context.device(), context.library(),
	        context.commandQueue(), inputBuffer, outputBuffer, shape))
	{
		return false;
	}
	return copyBufferToImage(outputBuffer, output3D, shape);
}

bool add3DTo4D(const Context& context, const Image& input3D, Image& output4D)
{
	if (!context.isValid() || !hasUsableImageMemory(input3D) ||
	    !hasUsableImageMemory(output4D) || !is3D(input3D) ||
	    !sameDimensions(input3D, output4D))
	{
		return false;
	}

	const ImageShape inputShape = makeShape(input3D.getParams());
	const ImageShape outputShape = makeShape(output4D.getParams());
	Buffer inputBuffer = copyImageToBuffer(context, input3D, inputShape);
	Buffer outputBuffer = copyImageToBuffer(context, output4D, outputShape);
	if (!inputBuffer.isValid() || !outputBuffer.isValid() ||
	    !launchImageAdd3DTo4D(context.device(), context.library(),
	        context.commandQueue(), inputBuffer, outputBuffer, outputShape))
	{
		return false;
	}
	return copyBufferToImage(outputBuffer, output4D, outputShape);
}

bool applyThreshold(const Context& context, Image& image3D, const Image& mask3D,
                    float threshold, float valLeScale, float valLeOffset,
                    float valGtScale, float valGtOffset)
{
	if (!context.isValid() || !hasUsableImageMemory(image3D) ||
	    !hasUsableImageMemory(mask3D) || !is3D(image3D) || !is3D(mask3D) ||
	    !sameDimensions(image3D, mask3D))
	{
		return false;
	}

	const ImageShape shape = makeShape(image3D.getParams());
	Buffer imageBuffer = copyImageToBuffer(context, image3D, shape);
	Buffer maskBuffer = copyImageToBuffer(context, mask3D, shape);
	if (!imageBuffer.isValid() || !maskBuffer.isValid() ||
	    !launchImageApplyThreshold(context.device(), context.library(),
	        context.commandQueue(), imageBuffer, maskBuffer, shape,
	        makeThresholdParams(threshold, valLeScale, valLeOffset, valGtScale,
	            valGtOffset)))
	{
		return false;
	}
	return copyBufferToImage(imageBuffer, image3D, shape);
}

bool applyThresholdBroadcast(const Context& context, Image& image4D,
                             const Image& mask3D, float threshold,
                             float valLeScale, float valLeOffset,
                             float valGtScale, float valGtOffset)
{
	if (!context.isValid() || !hasUsableImageMemory(image4D) ||
	    !hasUsableImageMemory(mask3D) || !is3D(mask3D) ||
	    !sameDimensions(image4D, mask3D))
	{
		return false;
	}

	const ImageShape imageShape = makeShape(image4D.getParams());
	const ImageShape maskShape = makeShape(mask3D.getParams());
	Buffer imageBuffer = copyImageToBuffer(context, image4D, imageShape);
	Buffer maskBuffer = copyImageToBuffer(context, mask3D, maskShape);
	if (!imageBuffer.isValid() || !maskBuffer.isValid() ||
	    !launchImageApplyThresholdBroadcast(context.device(), context.library(),
	        context.commandQueue(), imageBuffer, maskBuffer, imageShape,
	        makeThresholdParams(threshold, valLeScale, valLeOffset, valGtScale,
	            valGtOffset)))
	{
		return false;
	}
	return copyBufferToImage(imageBuffer, image4D, imageShape);
}

bool updateEMStatic(const Context& context, Image& image3D,
                    const Image& update3D,
                    const Image& sensitivity3D, float threshold)
{
	if (!context.isValid() || !hasUsableImageMemory(image3D) ||
	    !hasUsableImageMemory(update3D) || !hasUsableImageMemory(sensitivity3D) ||
	    !is3D(image3D) || !is3D(update3D) || !is3D(sensitivity3D) ||
	    !sameDimensions(image3D, update3D) ||
	    !sameDimensions(image3D, sensitivity3D))
	{
		return false;
	}

	const ImageShape shape = makeShape(image3D.getParams());
	Buffer imageBuffer = copyImageToBuffer(context, image3D, shape);
	Buffer updateBuffer = copyImageToBuffer(context, update3D, shape);
	Buffer sensitivityBuffer = copyImageToBuffer(context, sensitivity3D, shape);
	if (!imageBuffer.isValid() || !updateBuffer.isValid() ||
	    !sensitivityBuffer.isValid() ||
	    !launchImageUpdateEMStatic(context.device(), context.library(),
	        context.commandQueue(), updateBuffer, imageBuffer, sensitivityBuffer,
	        shape, threshold))
	{
		return false;
	}
	return copyBufferToImage(imageBuffer, image3D, shape);
}

bool updateEMDynamic(const Context& context, Image& image4D,
                     const Image& update4D,
                     const Image& sensitivity3D, float threshold)
{
	if (!context.isValid() || !hasUsableImageMemory(image4D) ||
	    !hasUsableImageMemory(update4D) || !hasUsableImageMemory(sensitivity3D) ||
	    !is3D(sensitivity3D) || !sameDimensions(image4D, update4D) ||
	    !sameFrameCount(image4D, update4D) ||
	    !sameDimensions(image4D, sensitivity3D))
	{
		return false;
	}

	const ImageShape imageShape = makeShape(image4D.getParams());
	const ImageShape sensitivityShape = makeShape(sensitivity3D.getParams());
	Buffer imageBuffer = copyImageToBuffer(context, image4D, imageShape);
	Buffer updateBuffer = copyImageToBuffer(context, update4D, imageShape);
	Buffer sensitivityBuffer =
	    copyImageToBuffer(context, sensitivity3D, sensitivityShape);
	if (!imageBuffer.isValid() || !updateBuffer.isValid() ||
	    !sensitivityBuffer.isValid() ||
	    !launchImageUpdateEMDynamic(context.device(), context.library(),
	        context.commandQueue(), updateBuffer, imageBuffer, sensitivityBuffer,
	        imageShape, threshold))
	{
		return false;
	}
	return copyBufferToImage(imageBuffer, image4D, imageShape);
}

}  // namespace yrt::backend::metal
