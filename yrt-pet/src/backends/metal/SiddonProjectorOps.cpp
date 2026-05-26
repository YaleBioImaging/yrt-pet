/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/SiddonProjectorOps.hpp"

#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/ProjectionBatchMetal.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorKernels.hpp"
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

bool fitsUint32Size(std::size_t value)
{
	return value > 0 &&
	       value <= static_cast<std::size_t>(
	                    std::numeric_limits<std::uint32_t>::max());
}

bool hasUsableImageMemory(const Image& image)
{
	const ImageParams& params = image.getParams();
	return image.isMemoryValid() && fitsUint32(params.nx) &&
	       fitsUint32(params.ny) && fitsUint32(params.nz) &&
	       fitsUint32(params.nt);
}

SiddonForwardImageParams makeSiddonParams(const ImageParams& params,
                                          std::uint32_t frame)
{
	return {static_cast<std::uint32_t>(params.nx),
	    static_cast<std::uint32_t>(params.ny),
	    static_cast<std::uint32_t>(params.nz),
	    static_cast<std::uint32_t>(params.nt),
	    frame,
	    params.length_x,
	    params.length_y,
	    params.length_z,
	    params.vx,
	    params.vy,
	    params.vz,
	    params.fovRadius};
}

std::size_t imageByteCount(const SiddonForwardImageParams& params)
{
	return sizeof(float) * static_cast<std::size_t>(params.nx) *
	       static_cast<std::size_t>(params.ny) *
	       static_cast<std::size_t>(params.nz) *
	       static_cast<std::size_t>(params.nt);
}

}  // namespace

bool forwardProjectSiddonSingleRay(const Context& context, const Image& image,
    ProjectionBatchMetal& batch, std::uint32_t frame)
{
	if (!context.isValid() || !hasUsableImageMemory(image) || !batch.isValid() ||
	    !fitsUint32Size(batch.size()))
	{
		return false;
	}

	const ImageParams& imageParams = image.getParams();
	const auto frameCount = static_cast<std::uint32_t>(imageParams.nt);
	if (frame >= frameCount)
	{
		return false;
	}

	const SiddonForwardImageParams params = makeSiddonParams(imageParams, frame);
	Buffer imageBuffer = Buffer::copyFromHost(context.device(),
	    image.getRawPointer(), imageByteCount(params));
	return imageBuffer.isValid() &&
	       launchSiddonForwardSingleRay(context.device(), context.library(),
	           context.commandQueue(), imageBuffer, batch.lorBuffer(),
	           batch.projectionValuesBuffer(), params, batch.size());
}

bool backProjectSiddonSingleRay(const Context& context,
    const ProjectionBatchMetal& batch, Image& image, std::uint32_t frame)
{
	if (!context.isValid() || !hasUsableImageMemory(image) || !batch.isValid() ||
	    !fitsUint32Size(batch.size()))
	{
		return false;
	}

	const ImageParams& imageParams = image.getParams();
	const auto frameCount = static_cast<std::uint32_t>(imageParams.nt);
	if (frame >= frameCount)
	{
		return false;
	}

	const SiddonForwardImageParams params = makeSiddonParams(imageParams, frame);
	Buffer imageBuffer = Buffer::copyFromHost(context.device(),
	    image.getRawPointer(), imageByteCount(params));
	if (!imageBuffer.isValid() ||
	    !launchSiddonBackProjectSingleRay(context.device(), context.library(),
	        context.commandQueue(), imageBuffer, batch.lorBuffer(),
	        batch.projectionValuesBuffer(), params, batch.size()))
	{
		return false;
	}

	return imageBuffer.copyToHost(image.getRawPointer(), imageByteCount(params));
}

}  // namespace yrt::backend::metal
