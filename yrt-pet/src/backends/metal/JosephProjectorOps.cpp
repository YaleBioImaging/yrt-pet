/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/JosephProjectorOps.hpp"

#include "yrt-pet/backends/metal/JosephProjectorKernels.hpp"
#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/ProjectionBatchMetal.hpp"
#include "yrt-pet/backends/metal/ProjectorProfile.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorOps.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"

#include <chrono>
#include <limits>

namespace yrt::backend::metal
{
namespace
{

using Clock = std::chrono::steady_clock;

double getElapsedSeconds(Clock::time_point start, Clock::time_point end)
{
	return std::chrono::duration<double>(end - start).count();
}

bool fitsUint32Size(std::size_t value)
{
	return value > 0 &&
	       value <= static_cast<std::size_t>(
	                    std::numeric_limits<std::uint32_t>::max());
}

}  // namespace

bool forwardProjectJosephSingleRay(const Context& context, const Image& image,
    ProjectionBatchMetal& batch, std::uint32_t frame,
    SiddonProjectorKernelProfile* profile)
{
	if (!context.isValid() || !batch.isValid() || !fitsUint32Size(batch.size()))
	{
		return false;
	}

	SiddonForwardImageParams params{};
	Buffer imageBuffer;
	if (!makeSiddonForwardImageParams(image, frame, params) ||
	    !uploadSiddonImageBuffer(context, image, imageBuffer, profile))
	{
		return false;
	}

	return forwardProjectJosephSingleRay(context, imageBuffer, batch, params,
	                                     profile);
}

bool forwardProjectJosephSingleRay(const Context& context,
    const Buffer& imageBuffer, ProjectionBatchMetal& batch,
    const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile* profile)
{
	if (!context.isValid() || !imageBuffer.isValid() || !batch.isValid() ||
	    !fitsUint32Size(batch.size()))
	{
		return false;
	}

	const auto kernelStart = Clock::now();
	const bool didRun = launchJosephForwardSingleRay(
	    context.device(), context.library(), context.commandQueue(),
	    imageBuffer, batch.lorBuffer(), batch.projectionValuesBuffer(), params,
	    batch.size());
	if (profile != nullptr)
	{
		profile->kernelSeconds += getElapsedSeconds(kernelStart, Clock::now());
	}
	return didRun;
}

bool backProjectJosephSingleRay(const Context& context,
    const ProjectionBatchMetal& batch, Image& image, std::uint32_t frame,
    SiddonProjectorKernelProfile* profile)
{
	if (!context.isValid() || !batch.isValid() || !fitsUint32Size(batch.size()))
	{
		return false;
	}

	SiddonForwardImageParams params{};
	Buffer imageBuffer;
	if (!makeSiddonForwardImageParams(image, frame, params) ||
	    !uploadSiddonImageBuffer(context, image, imageBuffer, profile))
	{
		return false;
	}

	return backProjectJosephSingleRay(context, batch, imageBuffer, params,
	                                  profile) &&
	       downloadSiddonImageBuffer(context, imageBuffer, image, profile);
}

bool backProjectJosephSingleRay(const Context& context,
    const ProjectionBatchMetal& batch, Buffer& imageBuffer,
    const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile* profile)
{
	if (!context.isValid() || !imageBuffer.isValid() || !batch.isValid() ||
	    !fitsUint32Size(batch.size()))
	{
		return false;
	}

	const auto kernelStart = Clock::now();
	const bool didRun = launchJosephBackProjectSingleRay(
	    context.device(), context.library(), context.commandQueue(),
	    imageBuffer, batch.lorBuffer(), batch.projectionValuesBuffer(), params,
	    batch.size());
	if (profile != nullptr)
	{
		profile->kernelSeconds += getElapsedSeconds(kernelStart, Clock::now());
	}
	return didRun;
}

}  // namespace yrt::backend::metal
