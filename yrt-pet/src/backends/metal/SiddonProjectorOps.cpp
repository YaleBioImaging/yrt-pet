/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/SiddonProjectorOps.hpp"

#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/ProjectionBatchMetal.hpp"
#include "yrt-pet/backends/metal/ProjectorProfile.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorKernels.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace yrt::backend::metal
{
namespace
{

using Clock = std::chrono::steady_clock;

double getElapsedSeconds(Clock::time_point start, Clock::time_point end)
{
	return std::chrono::duration<double>(end - start).count();
}

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
	    params.vx != 0.0f ? 1.0f / params.vx : 0.0f,
	    params.vy != 0.0f ? 1.0f / params.vy : 0.0f,
	    params.vz != 0.0f ? 1.0f / params.vz : 0.0f,
	    0.5f * params.length_x,
	    0.5f * params.length_y,
	    0.5f * params.length_z,
	    params.fovRadius};
}

std::size_t imageByteCount(const SiddonForwardImageParams& params)
{
	return sizeof(float) * static_cast<std::size_t>(params.nx) *
	       static_cast<std::size_t>(params.ny) *
	       static_cast<std::size_t>(params.nz) *
	       static_cast<std::size_t>(params.nt);
}

std::size_t imageVoxelCount(const SiddonForwardImageParams& params)
{
	return static_cast<std::size_t>(params.nx) *
	       static_cast<std::size_t>(params.ny) *
	       static_cast<std::size_t>(params.nz) *
	       static_cast<std::size_t>(params.nt);
}

std::size_t updateCountByteCount(std::size_t count)
{
	return sizeof(std::uint32_t) * count;
}

std::size_t percentileCount(const std::vector<std::uint32_t>& sortedCounts,
                            double percentile)
{
	if (sortedCounts.empty())
	{
		return 0;
	}
	const auto rank = static_cast<std::size_t>(
	    std::ceil(percentile * static_cast<double>(sortedCounts.size())));
	const std::size_t index =
	    std::min(sortedCounts.size() - 1, rank == 0 ? 0 : rank - 1);
	return sortedCounts[index];
}

bool collectBackProjectUpdateCounts(const Context& context,
    const ProjectionBatchMetal& batch, const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile& profile,
    const ProjectorKernelOptions* options)
{
	std::vector<std::uint32_t> counts(batch.size(), 0);
	const auto countStart = Clock::now();
	Buffer countBuffer =
	    Buffer::allocate(context.device(), updateCountByteCount(counts.size()));
	const bool didCount =
	    countBuffer.isValid() &&
	    launchSiddonBackProjectSingleRayUpdateCount(
	        context.device(), context.library(), context.commandQueue(),
	        batch.lorBuffer(), batch.projectionValuesBuffer(), countBuffer,
	        params, batch.size(), options) &&
	    countBuffer.copyToHost(counts.data(), updateCountByteCount(counts.size()));
	profile.adjointUpdateCountSeconds +=
	    getElapsedSeconds(countStart, Clock::now());
	if (!didCount)
	{
		return false;
	}

	std::size_t totalUpdates = 0;
	std::size_t raysWithUpdates = 0;
	std::uint32_t maxUpdates = 0;
	for (const std::uint32_t count : counts)
	{
		totalUpdates += count;
		if (count != 0)
		{
			raysWithUpdates += 1;
			maxUpdates = std::max(maxUpdates, count);
		}
	}

	profile.adjointVoxelUpdates += totalUpdates;
	profile.adjointRaysWithUpdates += raysWithUpdates;
	profile.adjointMaxUpdatesPerRay =
	    std::max<std::size_t>(profile.adjointMaxUpdatesPerRay, maxUpdates);
	return true;
}

bool collectBackProjectVoxelHitCounts(const Context& context,
    const ProjectionBatchMetal& batch, const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile& profile,
    const ProjectorKernelOptions* options)
{
	std::vector<std::uint32_t> counts(imageVoxelCount(params), 0);
	const auto countStart = Clock::now();
	Buffer countBuffer = Buffer::copyFromHost(
	    context.device(), counts.data(), updateCountByteCount(counts.size()));
	const bool didCount =
	    countBuffer.isValid() &&
	    launchSiddonBackProjectSingleRayVoxelHitCount(
	        context.device(), context.library(), context.commandQueue(),
	        batch.lorBuffer(), batch.projectionValuesBuffer(), countBuffer,
	        params, batch.size(), options) &&
	    countBuffer.copyToHost(counts.data(), updateCountByteCount(counts.size()));
	profile.adjointVoxelHitCountSeconds +=
	    getElapsedSeconds(countStart, Clock::now());
	if (!didCount)
	{
		return false;
	}

	std::vector<std::uint32_t> hitCounts;
	hitCounts.reserve(counts.size());
	std::size_t totalHits = 0;
	for (const std::uint32_t count : counts)
	{
		totalHits += count;
		if (count != 0)
		{
			hitCounts.push_back(count);
		}
	}
	std::sort(hitCounts.begin(), hitCounts.end());

	profile.adjointVoxelHitMaps += 1;
	profile.adjointBatchHitVoxels += hitCounts.size();
	profile.adjointVoxelHitTotalUpdates += totalHits;
	if (!hitCounts.empty())
	{
		profile.adjointMaxVoxelHits =
		    std::max<std::size_t>(profile.adjointMaxVoxelHits,
		                          hitCounts.back());
		profile.adjointMaxBatchP95VoxelHits =
		    std::max(profile.adjointMaxBatchP95VoxelHits,
		             percentileCount(hitCounts, 0.95));
		profile.adjointMaxBatchP99VoxelHits =
		    std::max(profile.adjointMaxBatchP99VoxelHits,
		             percentileCount(hitCounts, 0.99));
	}
	return true;
}

}  // namespace

bool makeSiddonForwardImageParams(const Image& image, std::uint32_t frame,
                                  SiddonForwardImageParams& params)
{
	if (!hasUsableImageMemory(image))
	{
		return false;
	}

	const ImageParams& imageParams = image.getParams();
	const auto frameCount = static_cast<std::uint32_t>(imageParams.nt);
	if (frame >= frameCount)
	{
		return false;
	}

	params = makeSiddonParams(imageParams, frame);
	return true;
}

bool uploadSiddonImageBuffer(const Context& context, const Image& image,
                             Buffer& imageBuffer,
                             SiddonProjectorKernelProfile* profile)
{
	SiddonForwardImageParams params{};
	if (!context.isValid() || !makeSiddonForwardImageParams(image, 0, params))
	{
		return false;
	}

	const auto imageUploadStart = Clock::now();
	imageBuffer = Buffer::copyFromHost(context.device(), image.getRawPointer(),
	    imageByteCount(params));
	if (profile != nullptr)
	{
		profile->imageUploadSeconds +=
		    getElapsedSeconds(imageUploadStart, Clock::now());
	}
	return imageBuffer.isValid();
}

bool downloadSiddonImageBuffer(const Buffer& imageBuffer, Image& image,
                               SiddonProjectorKernelProfile* profile)
{
	SiddonForwardImageParams params{};
	if (!makeSiddonForwardImageParams(image, 0, params))
	{
		return false;
	}

	const auto imageDownloadStart = Clock::now();
	const bool didCopy =
	    imageBuffer.copyToHost(image.getRawPointer(), imageByteCount(params));
	if (profile != nullptr)
	{
		profile->imageDownloadSeconds +=
		    getElapsedSeconds(imageDownloadStart, Clock::now());
	}
	return didCopy;
}

bool downloadSiddonImageBuffer(const Context& context,
                               const Buffer& imageBuffer, Image& image,
                               SiddonProjectorKernelProfile* profile)
{
	SiddonForwardImageParams params{};
	if (!context.isValid() || !makeSiddonForwardImageParams(image, 0, params))
	{
		return false;
	}

	const auto imageDownloadStart = Clock::now();
	const bool didCopy = imageBuffer.copyToHost(context.commandQueue(),
	    image.getRawPointer(), imageByteCount(params));
	if (profile != nullptr)
	{
		profile->imageDownloadSeconds +=
		    getElapsedSeconds(imageDownloadStart, Clock::now());
	}
	return didCopy;
}

bool forwardProjectSiddonSingleRay(const Context& context, const Image& image,
    ProjectionBatchMetal& batch, std::uint32_t frame,
    SiddonProjectorKernelProfile* profile,
    const ProjectorKernelOptions* options)
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

	return forwardProjectSiddonSingleRay(context, imageBuffer, batch, params,
	                                     profile, options);
}

bool forwardProjectSiddonSingleRay(const Context& context,
    const Buffer& imageBuffer, ProjectionBatchMetal& batch,
    const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile* profile,
    const ProjectorKernelOptions* options)
{
	if (!context.isValid() || !imageBuffer.isValid() || !batch.isValid() ||
	    !fitsUint32Size(batch.size()))
	{
		return false;
	}

	const auto kernelStart = Clock::now();
	const bool didRun = launchSiddonForwardSingleRay(
	    context.device(), context.library(), context.commandQueue(),
	    imageBuffer, batch.lorBuffer(), batch.projectionValuesBuffer(), params,
	    batch.size(), options);
	if (profile != nullptr)
	{
		profile->kernelSeconds += getElapsedSeconds(kernelStart, Clock::now());
	}
	return didRun;
}

bool backProjectSiddonSingleRay(const Context& context,
    const ProjectionBatchMetal& batch, Image& image, std::uint32_t frame,
    SiddonProjectorKernelProfile* profile,
    const ProjectorKernelOptions* options)
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

	return backProjectSiddonSingleRay(context, batch, imageBuffer, params,
	                                  profile, options) &&
	       downloadSiddonImageBuffer(context, imageBuffer, image, profile);
}

bool backProjectSiddonSingleRay(const Context& context,
    const ProjectionBatchMetal& batch, Buffer& imageBuffer,
    const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile* profile,
    const ProjectorKernelOptions* options)
{
	if (!context.isValid() || !imageBuffer.isValid() || !batch.isValid() ||
	    !fitsUint32Size(batch.size()))
	{
		return false;
	}

	const auto kernelStart = Clock::now();
	const bool didRun = launchSiddonBackProjectSingleRay(
	    context.device(), context.library(), context.commandQueue(), imageBuffer,
	    batch.lorBuffer(), batch.projectionValuesBuffer(), params,
	    batch.size(), options);
	if (profile != nullptr)
	{
		profile->kernelSeconds += getElapsedSeconds(kernelStart, Clock::now());
	}
	if (!didRun)
	{
		return false;
	}
	if (profile != nullptr && profile->diagnoseAdjointUpdateCounts &&
	    !collectBackProjectUpdateCounts(context, batch, params, *profile,
	        options))
	{
		return false;
	}
	if (profile != nullptr && profile->diagnoseAdjointVoxelHits &&
	    !collectBackProjectVoxelHitCounts(context, batch, params, *profile,
	        options))
	{
		return false;
	}
	return true;
}

}  // namespace yrt::backend::metal
