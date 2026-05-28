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

bool fitsUint32Size(std::size_t value)
{
	return value > 0 &&
	       value <= static_cast<std::size_t>(
	                    std::numeric_limits<std::uint32_t>::max());
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
    SiddonProjectorKernelProfile& profile)
{
	std::vector<std::uint32_t> counts(batch.size(), 0);
	const auto countStart = Clock::now();
	Buffer countBuffer =
	    Buffer::allocate(context.device(), updateCountByteCount(counts.size()));
	const bool didCount =
	    countBuffer.isValid() &&
	    launchJosephBackProjectSingleRayUpdateCount(
	        context.device(), context.library(), context.commandQueue(),
	        batch.lorBuffer(), batch.projectionValuesBuffer(), countBuffer,
	        params, batch.size()) &&
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
    SiddonProjectorKernelProfile& profile)
{
	std::vector<std::uint32_t> counts(imageVoxelCount(params), 0);
	const auto countStart = Clock::now();
	Buffer countBuffer = Buffer::copyFromHost(
	    context.device(), counts.data(), updateCountByteCount(counts.size()));
	const bool didCount =
	    countBuffer.isValid() &&
	    launchJosephBackProjectSingleRayVoxelHitCount(
	        context.device(), context.library(), context.commandQueue(),
	        batch.lorBuffer(), batch.projectionValuesBuffer(), countBuffer,
	        params, batch.size()) &&
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
	if (!didRun)
	{
		return false;
	}
	return true;
}

bool uploadJosephImageFrameTexture(const Context& context, const Image& image,
    std::uint32_t frame, Texture3D& texture, Sampler& sampler,
    SiddonProjectorKernelProfile* profile)
{
	if (!context.isValid())
	{
		return false;
	}

	SiddonForwardImageParams params{};
	if (!makeSiddonForwardImageParams(image, frame, params))
	{
		return false;
	}

	const std::size_t spatialCount = static_cast<std::size_t>(params.nx) *
	                                 static_cast<std::size_t>(params.ny) *
	                                 static_cast<std::size_t>(params.nz);
	const auto uploadStart = Clock::now();
	texture = Texture3D::allocateR32Float(context.device(), params.nx,
	    params.ny, params.nz);
	if (!sampler.isValid())
	{
		sampler = Sampler::createLinearClampToZero(context.device());
	}

	const float* framePtr = image.getRawPointer() +
	                        static_cast<std::size_t>(frame) * spatialCount;
	const bool didUpload =
	    texture.isValid() && sampler.isValid() &&
	    texture.copyFromHost(framePtr, sizeof(float) * params.nx,
	        sizeof(float) * static_cast<std::size_t>(params.nx) * params.ny);
	if (profile != nullptr)
	{
		profile->imageUploadSeconds +=
		    getElapsedSeconds(uploadStart, Clock::now());
	}
	return didUpload;
}

bool forwardProjectJosephSingleRayTexture(const Context& context,
    const Texture3D& imageTexture, const Sampler& sampler,
    ProjectionBatchMetal& batch, const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile* profile)
{
	if (!context.isValid() || !imageTexture.isValid() || !sampler.isValid() ||
	    !batch.isValid() || !fitsUint32Size(batch.size()))
	{
		return false;
	}

	const auto kernelStart = Clock::now();
	const bool didRun = launchJosephForwardSingleRayTexture(
	    context.device(), context.library(), context.commandQueue(),
	    imageTexture, sampler, batch.lorBuffer(),
	    batch.projectionValuesBuffer(), params, batch.size());
	if (profile != nullptr)
	{
		profile->kernelSeconds += getElapsedSeconds(kernelStart, Clock::now());
	}
	if (!didRun)
	{
		return false;
	}
	return true;
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
	if (!didRun)
	{
		return false;
	}
	if (profile != nullptr && profile->diagnoseAdjointUpdateCounts &&
	    !collectBackProjectUpdateCounts(context, batch, params, *profile))
	{
		return false;
	}
	if (profile != nullptr && profile->diagnoseAdjointVoxelHits &&
	    !collectBackProjectVoxelHitCounts(context, batch, params, *profile))
	{
		return false;
	}
	return true;
}

}  // namespace yrt::backend::metal
