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
constexpr std::uint32_t kVoxelHitTileSize = 8;

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

template <typename T>
std::size_t percentileCount(const std::vector<T>& sortedCounts,
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

template <typename T>
double topCountFraction(const std::vector<T>& sortedCounts,
                        double topFraction, std::size_t totalCount)
{
	if (sortedCounts.empty() || totalCount == 0)
	{
		return 0.0;
	}
	const std::size_t count =
	    std::max<std::size_t>(1,
	        static_cast<std::size_t>(
	            std::ceil(topFraction *
	                      static_cast<double>(sortedCounts.size()))));
	const std::size_t clampedCount = std::min(count, sortedCounts.size());
	std::size_t topTotal = 0;
	for (auto it = sortedCounts.rbegin();
	     it != sortedCounts.rend() &&
	     static_cast<std::size_t>(it - sortedCounts.rbegin()) < clampedCount;
	     ++it)
	{
		topTotal += static_cast<std::size_t>(*it);
	}
	return static_cast<double>(topTotal) /
	       static_cast<double>(totalCount);
}

std::vector<std::size_t> collectTileCounts(
    const std::vector<std::uint32_t>& counts,
    const SiddonForwardImageParams& params, std::uint32_t tileSize)
{
	const std::size_t nx = params.nx;
	const std::size_t ny = params.ny;
	const std::size_t nz = params.nz;
	const std::size_t nt = params.nt;
	const std::size_t spatialCount = nx * ny * nz;
	const std::size_t tileNx = (nx + tileSize - 1) / tileSize;
	const std::size_t tileNy = (ny + tileSize - 1) / tileSize;
	const std::size_t tileNz = (nz + tileSize - 1) / tileSize;
	const std::size_t tileSpatialCount = tileNx * tileNy * tileNz;
	std::vector<std::size_t> tileCounts(tileSpatialCount * nt, 0);

	for (std::size_t index = 0; index < counts.size(); ++index)
	{
		const std::uint32_t count = counts[index];
		if (count == 0 || spatialCount == 0)
		{
			continue;
		}

		const std::size_t frame = index / spatialCount;
		const std::size_t spatialIndex = index % spatialCount;
		const std::size_t z = spatialIndex / (nx * ny);
		const std::size_t xy = spatialIndex - z * nx * ny;
		const std::size_t y = xy / nx;
		const std::size_t x = xy - y * nx;
		const std::size_t tileX = x / tileSize;
		const std::size_t tileY = y / tileSize;
		const std::size_t tileZ = z / tileSize;
		const std::size_t tileIndex =
		    frame * tileSpatialCount +
		    tileX + tileNx * (tileY + tileNy * tileZ);
		tileCounts[tileIndex] += count;
	}
	return tileCounts;
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
	    launchJosephBackProjectSingleRayUpdateCount(
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
	    launchJosephBackProjectSingleRayVoxelHitCount(
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
		profile.adjointMaxBatchP50VoxelHits =
		    std::max(profile.adjointMaxBatchP50VoxelHits,
		             percentileCount(hitCounts, 0.50));
		profile.adjointMaxBatchP90VoxelHits =
		    std::max(profile.adjointMaxBatchP90VoxelHits,
		             percentileCount(hitCounts, 0.90));
		profile.adjointMaxBatchP95VoxelHits =
		    std::max(profile.adjointMaxBatchP95VoxelHits,
		             percentileCount(hitCounts, 0.95));
		profile.adjointMaxBatchP99VoxelHits =
		    std::max(profile.adjointMaxBatchP99VoxelHits,
		             percentileCount(hitCounts, 0.99));
		profile.adjointMaxBatchP999VoxelHits =
		    std::max(profile.adjointMaxBatchP999VoxelHits,
		             percentileCount(hitCounts, 0.999));
		profile.adjointMaxBatchMeanVoxelHits =
		    std::max(profile.adjointMaxBatchMeanVoxelHits,
		             static_cast<double>(totalHits) /
		                 static_cast<double>(hitCounts.size()));
		profile.adjointMaxBatchTop1PctVoxelHitFraction =
		    std::max(profile.adjointMaxBatchTop1PctVoxelHitFraction,
		             topCountFraction(hitCounts, 0.01, totalHits));
		profile.adjointMaxBatchTop01PctVoxelHitFraction =
		    std::max(profile.adjointMaxBatchTop01PctVoxelHitFraction,
		             topCountFraction(hitCounts, 0.001, totalHits));
	}

	std::vector<std::size_t> tileCounts =
	    collectTileCounts(counts, params, kVoxelHitTileSize);
	std::vector<std::size_t> hitTileCounts;
	hitTileCounts.reserve(tileCounts.size());
	std::size_t totalTileHits = 0;
	for (const std::size_t count : tileCounts)
	{
		totalTileHits += count;
		if (count != 0)
		{
			hitTileCounts.push_back(count);
		}
	}
	std::sort(hitTileCounts.begin(), hitTileCounts.end());

	profile.adjointTileSize = kVoxelHitTileSize;
	profile.adjointVoxelHitTiles += hitTileCounts.size();
	profile.adjointVoxelHitTileTotalUpdates += totalTileHits;
	if (!hitTileCounts.empty())
	{
		profile.adjointMaxTileHits =
		    std::max(profile.adjointMaxTileHits, hitTileCounts.back());
		profile.adjointMaxBatchP95TileHits =
		    std::max(profile.adjointMaxBatchP95TileHits,
		             percentileCount(hitTileCounts, 0.95));
		profile.adjointMaxBatchP99TileHits =
		    std::max(profile.adjointMaxBatchP99TileHits,
		             percentileCount(hitTileCounts, 0.99));
		profile.adjointMaxBatchMeanTileHits =
		    std::max(profile.adjointMaxBatchMeanTileHits,
		             static_cast<double>(totalTileHits) /
		                 static_cast<double>(hitTileCounts.size()));
		profile.adjointMaxBatchTop1PctTileHitFraction =
		    std::max(profile.adjointMaxBatchTop1PctTileHitFraction,
		             topCountFraction(hitTileCounts, 0.01, totalTileHits));
		profile.adjointMaxBatchTop01PctTileHitFraction =
		    std::max(profile.adjointMaxBatchTop01PctTileHitFraction,
		             topCountFraction(hitTileCounts, 0.001, totalTileHits));
	}
	return true;
}

}  // namespace

bool forwardProjectJosephSingleRay(const Context& context, const Image& image,
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

	return forwardProjectJosephSingleRay(context, imageBuffer, batch, params,
	                                     profile, options);
}

bool forwardProjectJosephSingleRay(const Context& context,
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
	const bool didRun = launchJosephForwardSingleRay(
	    context.device(), context.library(), context.commandQueue(),
	    imageBuffer, batch.lorBuffer(), batch.projectionValuesBuffer(), params,
	    batch.size(), options);
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

bool forwardProjectJosephSingleRayAxis(const Context& context,
    const Buffer& imageBuffer, ProjectionBatchMetal& batch,
    const SiddonForwardImageParams& params, std::uint32_t axis,
    SiddonProjectorKernelProfile* profile,
    const ProjectorKernelOptions* options)
{
	if (!context.isValid() || !imageBuffer.isValid() || !batch.isValid() ||
	    !fitsUint32Size(batch.size()))
	{
		return false;
	}

	const auto kernelStart = Clock::now();
	const bool didRun = launchJosephForwardSingleRayAxis(
	    context.device(), context.library(), context.commandQueue(),
	    imageBuffer, batch.lorBuffer(), batch.projectionValuesBuffer(), params,
	    batch.size(), axis, options);
	if (profile != nullptr)
	{
		profile->kernelSeconds += getElapsedSeconds(kernelStart, Clock::now());
	}
	return didRun;
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
    SiddonProjectorKernelProfile* profile,
    const ProjectorKernelOptions* options)
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
	    batch.projectionValuesBuffer(), params, batch.size(), options);
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

	return backProjectJosephSingleRay(context, batch, imageBuffer, params,
	                                  profile, options) &&
	       downloadSiddonImageBuffer(context, imageBuffer, image, profile);
}

bool backProjectJosephSingleRay(const Context& context,
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
	const bool didRun = launchJosephBackProjectSingleRay(
	    context.device(), context.library(), context.commandQueue(),
	    imageBuffer, batch.lorBuffer(), batch.projectionValuesBuffer(), params,
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

bool backProjectJosephSingleRayAxis(const Context& context,
    const ProjectionBatchMetal& batch, Buffer& imageBuffer,
    const SiddonForwardImageParams& params, std::uint32_t axis,
    SiddonProjectorKernelProfile* profile,
    const ProjectorKernelOptions* options)
{
	if (!context.isValid() || !imageBuffer.isValid() || !batch.isValid() ||
	    !fitsUint32Size(batch.size()))
	{
		return false;
	}

	const auto kernelStart = Clock::now();
	const bool didRun = launchJosephBackProjectSingleRayAxis(
	    context.device(), context.library(), context.commandQueue(),
	    imageBuffer, batch.lorBuffer(), batch.projectionValuesBuffer(), params,
	    batch.size(), axis, options);
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
