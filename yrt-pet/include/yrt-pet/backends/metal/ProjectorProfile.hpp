/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cstddef>

namespace yrt::backend::metal
{

struct SiddonProjectorKernelProfile
{
	bool diagnoseAdjointUpdateCounts = false;
	bool diagnoseAdjointVoxelHits = false;
	double imageUploadSeconds = 0.0;
	double kernelSeconds = 0.0;
	double imageDownloadSeconds = 0.0;
	double adjointUpdateCountSeconds = 0.0;
	double adjointVoxelHitCountSeconds = 0.0;
	std::size_t adjointVoxelUpdates = 0;
	std::size_t adjointRaysWithUpdates = 0;
	std::size_t adjointMaxUpdatesPerRay = 0;
	std::size_t adjointVoxelHitMaps = 0;
	std::size_t adjointBatchHitVoxels = 0;
	std::size_t adjointVoxelHitTotalUpdates = 0;
	std::size_t adjointMaxVoxelHits = 0;
	std::size_t adjointMaxBatchP95VoxelHits = 0;
	std::size_t adjointMaxBatchP99VoxelHits = 0;
};

struct OperatorProjectorMetalProfile
{
	double forwardGatherSeconds = 0.0;
	double forwardGatherCacheBuildSeconds = 0.0;
	double forwardGatherUncachedSeconds = 0.0;
	double forwardGatherDirectSeconds = 0.0;
	double forwardGatherConstrainedSeconds = 0.0;
	double forwardPackSeconds = 0.0;
	double forwardPackCacheBuildSeconds = 0.0;
	double forwardPackUncachedSeconds = 0.0;
	double forwardBatchUploadSeconds = 0.0;
	double forwardBatchUploadCacheBuildSeconds = 0.0;
	double forwardBatchUploadUncachedSeconds = 0.0;
	double forwardImageUploadSeconds = 0.0;
	double forwardKernelSeconds = 0.0;
	double forwardDownloadSeconds = 0.0;
	double forwardHostWriteSeconds = 0.0;

	double ratioPackSeconds = 0.0;
	double ratioBatchUploadSeconds = 0.0;
	double ratioKernelSeconds = 0.0;

	double adjointGatherSeconds = 0.0;
	double adjointGatherCacheBuildSeconds = 0.0;
	double adjointGatherUncachedSeconds = 0.0;
	double adjointGatherDirectSeconds = 0.0;
	double adjointGatherConstrainedSeconds = 0.0;
	double adjointPackSeconds = 0.0;
	double adjointPackCacheBuildSeconds = 0.0;
	double adjointPackUncachedSeconds = 0.0;
	double adjointBatchUploadSeconds = 0.0;
	double adjointBatchUploadCacheBuildSeconds = 0.0;
	double adjointBatchUploadUncachedSeconds = 0.0;
	double adjointImageUploadSeconds = 0.0;
	double adjointKernelSeconds = 0.0;
	double adjointImageDownloadSeconds = 0.0;
	double adjointHostImageCopySeconds = 0.0;
	double adjointUpdateCountSeconds = 0.0;
	double adjointVoxelHitCountSeconds = 0.0;

	bool diagnoseAdjointUpdateCounts = false;
	bool diagnoseAdjointVoxelHits = false;
	std::size_t forwardEvents = 0;
	std::size_t forwardBatches = 0;
	std::size_t adjointEvents = 0;
	std::size_t adjointNonzeroEvents = 0;
	std::size_t adjointBatches = 0;
	std::size_t adjointVoxelUpdates = 0;
	std::size_t adjointRaysWithUpdates = 0;
	std::size_t adjointMaxUpdatesPerRay = 0;
	std::size_t adjointVoxelHitMaps = 0;
	std::size_t adjointBatchHitVoxels = 0;
	std::size_t adjointVoxelHitTotalUpdates = 0;
	std::size_t adjointMaxVoxelHits = 0;
	std::size_t adjointMaxBatchP95VoxelHits = 0;
	std::size_t adjointMaxBatchP99VoxelHits = 0;
	std::size_t cacheLookups = 0;
	std::size_t cacheHits = 0;
	std::size_t cacheMisses = 0;
	std::size_t cacheBuilds = 0;
	std::size_t cacheSkipsOverBudget = 0;
	std::size_t cacheUsedBytes = 0;
	std::size_t cacheMaxBytes = 0;
	std::size_t uncachedBatches = 0;
};

}  // namespace yrt::backend::metal
