/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/OSEM_CPU.hpp"

#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"
#include "yrt-pet/operators/OperatorVarPsf.hpp"
#include "yrt-pet/operators/ProjectorDD.hpp"
#include "yrt-pet/operators/ProjectorSiddon.hpp"
#include "yrt-pet/recon/Corrector_CPU.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#if BUILD_METAL
#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/OperatorProjectorMetalBridge.hpp"
#endif

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_osem_cpu(pybind11::module& m)
{
	auto c = py::class_<OSEM_CPU, OSEM>(m, "OSEM_CPU");
	c.def("setExperimentalMetalProjectorEnabled",
	      &OSEM_CPU::setExperimentalMetalProjectorEnabled, "enabled"_a);
	c.def("isExperimentalMetalProjectorEnabled",
	      &OSEM_CPU::isExperimentalMetalProjectorEnabled);
	c.def("didLastExperimentalMetalProjectorRun",
	      &OSEM_CPU::didLastExperimentalMetalProjectorRun);
	c.def("setExperimentalMetalProjectorFusedRatioEnabled",
	      &OSEM_CPU::setExperimentalMetalProjectorFusedRatioEnabled,
	      "enabled"_a);
	c.def("isExperimentalMetalProjectorFusedRatioEnabled",
	      &OSEM_CPU::isExperimentalMetalProjectorFusedRatioEnabled);
	c.def("setExperimentalMetalProjectorKernel",
	      &OSEM_CPU::setExperimentalMetalProjectorKernel, "kernel"_a);
	c.def("getExperimentalMetalProjectorKernel",
	      &OSEM_CPU::getExperimentalMetalProjectorKernel);
	c.def("setExperimentalMetalProjectorProfilingEnabled",
	      &OSEM_CPU::setExperimentalMetalProjectorProfilingEnabled,
	      "enabled"_a);
	c.def("isExperimentalMetalProjectorProfilingEnabled",
	      &OSEM_CPU::isExperimentalMetalProjectorProfilingEnabled);
	c.def("setExperimentalMetalProjectorAdjointDiagnosticsEnabled",
	      &OSEM_CPU::setExperimentalMetalProjectorAdjointDiagnosticsEnabled,
	      "enabled"_a);
	c.def("isExperimentalMetalProjectorAdjointDiagnosticsEnabled",
	      &OSEM_CPU::isExperimentalMetalProjectorAdjointDiagnosticsEnabled);
	c.def("setExperimentalMetalProjectorAdjointHitDiagnosticsEnabled",
	      &OSEM_CPU::setExperimentalMetalProjectorAdjointHitDiagnosticsEnabled,
	      "enabled"_a);
	c.def("isExperimentalMetalProjectorAdjointHitDiagnosticsEnabled",
	      &OSEM_CPU::isExperimentalMetalProjectorAdjointHitDiagnosticsEnabled);
	c.def("resetExperimentalMetalProjectorTimings",
	      &OSEM_CPU::resetExperimentalMetalProjectorTimings);
	c.def("getExperimentalMetalProjectorTimings",
	      [](const OSEM_CPU& self)
	      {
		      const auto timings =
		          self.getExperimentalMetalProjectorTimings();
		      py::dict result;
		      result["setup_s"] = timings.setupSeconds;
		      result["forward_s"] = timings.forwardSeconds;
		      result["ratio_s"] = timings.ratioSeconds;
		      result["adjoint_s"] = timings.adjointSeconds;
		      result["total_s"] = timings.totalSeconds;
		      result["calls"] = timings.calls;
		      result["forward_gather_s"] = timings.forwardGatherSeconds;
		      result["forward_gather_cache_build_s"] =
		          timings.forwardGatherCacheBuildSeconds;
		      result["forward_gather_uncached_s"] =
		          timings.forwardGatherUncachedSeconds;
		      result["forward_gather_direct_s"] =
		          timings.forwardGatherDirectSeconds;
		      result["forward_gather_constrained_s"] =
		          timings.forwardGatherConstrainedSeconds;
		      result["forward_pack_s"] = timings.forwardPackSeconds;
		      result["forward_pack_cache_build_s"] =
		          timings.forwardPackCacheBuildSeconds;
		      result["forward_pack_uncached_s"] =
		          timings.forwardPackUncachedSeconds;
		      result["forward_batch_upload_s"] =
		          timings.forwardBatchUploadSeconds;
		      result["forward_batch_upload_cache_build_s"] =
		          timings.forwardBatchUploadCacheBuildSeconds;
		      result["forward_batch_upload_uncached_s"] =
		          timings.forwardBatchUploadUncachedSeconds;
		      result["forward_image_upload_s"] =
		          timings.forwardImageUploadSeconds;
		      result["forward_kernel_s"] = timings.forwardKernelSeconds;
		      result["forward_download_s"] = timings.forwardDownloadSeconds;
		      result["forward_host_write_s"] =
		          timings.forwardHostWriteSeconds;
		      result["ratio_pack_s"] = timings.ratioPackSeconds;
		      result["ratio_batch_upload_s"] =
		          timings.ratioBatchUploadSeconds;
		      result["ratio_kernel_s"] = timings.ratioKernelSeconds;
		      result["adjoint_gather_s"] = timings.adjointGatherSeconds;
		      result["adjoint_gather_cache_build_s"] =
		          timings.adjointGatherCacheBuildSeconds;
		      result["adjoint_gather_uncached_s"] =
		          timings.adjointGatherUncachedSeconds;
		      result["adjoint_gather_direct_s"] =
		          timings.adjointGatherDirectSeconds;
		      result["adjoint_gather_constrained_s"] =
		          timings.adjointGatherConstrainedSeconds;
		      result["adjoint_pack_s"] = timings.adjointPackSeconds;
		      result["adjoint_pack_cache_build_s"] =
		          timings.adjointPackCacheBuildSeconds;
		      result["adjoint_pack_uncached_s"] =
		          timings.adjointPackUncachedSeconds;
		      result["adjoint_batch_upload_s"] =
		          timings.adjointBatchUploadSeconds;
		      result["adjoint_batch_upload_cache_build_s"] =
		          timings.adjointBatchUploadCacheBuildSeconds;
		      result["adjoint_batch_upload_uncached_s"] =
		          timings.adjointBatchUploadUncachedSeconds;
		      result["adjoint_image_upload_s"] =
		          timings.adjointImageUploadSeconds;
		      result["adjoint_kernel_s"] = timings.adjointKernelSeconds;
		      result["adjoint_image_download_s"] =
		          timings.adjointImageDownloadSeconds;
		      result["adjoint_host_image_copy_s"] =
		          timings.adjointHostImageCopySeconds;
		      result["adjoint_update_count_s"] =
		          timings.adjointUpdateCountSeconds;
		      result["adjoint_voxel_hit_count_s"] =
		          timings.adjointVoxelHitCountSeconds;
		      result["forward_events"] = timings.forwardEvents;
		      result["forward_batches"] = timings.forwardBatches;
		      result["adjoint_events"] = timings.adjointEvents;
		      result["adjoint_nonzero_events"] =
		          timings.adjointNonzeroEvents;
		      result["adjoint_batches"] = timings.adjointBatches;
		      result["adjoint_voxel_updates"] =
		          timings.adjointVoxelUpdates;
		      result["adjoint_rays_with_updates"] =
		          timings.adjointRaysWithUpdates;
		      result["adjoint_max_updates_per_ray"] =
		          timings.adjointMaxUpdatesPerRay;
		      result["adjoint_voxel_hit_maps"] =
		          timings.adjointVoxelHitMaps;
		      result["adjoint_batch_hit_voxels"] =
		          timings.adjointBatchHitVoxels;
		      result["adjoint_voxel_hit_total_updates"] =
		          timings.adjointVoxelHitTotalUpdates;
		      result["adjoint_max_voxel_hits"] =
		          timings.adjointMaxVoxelHits;
		      result["adjoint_max_batch_p95_voxel_hits"] =
		          timings.adjointMaxBatchP95VoxelHits;
		      result["adjoint_max_batch_p99_voxel_hits"] =
		          timings.adjointMaxBatchP99VoxelHits;
		      result["cache_lookups"] = timings.cacheLookups;
		      result["cache_hits"] = timings.cacheHits;
		      result["cache_misses"] = timings.cacheMisses;
		      result["cache_builds"] = timings.cacheBuilds;
		      result["cache_skips_over_budget"] =
		          timings.cacheSkipsOverBudget;
		      result["cache_used_bytes"] = timings.cacheUsedBytes;
		      result["cache_max_bytes"] = timings.cacheMaxBytes;
		      result["uncached_batches"] = timings.uncachedBatches;
		      result["uncached_chunks"] = timings.uncachedBatches;
		      return result;
	      });
	c.def("getExperimentalMetalProjectorSubsetTimings",
	      [](const OSEM_CPU& self)
	      {
		      auto memoryToDict =
		          [](const OSEM_CPU::ExperimentalMetalProjectorMemorySnapshot&
		                 snapshot)
		      {
			      py::dict result;
			      result["available"] = snapshot.available;
			      result["total_bytes"] = snapshot.totalBytes;
			      result["available_bytes"] = snapshot.availableBytes;
			      result["used_bytes"] = snapshot.usedBytes;
			      result["free_bytes"] = snapshot.freeBytes;
			      result["speculative_bytes"] = snapshot.speculativeBytes;
			      result["active_bytes"] = snapshot.activeBytes;
			      result["inactive_bytes"] = snapshot.inactiveBytes;
			      result["wired_bytes"] = snapshot.wiredBytes;
			      result["compressed_bytes"] = snapshot.compressedBytes;
			      result["pageins"] = snapshot.pageins;
			      result["pageouts"] = snapshot.pageouts;
			      result["decompressions"] = snapshot.decompressions;
			      result["compressions"] = snapshot.compressions;
			      result["swapins"] = snapshot.swapins;
			      result["swapouts"] = snapshot.swapouts;
			      result["available_ratio"] = snapshot.availableRatio;
			      result["free_ratio"] = snapshot.freeRatio;
			      result["compressed_ratio"] = snapshot.compressedRatio;
			      result["pressure_level"] = snapshot.pressureLevel;
			      return result;
		      };

		      py::list result;
		      for (const auto& timing :
		           self.getExperimentalMetalProjectorSubsetTimings())
		      {
			      py::dict row;
			      row["iteration_index"] = timing.iteration;
			      row["subset_index"] = timing.subset;
			      row["iteration"] = timing.iteration + 1;
			      row["subset"] = timing.subset + 1;
			      row["events"] = timing.events;
			      row["metal_ran"] = timing.metalRan;
			      row["setup_s"] = timing.setupSeconds;
			      row["forward_s"] = timing.forwardSeconds;
			      row["ratio_s"] = timing.ratioSeconds;
			      row["adjoint_s"] = timing.adjointSeconds;
			      row["total_s"] = timing.totalSeconds;
			      row["forward_gather_s"] = timing.forwardGatherSeconds;
			      row["forward_gather_cache_build_s"] =
			          timing.forwardGatherCacheBuildSeconds;
			      row["forward_gather_uncached_s"] =
			          timing.forwardGatherUncachedSeconds;
			      row["forward_pack_s"] = timing.forwardPackSeconds;
			      row["forward_batch_upload_s"] =
			          timing.forwardBatchUploadSeconds;
			      row["forward_image_upload_s"] =
			          timing.forwardImageUploadSeconds;
			      row["forward_kernel_s"] = timing.forwardKernelSeconds;
			      row["forward_download_s"] = timing.forwardDownloadSeconds;
			      row["ratio_pack_s"] = timing.ratioPackSeconds;
			      row["ratio_batch_upload_s"] =
			          timing.ratioBatchUploadSeconds;
			      row["ratio_kernel_s"] = timing.ratioKernelSeconds;
			      row["adjoint_batch_upload_s"] =
			          timing.adjointBatchUploadSeconds;
			      row["adjoint_image_upload_s"] =
			          timing.adjointImageUploadSeconds;
			      row["adjoint_kernel_s"] = timing.adjointKernelSeconds;
			      row["adjoint_image_download_s"] =
			          timing.adjointImageDownloadSeconds;
			      row["adjoint_host_image_copy_s"] =
			          timing.adjointHostImageCopySeconds;
			      row["forward_events"] = timing.forwardEvents;
			      row["forward_batches"] = timing.forwardBatches;
			      row["adjoint_events"] = timing.adjointEvents;
			      row["adjoint_batches"] = timing.adjointBatches;
			      row["cache_lookups"] = timing.cacheLookups;
			      row["cache_hits"] = timing.cacheHits;
			      row["cache_misses"] = timing.cacheMisses;
			      row["cache_builds"] = timing.cacheBuilds;
			      row["cache_skips_over_budget"] =
			          timing.cacheSkipsOverBudget;
			      row["cache_used_bytes"] = timing.cacheUsedBytes;
			      row["cache_max_bytes"] = timing.cacheMaxBytes;
			      row["uncached_batches"] = timing.uncachedBatches;
			      row["memory_before"] = memoryToDict(timing.memoryBefore);
			      row["memory_after"] = memoryToDict(timing.memoryAfter);
			      result.append(row);
		      }
		      return result;
	      });
	c.def("setExperimentalMetalProjectorCacheEnabled",
	      &OSEM_CPU::setExperimentalMetalProjectorCacheEnabled, "enabled"_a);
	c.def("isExperimentalMetalProjectorCacheEnabled",
	      &OSEM_CPU::isExperimentalMetalProjectorCacheEnabled);
	c.def("setExperimentalMetalProjectorCacheMaxBytes",
	      &OSEM_CPU::setExperimentalMetalProjectorCacheMaxBytes,
	      "max_bytes"_a);
	c.def("getExperimentalMetalProjectorCacheMaxBytes",
	      &OSEM_CPU::getExperimentalMetalProjectorCacheMaxBytes);
	c.def("setExperimentalMetalProjectorMaxBatchEvents",
	      &OSEM_CPU::setExperimentalMetalProjectorMaxBatchEvents,
	      "max_batch_events"_a);
	c.def("getExperimentalMetalProjectorMaxBatchEvents",
	      &OSEM_CPU::getExperimentalMetalProjectorMaxBatchEvents);
	c.def("setExperimentalMetalProjectorMaxChunkEvents",
	      &OSEM_CPU::setExperimentalMetalProjectorMaxChunkEvents,
	      "max_chunk_events"_a);
	c.def("getExperimentalMetalProjectorMaxChunkEvents",
	      &OSEM_CPU::getExperimentalMetalProjectorMaxChunkEvents);
}
}  // namespace yrt
#endif

namespace yrt
{
namespace
{

#if BUILD_METAL
using Clock = std::chrono::steady_clock;

double getElapsedSeconds(Clock::time_point start, Clock::time_point end)
{
	return std::chrono::duration<double>(end - start).count();
}

OSEM_CPU::ExperimentalMetalProjectorMemorySnapshot
sampleExperimentalMetalMemory()
{
	OSEM_CPU::ExperimentalMetalProjectorMemorySnapshot snapshot;
	snapshot.pressureLevel = "unavailable";

#if defined(__APPLE__)
	vm_size_t pageSize = 0;
	const host_t host = mach_host_self();
	if (host_page_size(host, &pageSize) != KERN_SUCCESS)
	{
		return snapshot;
	}

	vm_statistics64_data_t vmStats{};
	mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
	if (host_statistics64(host, HOST_VM_INFO64,
	        reinterpret_cast<host_info64_t>(&vmStats), &count) != KERN_SUCCESS)
	{
		return snapshot;
	}

	std::uint64_t totalBytes = 0;
	std::size_t totalBytesSize = sizeof(totalBytes);
	if (sysctlbyname("hw.memsize", &totalBytes, &totalBytesSize, nullptr, 0) !=
	    0)
	{
		totalBytes = 0;
	}

	const auto pagesToBytes = [pageSize](auto pages) -> size_t
	{
		return static_cast<size_t>(pages) * static_cast<size_t>(pageSize);
	};

	snapshot.available = true;
	snapshot.totalBytes = static_cast<size_t>(totalBytes);
	snapshot.freeBytes = pagesToBytes(vmStats.free_count);
	snapshot.speculativeBytes = pagesToBytes(vmStats.speculative_count);
	snapshot.activeBytes = pagesToBytes(vmStats.active_count);
	snapshot.inactiveBytes = pagesToBytes(vmStats.inactive_count);
	snapshot.wiredBytes = pagesToBytes(vmStats.wire_count);
	snapshot.compressedBytes = pagesToBytes(vmStats.compressor_page_count);
	snapshot.pageins = static_cast<size_t>(vmStats.pageins);
	snapshot.pageouts = static_cast<size_t>(vmStats.pageouts);
	snapshot.decompressions = static_cast<size_t>(vmStats.decompressions);
	snapshot.compressions = static_cast<size_t>(vmStats.compressions);
	snapshot.swapins = static_cast<size_t>(vmStats.swapins);
	snapshot.swapouts = static_cast<size_t>(vmStats.swapouts);

	// free_count already includes speculative_count on macOS.
	snapshot.availableBytes = snapshot.freeBytes + snapshot.inactiveBytes;
	if (snapshot.totalBytes > 0)
	{
		snapshot.usedBytes =
		    snapshot.totalBytes > snapshot.availableBytes ?
		        snapshot.totalBytes - snapshot.availableBytes :
		        0;
		snapshot.availableRatio =
		    static_cast<double>(snapshot.availableBytes) /
		    static_cast<double>(snapshot.totalBytes);
		snapshot.freeRatio = static_cast<double>(snapshot.freeBytes) /
		                     static_cast<double>(snapshot.totalBytes);
		snapshot.compressedRatio =
		    static_cast<double>(snapshot.compressedBytes) /
		    static_cast<double>(snapshot.totalBytes);
	}

	if (snapshot.availableRatio < 0.08 ||
	    snapshot.compressedRatio > 0.65 ||
	    (snapshot.freeRatio < 0.01 && snapshot.compressedRatio > 0.55))
	{
		snapshot.pressureLevel = "red";
	}
	else if (snapshot.availableRatio < 0.16 ||
	         snapshot.freeRatio < 0.05 ||
	         snapshot.compressedRatio > 0.25)
	{
		snapshot.pressureLevel = "yellow";
	}
	else
	{
		snapshot.pressureLevel = "green";
	}
#endif

	return snapshot;
}

void addBridgeProfileToTimings(
    OSEM_CPU::ExperimentalMetalProjectorTimings& timings, double setupSeconds,
    double forwardSeconds, double ratioSeconds, double adjointSeconds,
    double totalSeconds,
    const backend::metal::OperatorProjectorMetalProfile& bridgeProfile)
{
	timings.setupSeconds += setupSeconds;
	timings.forwardSeconds += forwardSeconds;
	timings.ratioSeconds += ratioSeconds;
	timings.adjointSeconds += adjointSeconds;
	timings.totalSeconds += totalSeconds;
	timings.calls += 1;
	timings.forwardGatherSeconds += bridgeProfile.forwardGatherSeconds;
	timings.forwardGatherCacheBuildSeconds +=
	    bridgeProfile.forwardGatherCacheBuildSeconds;
	timings.forwardGatherUncachedSeconds +=
	    bridgeProfile.forwardGatherUncachedSeconds;
	timings.forwardGatherDirectSeconds +=
	    bridgeProfile.forwardGatherDirectSeconds;
	timings.forwardGatherConstrainedSeconds +=
	    bridgeProfile.forwardGatherConstrainedSeconds;
	timings.forwardPackSeconds += bridgeProfile.forwardPackSeconds;
	timings.forwardPackCacheBuildSeconds +=
	    bridgeProfile.forwardPackCacheBuildSeconds;
	timings.forwardPackUncachedSeconds +=
	    bridgeProfile.forwardPackUncachedSeconds;
	timings.forwardBatchUploadSeconds +=
	    bridgeProfile.forwardBatchUploadSeconds;
	timings.forwardBatchUploadCacheBuildSeconds +=
	    bridgeProfile.forwardBatchUploadCacheBuildSeconds;
	timings.forwardBatchUploadUncachedSeconds +=
	    bridgeProfile.forwardBatchUploadUncachedSeconds;
	timings.forwardImageUploadSeconds += bridgeProfile.forwardImageUploadSeconds;
	timings.forwardKernelSeconds += bridgeProfile.forwardKernelSeconds;
	timings.forwardDownloadSeconds += bridgeProfile.forwardDownloadSeconds;
	timings.forwardHostWriteSeconds += bridgeProfile.forwardHostWriteSeconds;
	timings.ratioPackSeconds += bridgeProfile.ratioPackSeconds;
	timings.ratioBatchUploadSeconds += bridgeProfile.ratioBatchUploadSeconds;
	timings.ratioKernelSeconds += bridgeProfile.ratioKernelSeconds;
	timings.adjointGatherSeconds += bridgeProfile.adjointGatherSeconds;
	timings.adjointGatherCacheBuildSeconds +=
	    bridgeProfile.adjointGatherCacheBuildSeconds;
	timings.adjointGatherUncachedSeconds +=
	    bridgeProfile.adjointGatherUncachedSeconds;
	timings.adjointGatherDirectSeconds +=
	    bridgeProfile.adjointGatherDirectSeconds;
	timings.adjointGatherConstrainedSeconds +=
	    bridgeProfile.adjointGatherConstrainedSeconds;
	timings.adjointPackSeconds += bridgeProfile.adjointPackSeconds;
	timings.adjointPackCacheBuildSeconds +=
	    bridgeProfile.adjointPackCacheBuildSeconds;
	timings.adjointPackUncachedSeconds +=
	    bridgeProfile.adjointPackUncachedSeconds;
	timings.adjointBatchUploadSeconds +=
	    bridgeProfile.adjointBatchUploadSeconds;
	timings.adjointBatchUploadCacheBuildSeconds +=
	    bridgeProfile.adjointBatchUploadCacheBuildSeconds;
	timings.adjointBatchUploadUncachedSeconds +=
	    bridgeProfile.adjointBatchUploadUncachedSeconds;
	timings.adjointImageUploadSeconds += bridgeProfile.adjointImageUploadSeconds;
	timings.adjointKernelSeconds += bridgeProfile.adjointKernelSeconds;
	timings.adjointImageDownloadSeconds +=
	    bridgeProfile.adjointImageDownloadSeconds;
	timings.adjointHostImageCopySeconds +=
	    bridgeProfile.adjointHostImageCopySeconds;
	timings.adjointUpdateCountSeconds +=
	    bridgeProfile.adjointUpdateCountSeconds;
	timings.adjointVoxelHitCountSeconds +=
	    bridgeProfile.adjointVoxelHitCountSeconds;
	timings.forwardEvents += bridgeProfile.forwardEvents;
	timings.forwardBatches += bridgeProfile.forwardBatches;
	timings.adjointEvents += bridgeProfile.adjointEvents;
	timings.adjointNonzeroEvents += bridgeProfile.adjointNonzeroEvents;
	timings.adjointBatches += bridgeProfile.adjointBatches;
	timings.adjointVoxelUpdates += bridgeProfile.adjointVoxelUpdates;
	timings.adjointRaysWithUpdates += bridgeProfile.adjointRaysWithUpdates;
	timings.adjointMaxUpdatesPerRay =
	    std::max(timings.adjointMaxUpdatesPerRay,
	             static_cast<size_t>(bridgeProfile.adjointMaxUpdatesPerRay));
	timings.adjointVoxelHitMaps += bridgeProfile.adjointVoxelHitMaps;
	timings.adjointBatchHitVoxels += bridgeProfile.adjointBatchHitVoxels;
	timings.adjointVoxelHitTotalUpdates +=
	    bridgeProfile.adjointVoxelHitTotalUpdates;
	timings.adjointMaxVoxelHits =
	    std::max(timings.adjointMaxVoxelHits,
	             static_cast<size_t>(bridgeProfile.adjointMaxVoxelHits));
	timings.adjointMaxBatchP95VoxelHits =
	    std::max(timings.adjointMaxBatchP95VoxelHits,
	             static_cast<size_t>(
	                 bridgeProfile.adjointMaxBatchP95VoxelHits));
	timings.adjointMaxBatchP99VoxelHits =
	    std::max(timings.adjointMaxBatchP99VoxelHits,
	             static_cast<size_t>(
	                 bridgeProfile.adjointMaxBatchP99VoxelHits));
	timings.cacheLookups += bridgeProfile.cacheLookups;
	timings.cacheHits += bridgeProfile.cacheHits;
	timings.cacheMisses += bridgeProfile.cacheMisses;
	timings.cacheBuilds += bridgeProfile.cacheBuilds;
	timings.cacheSkipsOverBudget += bridgeProfile.cacheSkipsOverBudget;
	timings.cacheUsedBytes = bridgeProfile.cacheUsedBytes;
	timings.cacheMaxBytes = bridgeProfile.cacheMaxBytes;
	timings.uncachedBatches += bridgeProfile.uncachedBatches;
}
#endif

class ProjectionDataValuesOverlay final : public ProjectionData
{
public:
	ProjectionDataValuesOverlay(const ProjectionData& source, float value,
	                            const BinIterator* binIterator = nullptr)
	    : ProjectionData(source.getScanner()),
	      m_source{source},
	      m_defaultValue{value}
	{
		size_t valueCount = source.count();
		if (binIterator != nullptr && binIterator->size() > 0 &&
		    dynamic_cast<const BinIteratorRange*>(binIterator) != nullptr)
		{
			m_useStridedSubset = true;
			m_firstBin = binIterator->get(0);
			m_binStride =
			    binIterator->size() > 1 ? binIterator->get(1) - m_firstBin :
			                              1;
			valueCount = binIterator->size();
		}
		m_values.assign(valueCount, value);
	}

	size_t count() const override
	{
		return m_source.count();
	}

	float getProjectionValue(bin_t id) const override
	{
		size_t index = 0;
		if (!getValueIndex(id, index))
		{
			return m_defaultValue;
		}
		return m_values[index];
	}

	void setProjectionValue(bin_t id, float val) override
	{
		size_t index = 0;
		ASSERT_MSG(getValueIndex(id, index),
		           "Projection overlay bin is outside the stored subset");
		m_values[index] = val;
	}

	bool hasSequentialSubsetStorage() const
	{
		return m_useStridedSubset;
	}

	float getStoredValueAt(size_t index) const
	{
		ASSERT(index < m_values.size());
		return m_values[index];
	}

	void setStoredValueAt(size_t index, float value)
	{
		ASSERT(index < m_values.size());
		m_values[index] = value;
	}

	det_id_t getDetector1(bin_t id) const override
	{
		return m_source.getDetector1(id);
	}

	det_id_t getDetector2(bin_t id) const override
	{
		return m_source.getDetector2(id);
	}

	det_pair_t getDetectorPair(bin_t id) const override
	{
		return m_source.getDetectorPair(id);
	}

	histo_bin_t getHistogramBin(bin_t bin) const override
	{
		return m_source.getHistogramBin(bin);
	}

	std::unique_ptr<BinIterator> getBinIter(int numSubsets,
	                                        int idxSubset) const override
	{
		return m_source.getBinIter(numSubsets, idxSubset);
	}

	timestamp_t getTimestamp(bin_t id) const override
	{
		return m_source.getTimestamp(id);
	}

	frame_t getDynamicFrame(bin_t id) const override
	{
		return m_source.getDynamicFrame(id);
	}

	frame_t getMotionFrame(bin_t id) const override
	{
		return m_source.getMotionFrame(id);
	}

	bool isUniform() const override
	{
		return m_source.isUniform();
	}

	bool hasRandomsEstimates() const override
	{
		return m_source.hasRandomsEstimates();
	}

	float getRandomsEstimate(bin_t id) const override
	{
		return m_source.getRandomsEstimate(id);
	}

	bool hasTOF() const override
	{
		return m_source.hasTOF();
	}

	float getTOFValue(bin_t id) const override
	{
		return m_source.getTOFValue(id);
	}

	bool hasMotion() const override
	{
		return m_source.hasMotion();
	}

	bool hasDynamicFraming() const override
	{
		return m_source.hasDynamicFraming();
	}

	size_t getNumDynamicFrames() const override
	{
		return m_source.getNumDynamicFrames();
	}

	size_t getNumMotionFrames() const override
	{
		return m_source.getNumMotionFrames();
	}

	transform_t getTransformOfMotionFrame(frame_t frame) const override
	{
		return m_source.getTransformOfMotionFrame(frame);
	}

	float getDurationOfMotionFrame(frame_t frame) const override
	{
		return m_source.getDurationOfMotionFrame(frame);
	}

	timestamp_t getScanDuration() const override
	{
		return m_source.getScanDuration();
	}

	bool hasArbitraryLORs() const override
	{
		return m_source.hasArbitraryLORs();
	}

	Line3D getArbitraryLOR(bin_t id) const override
	{
		return m_source.getArbitraryLOR(id);
	}

	std::set<ProjectionPropertyType> getProjectionPropertyTypes() const override
	{
		return m_source.getProjectionPropertyTypes();
	}

	void collectProjectionProperties(const ProjectionPropertyManager& propManager,
	                                 PropertyUnit* props, size_t pos,
	                                 bin_t bin) const override
	{
		m_source.collectProjectionProperties(propManager, props, pos, bin);
	}

private:
	bool getValueIndex(bin_t id, size_t& index) const
	{
		if (!m_useStridedSubset)
		{
			index = static_cast<size_t>(id);
			return index < m_values.size();
		}
		if (id < m_firstBin || m_binStride == 0)
		{
			return false;
		}
		const bin_t offset = id - m_firstBin;
		if (offset % m_binStride != 0)
		{
			return false;
		}
		index = static_cast<size_t>(offset / m_binStride);
		return index < m_values.size();
	}

	const ProjectionData& m_source;
	float m_defaultValue = 0.0f;
	bool m_useStridedSubset = false;
	bin_t m_firstBin = 0;
	bin_t m_binStride = 1;
	std::vector<float> m_values;
};

}  // namespace


OSEM_CPU::OSEM_CPU(const Scanner& pr_scanner)
    : OSEM(pr_scanner),
      mp_tempSensImageBuffer{nullptr},
      mp_mlemImageTmpEMRatio{nullptr}
{
	std::cout << "Creating an instance of OSEM CPU..." << std::endl;

	mp_corrector = std::make_unique<Corrector_CPU>(pr_scanner);
}

OSEM_CPU::~OSEM_CPU() = default;

void OSEM_CPU::addImagePSF(const std::string& p_imagePsf_fname,
                           ImagePSFMode p_imagePSFMode)
{
	ASSERT_MSG(!p_imagePsf_fname.empty(), "Empty filename for Image-space PSF");
	if (p_imagePSFMode == ImagePSFMode::UNIFORM)
	{
		imagePsf = std::make_unique<OperatorPsf>(p_imagePsf_fname);
		m_imagePSFMode = ImagePSFMode::UNIFORM;
	}
	else
	{
		ASSERT_MSG(imageParams.isValid(),
		           "For spatially variant PSF, image parameters have to be set "
		           "before calling addImagePSF");
		imagePsf =
		    std::make_unique<OperatorVarPsf>(p_imagePsf_fname, imageParams);
		m_imagePSFMode = ImagePSFMode::VARIANT;
	}
	flagImagePSF = true;
}

void OSEM_CPU::addUniformGaussianImagePSFFromFWHM(float fwhmX, float fwhmY,
                                                  float fwhmZ,
                                                  const size_t* kerSizeX,
                                                  const size_t* kerSizeY,
                                                  const size_t* kerSizeZ)
{
	ASSERT_MSG(imageParams.isValid(), "Image parameters not set");
	imagePsf = OperatorPsf::createGaussianFromFWHM(
	    fwhmX, fwhmY, fwhmZ, imageParams.vx, imageParams.vy, imageParams.vz,
	    kerSizeX, kerSizeY, kerSizeZ);
	m_imagePSFMode = ImagePSFMode::UNIFORM;
}

void OSEM_CPU::addUniformGaussianImagePSFFromSigma(float sigmaX, float sigmaY,
                                                   float sigmaZ,
                                                   const size_t* kerSizeX,
                                                   const size_t* kerSizeY,
                                                   const size_t* kerSizeZ)
{
	ASSERT_MSG(imageParams.isValid(), "Image parameters not set");
	imagePsf = OperatorPsf::createGaussianFromSigma(
	    sigmaX, sigmaY, sigmaZ, imageParams.vx, imageParams.vy, imageParams.vz,
	    kerSizeX, kerSizeY, kerSizeZ);
	m_imagePSFMode = ImagePSFMode::UNIFORM;
}

void OSEM_CPU::setExperimentalMetalProjectorEnabled(bool enabled)
{
	m_experimentalMetalProjectorEnabled = enabled;
}

bool OSEM_CPU::isExperimentalMetalProjectorEnabled() const
{
	return m_experimentalMetalProjectorEnabled;
}

bool OSEM_CPU::didLastExperimentalMetalProjectorRun() const
{
	return m_experimentalMetalProjectorRanLastCompute;
}

void OSEM_CPU::setExperimentalMetalProjectorFusedRatioEnabled(bool enabled)
{
	m_experimentalMetalProjectorFusedRatioEnabled = enabled;
}

bool OSEM_CPU::isExperimentalMetalProjectorFusedRatioEnabled() const
{
	return m_experimentalMetalProjectorFusedRatioEnabled;
}

void OSEM_CPU::setExperimentalMetalProjectorKernel(const std::string& kernel)
{
	if (kernel != "siddon" && kernel != "joseph")
	{
		throw std::invalid_argument(
		    "Experimental Metal projector kernel must be 'siddon' or 'joseph'");
	}
	m_experimentalMetalProjectorKernel = kernel;
}

std::string OSEM_CPU::getExperimentalMetalProjectorKernel() const
{
	return m_experimentalMetalProjectorKernel;
}

void OSEM_CPU::setExperimentalMetalProjectorProfilingEnabled(bool enabled)
{
	m_experimentalMetalProjectorProfilingEnabled = enabled;
}

bool OSEM_CPU::isExperimentalMetalProjectorProfilingEnabled() const
{
	return m_experimentalMetalProjectorProfilingEnabled;
}

void OSEM_CPU::setExperimentalMetalProjectorAdjointDiagnosticsEnabled(
    bool enabled)
{
	m_experimentalMetalProjectorAdjointDiagnosticsEnabled = enabled;
}

bool OSEM_CPU::isExperimentalMetalProjectorAdjointDiagnosticsEnabled() const
{
	return m_experimentalMetalProjectorAdjointDiagnosticsEnabled;
}

void OSEM_CPU::setExperimentalMetalProjectorAdjointHitDiagnosticsEnabled(
    bool enabled)
{
	m_experimentalMetalProjectorAdjointHitDiagnosticsEnabled = enabled;
}

bool OSEM_CPU::isExperimentalMetalProjectorAdjointHitDiagnosticsEnabled() const
{
	return m_experimentalMetalProjectorAdjointHitDiagnosticsEnabled;
}

void OSEM_CPU::resetExperimentalMetalProjectorTimings()
{
	m_experimentalMetalProjectorTimings = ExperimentalMetalProjectorTimings{};
	m_experimentalMetalProjectorSubsetTimings.clear();
}

OSEM_CPU::ExperimentalMetalProjectorTimings
OSEM_CPU::getExperimentalMetalProjectorTimings() const
{
	return m_experimentalMetalProjectorTimings;
}

std::vector<OSEM_CPU::ExperimentalMetalProjectorSubsetTiming>
OSEM_CPU::getExperimentalMetalProjectorSubsetTimings() const
{
	return m_experimentalMetalProjectorSubsetTimings;
}

void OSEM_CPU::setExperimentalMetalProjectorCacheEnabled(bool enabled)
{
	m_experimentalMetalProjectorCacheEnabled = enabled;
#if BUILD_METAL
	if (!enabled && mp_experimentalMetalProjectorCache != nullptr)
	{
		mp_experimentalMetalProjectorCache->clear();
	}
#endif
}

bool OSEM_CPU::isExperimentalMetalProjectorCacheEnabled() const
{
	return m_experimentalMetalProjectorCacheEnabled;
}

void OSEM_CPU::setExperimentalMetalProjectorCacheMaxBytes(size_t maxBytes)
{
	m_experimentalMetalProjectorCacheMaxBytes = maxBytes;
#if BUILD_METAL
	if (mp_experimentalMetalProjectorCache != nullptr)
	{
		mp_experimentalMetalProjectorCache->setMaxBytes(maxBytes);
	}
#endif
}

size_t OSEM_CPU::getExperimentalMetalProjectorCacheMaxBytes() const
{
	return m_experimentalMetalProjectorCacheMaxBytes;
}

void OSEM_CPU::setExperimentalMetalProjectorMaxBatchEvents(
    size_t maxBatchEvents)
{
	m_experimentalMetalProjectorMaxBatchEvents = maxBatchEvents;
#if BUILD_METAL
	if (mp_experimentalMetalProjectorCache != nullptr)
	{
		mp_experimentalMetalProjectorCache->setMaxBatchEvents(maxBatchEvents);
	}
#endif
}

size_t OSEM_CPU::getExperimentalMetalProjectorMaxBatchEvents() const
{
	return m_experimentalMetalProjectorMaxBatchEvents;
}

void OSEM_CPU::setExperimentalMetalProjectorMaxChunkEvents(
    size_t maxChunkEvents)
{
	setExperimentalMetalProjectorMaxBatchEvents(maxChunkEvents);
}

size_t OSEM_CPU::getExperimentalMetalProjectorMaxChunkEvents() const
{
	return getExperimentalMetalProjectorMaxBatchEvents();
}

void OSEM_CPU::setupProjectorForSensImgGen()
{
	ASSERT(projectorParams.numRays > 0);

	// Note: The projector updater for the sensitivity image generation is
	//  always DEFAULT4D

	// Create ProjectorParams object only for sensitivity image, without TOF
	ProjectorParams projParams(scanner);
	projParams.projectorType = projectorParams.projectorType;
	projParams.projPsf_fname = projectorParams.projPsf_fname;
	projParams.numRays = projectorParams.numRays;

	mp_projector = Projector::create(projParams);
}

void OSEM_CPU::prepareBuffersForSensImgGen()
{
	auto imageParamsSens = getImageParamsForSensImgGen();

	auto tempSensImageBuffer = std::make_unique<ImageOwned>(imageParamsSens);
	tempSensImageBuffer->allocate();
	mp_tempSensImageBuffer = std::move(tempSensImageBuffer);

	if (flagImagePSF)
	{
		mp_imageTmpPsf = std::make_unique<ImageOwned>(imageParams);
		reinterpret_cast<ImageOwned*>(mp_imageTmpPsf.get())->allocate();
	}

	initBinLoader(false);
}

std::unique_ptr<Image> OSEM_CPU::generateSensitivityImageForCurrentSubset()
{
	Image* sensImagePtr = mp_tempSensImageBuffer.get();
	ASSERT(sensImagePtr != nullptr);
	const Projector* projector = mp_projector.get();
	ASSERT(projector != nullptr);
	const BinIterator* binIter = getBinIterator(getCurrentOSEMSubset());
	ASSERT(binIter != nullptr);

	const Corrector_CPU& corrector = getCorrector_CPU();
	const ProjectionData* sensImgGenProjData =
	    corrector.getSensImgGenProjData();
	ASSERT(sensImgGenProjData != nullptr);

	const float globalScale = mp_corrector->getGlobalScalingFactor();

	// Reset sens image buffer
	sensImagePtr->fill(0.0f);

	mp_binLoader->parallelDoOnBins<true>(
	    *sensImgGenProjData, *binIter,
	    [&corrector, &sensImgGenProjData, &projector,
	     &sensImagePtr](const ProjectionPropertyManager& propManager,
	                    PropertyUnit* props, size_t pos, bin_t bin)
	    {
		    const float projValue = corrector.getMultiplicativeCorrectionFactor(
		        *sensImgGenProjData, bin);
		    projector->backProjection(sensImagePtr, propManager, props, pos,
		                              projValue);
	    });

	if (flagImagePSF)
	{
		imagePsf->applyAH(mp_tempSensImageBuffer.get(), mp_imageTmpPsf.get());

		mp_imageTmpPsf->applyThreshold(mp_tempSensImageBuffer.get(), EPS_FLT,
		                               0.0f, 0.0f, 1.0f, 0.0f);
		mp_tempSensImageBuffer.swap(mp_imageTmpPsf);
	}

	// All voxels lower than "hardThreshold" will be put to 0
	std::cout << "Applying threshold..." << std::endl;
	mp_tempSensImageBuffer->applyThreshold(mp_tempSensImageBuffer.get(),
	                                       hardThreshold, 0.0f, 0.0f,
	                                       globalScale, 0.0f);

	// Return generated image, but allocate for the next subset if needed
	// This will dereference mp_tempSensImageBuffer
	auto img = std::move(mp_tempSensImageBuffer);

	// Which requires another allocation for the next subset (if there is one)
	if (getCurrentOSEMSubset() != num_OSEM_subsets - 1)
	{
		auto tempSensImageBuffer =
		    std::make_unique<ImageOwned>(getImageParamsForSensImgGen());
		tempSensImageBuffer->allocate();
		mp_tempSensImageBuffer = std::move(tempSensImageBuffer);
	}

	return img;
}

void OSEM_CPU::endSensImgGen()
{
	// Clear temporary buffers
	mp_tempSensImageBuffer = nullptr;
}

void OSEM_CPU::setupForDynamicRecon()
{
	OSEM::setupForDynamicRecon();
}

void OSEM_CPU::setupProjectorForRecon()
{
	if (projectorParams.projectorType == ProjectorType::SIDDON)
	{
		mp_projector = std::make_unique<ProjectorSiddon>(projectorParams);
	}
	else if (projectorParams.projectorType == ProjectorType::DD)
	{
		mp_projector = std::make_unique<ProjectorDD>(projectorParams);
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}
}

void OSEM_CPU::prepareBuffersForRecon()
{
	// Allocate for projection-space buffers
	const ProjectionData* dataInput = getDataInput();

	// Allocate for image-space buffers
	mp_mlemImageTmpEMRatio = std::make_unique<ImageOwned>(getImageParams());
	reinterpret_cast<ImageOwned*>(mp_mlemImageTmpEMRatio.get())->allocate();
	if (flagImagePSF)
	{
		mp_imageTmpPsf = std::make_unique<ImageOwned>(getImageParams());
		reinterpret_cast<ImageOwned*>(mp_imageTmpPsf.get())->allocate();
	}

	// Initialize output image
	if (initialEstimate != nullptr)
	{
		outImage->copyFromImage(initialEstimate);
	}
	else
	{
		outImage->fill(INITIAL_VALUE_MLEM);
	}

	// Apply mask function
	auto applyMask = [this](const Image* maskImage) -> void
	{
		outImage->applyThresholdBroadcast(maskImage, 0.0f, 0.0f, 0.0f, 1.0f,
		                                  0.0f);
	};

	// Apply mask image
	std::cout << "Applying threshold..." << std::endl;
	if (maskImage != nullptr)
	{
		applyMask(maskImage);
	}
	else if (num_OSEM_subsets == 1 || usingListModeInput)
	{
		// No need to sum all sensitivity images, just use the only one
		applyMask(getSensitivityImage(0));
	}
	else
	{
		std::cout << "Summing sensitivity images to generate mask image..."
		          << std::endl;
		for (int i = 0; i < num_OSEM_subsets; ++i)
		{
			getSensitivityImage(i)->addFirstImageToSecond(
			    mp_mlemImageTmpEMRatio.get());
		}
		applyMask(mp_mlemImageTmpEMRatio.get());
	}
	mp_mlemImageTmpEMRatio->fill(0.0f);

	mp_corrector->precomputeCorrectionFactors(*dataInput);

	initBinLoader(true);

#if BUILD_METAL
	if (mp_experimentalMetalProjectorCache != nullptr)
	{
		mp_experimentalMetalProjectorCache->clear();
	}
#endif
}

void OSEM_CPU::loadCurrentSubset(bool /*forRecon*/) {}

void OSEM_CPU::resetEMUpdateImage()
{
	mp_mlemImageTmpEMRatio->fill(0.0);
}

void OSEM_CPU::computeEMUpdateImage()
{
	m_experimentalMetalProjectorRanLastCompute = false;

	if (flagImagePSF)
	{
		mp_imageTmpPsf->fill(0.0);
		imagePsf->applyA(outImage.get(), mp_imageTmpPsf.get());

		// We swap here so that the outImage buffer stores the MLEM image with
		//  the PSF applied to it and the imageTmpPsf buffer stores the original
		//  MLEM image.
		outImage.swap(mp_imageTmpPsf);
	}

	const Image* inputImageForForwardProj = outImage.get();
	Image* destImageForBackproj =
	    dynamic_cast<Image*>(getEMUpdateImageBuffer());
	ASSERT(destImageForBackproj != nullptr);

	const Projector* projector = mp_projector.get();
	ASSERT(projector != nullptr);
	const BinIterator* binIter = getBinIterator(getCurrentOSEMSubset());
	ASSERT(binIter != nullptr);
	const ProjectionData* measurements = getDataInput();
	ASSERT(measurements != nullptr);
	const Corrector_CPU& corrector = getCorrector_CPU();

	const float globalScaleFactor = corrector.getGlobalScalingFactor();
	const bool hasSensitivity = corrector.hasSensitivityHistogram();
	const bool hasAttenuation = corrector.hasAttenuation();
	const bool hasScatterEstimates = corrector.hasScatterEstimates();
	const bool hasRandomsEstimates =
	    corrector.hasRandomsEstimates(*measurements);
	const bool hasInVivoAttenuation = corrector.hasInVivoAttenuation();

	corrector.assertMeasurementsMatchCache(measurements);

	bool computedWithExperimentalMetalProjector = false;
	if (m_experimentalMetalProjectorEnabled && !flagImagePSF)
	{
		computedWithExperimentalMetalProjector =
		    computeEMUpdateImageWithExperimentalMetalProjector(
		        *inputImageForForwardProj, *destImageForBackproj,
		        *measurements, *binIter, corrector, globalScaleFactor,
		        hasSensitivity, hasAttenuation, hasScatterEstimates,
		        hasRandomsEstimates, hasInVivoAttenuation);
		m_experimentalMetalProjectorRanLastCompute =
		    computedWithExperimentalMetalProjector;
	}

	if (!computedWithExperimentalMetalProjector)
	{
		mp_binLoader->parallelDoOnBins<false>(
		    *measurements, *binIter,
		    [&projector, &corrector, &measurements, hasRandomsEstimates,
		     hasScatterEstimates, hasInVivoAttenuation, hasAttenuation,
		     hasSensitivity, globalScaleFactor, &destImageForBackproj,
		     &inputImageForForwardProj,
		     this](const ProjectionPropertyManager& propManager,
		           PropertyUnit* props, size_t pos, bin_t bin)
		    {
			    float update = projector->forwardProjection(
			        inputImageForForwardProj, propManager, props, pos);

			    if (hasSensitivity)
			    {
				    update *= corrector.getPrecomputedSensitivityFactor(bin);
			    }
			    if (hasAttenuation)
			    {
				    update *= corrector.getPrecomputedAttenuationFactor(bin);
			    }
			    update *= globalScaleFactor;

			    if (hasRandomsEstimates)
			    {
				    update += corrector.getPrecomputedRandomsEstimate(bin);
			    }
			    if (hasScatterEstimates)
			    {
				    update += corrector.getPrecomputedScatterEstimate(bin);
			    }

			    if (hasInVivoAttenuation)
			    {
				    update *=
				        corrector.getPrecomputedInVivoAttenuationFactor(bin);
			    }

			    // to prevent numerical instability
			    if (std::abs(update) > denomThreshold)
			    {
				    const float measurement =
				        measurements->getProjectionValue(bin);
				    update = measurement / update;

				    if (hasSensitivity)
				    {
					    update *= corrector.getPrecomputedSensitivityFactor(bin);
				    }
				    if (hasAttenuation)
				    {
					    update *= corrector.getPrecomputedAttenuationFactor(bin);
				    }
				    update *= globalScaleFactor;

				    projector->backProjection(destImageForBackproj,
				                              propManager, props, pos, update);
			    }
		    });
	}

	// Backward PSF
	if (flagImagePSF)
	{
		// We swap again here so that outImage gets back to storing the original
		//  MLEM image. This way, we can use the imageTmpPsf buffer to compute
		//  the PSF
		mp_imageTmpPsf.swap(outImage);

		// YN: Is this initialization necessary ?
		mp_imageTmpPsf->fill(0.0);
		imagePsf->applyAH(mp_mlemImageTmpEMRatio.get(), mp_imageTmpPsf.get());

		// We swap these two buffers so that we can use mlemImageTmpEMRatio to
		//  apply the image update
		mp_mlemImageTmpEMRatio.swap(mp_imageTmpPsf);
	}
}

bool OSEM_CPU::computeEMUpdateImageWithExperimentalMetalProjector(
    const Image& inputImageForForwardProj, Image& destImageForBackproj,
    const ProjectionData& measurements, const BinIterator& binIter,
    const Corrector_CPU& corrector, float globalScaleFactor,
    bool hasSensitivity, bool hasAttenuation, bool hasScatterEstimates,
    bool hasRandomsEstimates, bool hasInVivoAttenuation)
{
#if BUILD_METAL
	if (flagImagePSF || mp_binLoader == nullptr)
	{
		return false;
	}

	const auto memoryBefore = m_experimentalMetalProjectorProfilingEnabled ?
	                              sampleExperimentalMetalMemory() :
	                              ExperimentalMetalProjectorMemorySnapshot{};
	const auto totalStart = Clock::now();
	double setupSeconds = 0.0;
	double forwardSeconds = 0.0;
	double ratioSeconds = 0.0;
	double adjointSeconds = 0.0;

	const auto setupStart = Clock::now();
	const backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	std::vector<Constraint*> constraints = getConstraintsAsVectorOfPointers();
	OperatorProjector projector(projectorParams, &binIter, constraints);
	const BinLoader* bridgeBinLoader = projector.getBinLoader();
	if (bridgeBinLoader == nullptr)
	{
		return false;
	}

	backend::metal::OperatorProjectorMetalProfile bridgeProfile;
	bridgeProfile.diagnoseAdjointUpdateCounts =
	    m_experimentalMetalProjectorAdjointDiagnosticsEnabled;
	bridgeProfile.diagnoseAdjointVoxelHits =
	    m_experimentalMetalProjectorAdjointHitDiagnosticsEnabled;
	backend::metal::OperatorProjectorMetalCache* bridgeCache = nullptr;
	if (m_experimentalMetalProjectorCacheEnabled)
	{
		if (mp_experimentalMetalProjectorCache == nullptr)
		{
			mp_experimentalMetalProjectorCache =
			    std::make_unique<backend::metal::OperatorProjectorMetalCache>();
		}
		mp_experimentalMetalProjectorCache->setMaxBytes(
		    m_experimentalMetalProjectorCacheMaxBytes);
		mp_experimentalMetalProjectorCache->setMaxBatchEvents(
		    m_experimentalMetalProjectorMaxBatchEvents);
		bridgeCache = mp_experimentalMetalProjectorCache.get();
	}
	const backend::metal::OperatorProjectorMetalBridge bridge(
	    context, m_experimentalMetalProjectorProfilingEnabled ? &bridgeProfile
	                                                          : nullptr,
	    bridgeCache);
	if (!bridge.canRunSiddon(projector).supported)
	{
		return false;
	}

	setupSeconds = getElapsedSeconds(setupStart, Clock::now());

	const auto metalProjectorKernel =
	    m_experimentalMetalProjectorKernel == "joseph" ?
	        backend::metal::OperatorProjectorMetalKernel::Joseph :
	        backend::metal::OperatorProjectorMetalKernel::Siddon;
	bool didRun = false;
	if (m_experimentalMetalProjectorFusedRatioEnabled)
	{
		const backend::metal::OperatorProjectorMetalOsemConfig metalOsemConfig{
		    globalScaleFactor,
		    denomThreshold,
		    hasSensitivity,
		    hasAttenuation,
		    hasScatterEstimates,
		    hasRandomsEstimates,
		    hasInVivoAttenuation,
		    true,
		    metalProjectorKernel};
		didRun = bridge.applyOsemEMUpdate(projector, inputImageForForwardProj,
		    destImageForBackproj, measurements, binIter, *bridgeBinLoader,
		    corrector, metalOsemConfig);

		if (m_experimentalMetalProjectorProfilingEnabled)
		{
			forwardSeconds = bridgeProfile.forwardGatherSeconds +
			                 bridgeProfile.forwardPackSeconds +
			                 bridgeProfile.forwardBatchUploadSeconds +
			                 bridgeProfile.forwardImageUploadSeconds +
			                 bridgeProfile.forwardKernelSeconds +
			                 bridgeProfile.forwardDownloadSeconds +
			                 bridgeProfile.forwardHostWriteSeconds;
			ratioSeconds = bridgeProfile.ratioPackSeconds +
			               bridgeProfile.ratioBatchUploadSeconds +
			               bridgeProfile.ratioKernelSeconds;
			adjointSeconds = bridgeProfile.adjointGatherSeconds +
			                 bridgeProfile.adjointPackSeconds +
			                 bridgeProfile.adjointBatchUploadSeconds +
			                 bridgeProfile.adjointImageUploadSeconds +
			                 bridgeProfile.adjointKernelSeconds +
			                 bridgeProfile.adjointImageDownloadSeconds +
			                 bridgeProfile.adjointHostImageCopySeconds;
		}
	}
	else
	{
		const backend::metal::OperatorProjectorMetalOsemConfig metalOsemConfig{
		    globalScaleFactor,
		    denomThreshold,
		    hasSensitivity,
		    hasAttenuation,
		    hasScatterEstimates,
		    hasRandomsEstimates,
		    hasInVivoAttenuation,
		    true,
		    metalProjectorKernel};
		didRun = bridge.applyOsemEMUpdateHostRatio(projector,
		    inputImageForForwardProj, destImageForBackproj, measurements,
		    binIter, *bridgeBinLoader, corrector, metalOsemConfig);

		if (didRun && m_experimentalMetalProjectorProfilingEnabled)
		{
			forwardSeconds = bridgeProfile.forwardGatherSeconds +
			                 bridgeProfile.forwardPackSeconds +
			                 bridgeProfile.forwardBatchUploadSeconds +
			                 bridgeProfile.forwardImageUploadSeconds +
			                 bridgeProfile.forwardKernelSeconds +
			                 bridgeProfile.forwardDownloadSeconds +
			                 bridgeProfile.forwardHostWriteSeconds;
			ratioSeconds = bridgeProfile.ratioPackSeconds +
			               bridgeProfile.ratioBatchUploadSeconds +
			               bridgeProfile.ratioKernelSeconds;
			adjointSeconds = bridgeProfile.adjointGatherSeconds +
			                 bridgeProfile.adjointPackSeconds +
			                 bridgeProfile.adjointBatchUploadSeconds +
			                 bridgeProfile.adjointImageUploadSeconds +
			                 bridgeProfile.adjointKernelSeconds +
			                 bridgeProfile.adjointImageDownloadSeconds +
			                 bridgeProfile.adjointHostImageCopySeconds;
		}
	}
	if (!didRun && !m_experimentalMetalProjectorFusedRatioEnabled)
	{
		ProjectionDataValuesOverlay estimatedProjections(measurements, 0.0f,
		                                                 &binIter);
		const auto forwardStart = Clock::now();
		if (!bridge.applyA(projector, inputImageForForwardProj,
		        estimatedProjections, binIter, *bridgeBinLoader))
		{
			return false;
		}
		forwardSeconds = getElapsedSeconds(forwardStart, Clock::now());

		ProjectionDataValuesOverlay ratioProjections(measurements, 0.0f,
		                                             &binIter);
		const auto ratioStart = Clock::now();
		auto computeRatioValue = [&](bin_t bin, float update,
		                             float& ratioValue) -> bool
		{
			if (hasSensitivity)
			{
				update *= corrector.getPrecomputedSensitivityFactor(bin);
			}
			if (hasAttenuation)
			{
				update *= corrector.getPrecomputedAttenuationFactor(bin);
			}
			update *= globalScaleFactor;

			if (hasRandomsEstimates)
			{
				update += corrector.getPrecomputedRandomsEstimate(bin);
			}
			if (hasScatterEstimates)
			{
				update += corrector.getPrecomputedScatterEstimate(bin);
			}

			if (hasInVivoAttenuation)
			{
				update *= corrector.getPrecomputedInVivoAttenuationFactor(bin);
			}

			if (std::abs(update) <= denomThreshold)
			{
				return false;
			}

			update = measurements.getProjectionValue(bin) / update;

			if (hasSensitivity)
			{
				update *= corrector.getPrecomputedSensitivityFactor(bin);
			}
			if (hasAttenuation)
			{
				update *= corrector.getPrecomputedAttenuationFactor(bin);
			}
			update *= globalScaleFactor;
			ratioValue = update;
			return true;
		};

		if (estimatedProjections.hasSequentialSubsetStorage() &&
		    ratioProjections.hasSequentialSubsetStorage())
		{
			BinLoader ratioBinLoader(getConstraintsAsVectorOfPointers(), {});
			const int numThreads = globals::getNumThreads();
			ratioBinLoader.allocate(numThreads);
			BinFilter::CollectInfoFlags collectInfoFlags(false);
			ratioBinLoader.collectFlags(collectInfoFlags);
			util::parallelForChunked(
			    binIter.size(), numThreads,
			    [&](size_t binIdx, int tid)
			    {
				    const bin_t bin =
				        binIter.get(static_cast<bin_t>(binIdx));
				    ratioBinLoader.collectInfo(bin, tid, tid, measurements,
				                               collectInfoFlags);
				    if (!ratioBinLoader.verifyConstraints(tid))
				    {
					    return;
				    }
				    float ratioValue = 0.0f;
				    if (computeRatioValue(
				            bin, estimatedProjections.getStoredValueAt(binIdx),
				            ratioValue))
				    {
					    ratioProjections.setStoredValueAt(binIdx, ratioValue);
				    }
			    });
		}
		else
		{
			mp_binLoader->parallelDoOnBins<false>(
			    measurements, binIter,
			    [&estimatedProjections, &ratioProjections, &computeRatioValue](
			        const ProjectionPropertyManager& /*propManager*/,
			        PropertyUnit* /*props*/, size_t /*pos*/, bin_t bin)
			    {
				    float ratioValue = 0.0f;
				    if (computeRatioValue(
				            bin, estimatedProjections.getProjectionValue(bin),
				            ratioValue))
				    {
					    ratioProjections.setProjectionValue(bin, ratioValue);
				    }
			    });
		}
		ratioSeconds = getElapsedSeconds(ratioStart, Clock::now());

		const auto adjointStart = Clock::now();
		didRun = bridge.applyAH(projector, ratioProjections,
		                        destImageForBackproj, binIter,
		                        *bridgeBinLoader);
		adjointSeconds = getElapsedSeconds(adjointStart, Clock::now());
	}

	if (didRun && m_experimentalMetalProjectorProfilingEnabled)
	{
		const double totalSeconds = getElapsedSeconds(totalStart, Clock::now());
		addBridgeProfileToTimings(m_experimentalMetalProjectorTimings,
		    setupSeconds, forwardSeconds, ratioSeconds, adjointSeconds,
		    totalSeconds, bridgeProfile);

		ExperimentalMetalProjectorSubsetTiming subsetTiming;
		subsetTiming.iteration = getCurrentMLEMIteration();
		subsetTiming.subset = getCurrentOSEMSubset();
		subsetTiming.events = binIter.size();
		subsetTiming.metalRan = didRun;
		subsetTiming.memoryBefore = memoryBefore;
		addBridgeProfileToTimings(subsetTiming, setupSeconds, forwardSeconds,
		    ratioSeconds, adjointSeconds, totalSeconds, bridgeProfile);
		subsetTiming.memoryAfter = sampleExperimentalMetalMemory();
		m_experimentalMetalProjectorSubsetTimings.push_back(subsetTiming);
	}

	return didRun;
#else
	(void)inputImageForForwardProj;
	(void)destImageForBackproj;
	(void)measurements;
	(void)binIter;
	(void)corrector;
	(void)globalScaleFactor;
	(void)hasSensitivity;
	(void)hasAttenuation;
	(void)hasScatterEstimates;
	(void)hasRandomsEstimates;
	(void)hasInVivoAttenuation;
	return false;
#endif
}

void OSEM_CPU::applyImageUpdate()
{
	// Apply update using the correct sensitivity image
	const ImageBase* sensImage = getSensImageBuffer();

	// Apply the update on the outImage buffer
	outImage->updateEMThresholdDynamic(mp_mlemImageTmpEMRatio.get(), sensImage,
	                                   EPS_FLT);
}


void OSEM_CPU::completeSubset() {}

void OSEM_CPU::completeMLEMIteration() {}

void OSEM_CPU::endRecon()
{
	// Clear temporary buffers
	mp_mlemImageTmpEMRatio = nullptr;
}

ImageBase* OSEM_CPU::getSensImageBuffer()
{
	// In case we are currently generating the sensitivity image
	if (mp_tempSensImageBuffer != nullptr)
	{
		return mp_tempSensImageBuffer.get();
	}
	// In case we are reconstructing
	return getSensitivityImage(usingListModeInput ? 0 : getCurrentOSEMSubset());
}

ImageBase* OSEM_CPU::getMLEMImageBuffer()
{
	return outImage.get();
}

ImageBase* OSEM_CPU::getEMUpdateImageBuffer()
{
	return mp_mlemImageTmpEMRatio.get();
}

const Corrector& OSEM_CPU::getCorrector() const
{
	return *mp_corrector;
}

Corrector& OSEM_CPU::getCorrector()
{
	return *mp_corrector;
}

const Corrector_CPU& OSEM_CPU::getCorrector_CPU() const
{
	return *mp_corrector;
}

void OSEM_CPU::initBinLoader(bool forRecon)
{
	std::vector<Constraint*> constraints = getConstraintsAsVectorOfPointers();
	std::set<ProjectionPropertyType> properties = getNeededProperties(forRecon);

	mp_binLoader = std::make_unique<BinLoader>(constraints, properties);

	const int numThreads = globals::getNumThreads();
	mp_binLoader->allocate(numThreads);
}

std::set<ProjectionPropertyType>
    OSEM_CPU::getNeededProperties(bool forRecon) const
{
	std::set<ProjectionPropertyType> properties;

	// The correction factors are gathered directly from the corrector
	//  instead of being stored in the projection properties structure

	properties.merge(mp_projector->getProjectionPropertyTypes());

	if (forRecon && getDataInput()->hasDynamicFraming())
	{
		// Dynamic frame is necessary if the input data has dynamic framing
		properties.insert(ProjectionPropertyType::DYNAMIC_FRAME);
	}

	return properties;
}

}  // namespace yrt
