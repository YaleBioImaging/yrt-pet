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
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#if BUILD_METAL
#include "yrt-pet/backends/metal/ImageSpaceKernels.hpp"
#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/OperatorProjectorMetalBridge.hpp"
#include "yrt-pet/backends/metal/OperatorPsfMetal.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorOps.hpp"
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
	c.def("setExperimentalMetalProjectorResidentImagesEnabled",
	      &OSEM_CPU::setExperimentalMetalProjectorResidentImagesEnabled,
	      "enabled"_a);
	c.def("isExperimentalMetalProjectorResidentImagesEnabled",
	      &OSEM_CPU::isExperimentalMetalProjectorResidentImagesEnabled);
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
		      result["setup_context_s"] = timings.setupContextSeconds;
		      result["setup_projector_s"] = timings.setupProjectorSeconds;
		      result["setup_cache_s"] = timings.setupCacheSeconds;
		      result["setup_bridge_s"] = timings.setupBridgeSeconds;
		      result["setup_can_run_s"] = timings.setupCanRunSeconds;
		      result["forward_s"] = timings.forwardSeconds;
		      result["ratio_s"] = timings.ratioSeconds;
		      result["adjoint_s"] = timings.adjointSeconds;
		      result["total_s"] = timings.totalSeconds;
		      result["metal_path_overhead_s"] =
		          timings.metalPathOverheadSeconds;
		      result["compute_update_image_s"] =
		          timings.computeUpdateImageSeconds;
		      result["image_update_s"] = timings.imageUpdateSeconds;
		      result["recon_initialize_s"] = timings.reconInitializeSeconds;
		      result["recon_setup_dynamic_s"] =
		          timings.reconSetupDynamicSeconds;
		      result["recon_initialize_out_image_s"] =
		          timings.reconInitializeOutImageSeconds;
		      result["recon_initialize_sens_image_s"] =
		          timings.reconInitializeSensImageSeconds;
		      result["recon_corrector_setup_s"] =
		          timings.reconCorrectorSetupSeconds;
		      result["recon_initialize_bin_iterators_s"] =
		          timings.reconInitializeBinIteratorsSeconds;
		      result["recon_collect_constraints_s"] =
		          timings.reconCollectConstraintsSeconds;
		      result["recon_setup_projector_s"] =
		          timings.reconSetupProjectorSeconds;
		      result["recon_prepare_buffers_s"] =
		          timings.reconPrepareBuffersSeconds;
		      result["recon_iterate_s"] = timings.reconIterateSeconds;
		      result["recon_load_subset_s"] = timings.reconLoadSubsetSeconds;
		      result["recon_reset_update_s"] = timings.reconResetUpdateSeconds;
		      result["recon_compute_update_phase_s"] =
		          timings.reconComputeUpdatePhaseSeconds;
		      result["recon_apply_update_phase_s"] =
		          timings.reconApplyUpdatePhaseSeconds;
		      result["recon_complete_subset_s"] =
		          timings.reconCompleteSubsetSeconds;
		      result["recon_save_iteration_s"] =
		          timings.reconSaveIterationSeconds;
		      result["recon_complete_mlem_s"] =
		          timings.reconCompleteMLEMSeconds;
		      result["recon_end_s"] = timings.reconEndSeconds;
		      result["prepare_allocate_images_s"] =
		          timings.prepareAllocateImagesSeconds;
		      result["prepare_initialize_output_s"] =
		          timings.prepareInitializeOutputSeconds;
		      result["prepare_apply_mask_s"] =
		          timings.prepareApplyMaskSeconds;
		      result["prepare_clear_update_s"] =
		          timings.prepareClearUpdateSeconds;
		      result["prepare_precompute_corrections_s"] =
		          timings.preparePrecomputeCorrectionsSeconds;
		      result["prepare_init_bin_loader_s"] =
		          timings.prepareInitBinLoaderSeconds;
		      result["prepare_clear_metal_cache_s"] =
		          timings.prepareClearMetalCacheSeconds;
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
		      result["ratio_correction_cache_build_s"] =
		          timings.ratioCorrectionCacheBuildSeconds;
		      result["ratio_nonzero_diagnostic_s"] =
		          timings.ratioNonzeroDiagnosticSeconds;
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
		      result["cache_lookup_s"] = timings.cacheLookupSeconds;
		      result["cache_admission_s"] = timings.cacheAdmissionSeconds;
		      result["cache_admission_gather_s"] =
		          timings.cacheAdmissionGatherSeconds;
		      result["cache_admission_pack_s"] =
		          timings.cacheAdmissionPackSeconds;
		      result["cache_admission_batch_upload_s"] =
		          timings.cacheAdmissionBatchUploadSeconds;
		      result["cache_admission_correction_build_s"] =
		          timings.cacheAdmissionCorrectionBuildSeconds;
		      result["cache_admission_correction_fill_s"] =
		          timings.cacheAdmissionCorrectionFillSeconds;
		      result["cache_admission_correction_upload_s"] =
		          timings.cacheAdmissionCorrectionUploadSeconds;
		      result["cache_admission_correction_measurement_s"] =
		          timings.cacheAdmissionCorrectionMeasurementSeconds;
		      result["cache_admission_correction_multiplicative_s"] =
		          timings.cacheAdmissionCorrectionMultiplicativeSeconds;
		      result["cache_admission_correction_additive_s"] =
		          timings.cacheAdmissionCorrectionAdditiveSeconds;
		      result["cache_admission_correction_in_vivo_s"] =
		          timings.cacheAdmissionCorrectionInVivoSeconds;
		      result["cache_insert_s"] = timings.cacheInsertSeconds;
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
		      result["adjoint_max_batch_p50_voxel_hits"] =
		          timings.adjointMaxBatchP50VoxelHits;
		      result["adjoint_max_batch_p90_voxel_hits"] =
		          timings.adjointMaxBatchP90VoxelHits;
		      result["adjoint_max_batch_p95_voxel_hits"] =
		          timings.adjointMaxBatchP95VoxelHits;
		      result["adjoint_max_batch_p99_voxel_hits"] =
		          timings.adjointMaxBatchP99VoxelHits;
		      result["adjoint_max_batch_p999_voxel_hits"] =
		          timings.adjointMaxBatchP999VoxelHits;
		      result["adjoint_max_batch_mean_voxel_hits"] =
		          timings.adjointMaxBatchMeanVoxelHits;
		      result["adjoint_max_batch_top_1pct_voxel_hit_fraction"] =
		          timings.adjointMaxBatchTop1PctVoxelHitFraction;
		      result["adjoint_max_batch_top_0_1pct_voxel_hit_fraction"] =
		          timings.adjointMaxBatchTop01PctVoxelHitFraction;
		      result["adjoint_tile_size"] = timings.adjointTileSize;
		      result["adjoint_voxel_hit_tiles"] =
		          timings.adjointVoxelHitTiles;
		      result["adjoint_voxel_hit_tile_total_updates"] =
		          timings.adjointVoxelHitTileTotalUpdates;
		      result["adjoint_max_tile_hits"] = timings.adjointMaxTileHits;
		      result["adjoint_max_batch_p95_tile_hits"] =
		          timings.adjointMaxBatchP95TileHits;
		      result["adjoint_max_batch_p99_tile_hits"] =
		          timings.adjointMaxBatchP99TileHits;
		      result["adjoint_max_batch_mean_tile_hits"] =
		          timings.adjointMaxBatchMeanTileHits;
		      result["adjoint_max_batch_top_1pct_tile_hit_fraction"] =
		          timings.adjointMaxBatchTop1PctTileHitFraction;
		      result["adjoint_max_batch_top_0_1pct_tile_hit_fraction"] =
		          timings.adjointMaxBatchTop01PctTileHitFraction;
		      result["cache_lookups"] = timings.cacheLookups;
		      result["cache_hits"] = timings.cacheHits;
		      result["cache_misses"] = timings.cacheMisses;
		      result["cache_builds"] = timings.cacheBuilds;
		      result["cache_skips_over_budget"] =
		          timings.cacheSkipsOverBudget;
		      result["cache_used_bytes"] = timings.cacheUsedBytes;
		      result["cache_max_bytes"] = timings.cacheMaxBytes;
		      result["cache_correction_reserve_bytes"] =
		          timings.cacheCorrectionReserveBytes;
		      result["uncached_batches"] = timings.uncachedBatches;
		      result["uncached_chunks"] = timings.uncachedBatches;
		      result["ratio_correction_cache_builds"] =
		          timings.ratioCorrectionCacheBuilds;
		      result["ratio_correction_cache_hits"] =
		          timings.ratioCorrectionCacheHits;
		      result["ratio_correction_cache_misses"] =
		          timings.ratioCorrectionCacheMisses;
		      result["ratio_correction_cache_bytes"] =
		          timings.ratioCorrectionCacheBytes;
		      result["ratio_values"] = timings.ratioValues;
		      result["ratio_nonzero_values"] = timings.ratioNonzeroValues;
		      result["ratio_zero_values"] = timings.ratioZeroValues;
		      result["ratio_nonzero_diagnostic_batches"] =
		          timings.ratioNonzeroDiagnosticBatches;
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
			      row["setup_context_s"] = timing.setupContextSeconds;
			      row["setup_projector_s"] = timing.setupProjectorSeconds;
			      row["setup_cache_s"] = timing.setupCacheSeconds;
			      row["setup_bridge_s"] = timing.setupBridgeSeconds;
			      row["setup_can_run_s"] = timing.setupCanRunSeconds;
			      row["forward_s"] = timing.forwardSeconds;
			      row["ratio_s"] = timing.ratioSeconds;
			      row["adjoint_s"] = timing.adjointSeconds;
			      row["total_s"] = timing.totalSeconds;
			      row["metal_path_overhead_s"] =
			          timing.metalPathOverheadSeconds;
			      row["compute_update_image_s"] =
			          timing.computeUpdateImageSeconds;
			      row["image_update_s"] = timing.imageUpdateSeconds;
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
			      row["ratio_correction_cache_build_s"] =
			          timing.ratioCorrectionCacheBuildSeconds;
			      row["ratio_nonzero_diagnostic_s"] =
			          timing.ratioNonzeroDiagnosticSeconds;
			      row["adjoint_batch_upload_s"] =
			          timing.adjointBatchUploadSeconds;
			      row["adjoint_image_upload_s"] =
			          timing.adjointImageUploadSeconds;
			      row["adjoint_kernel_s"] = timing.adjointKernelSeconds;
			      row["adjoint_image_download_s"] =
			          timing.adjointImageDownloadSeconds;
			      row["adjoint_host_image_copy_s"] =
			          timing.adjointHostImageCopySeconds;
			      row["adjoint_update_count_s"] =
			          timing.adjointUpdateCountSeconds;
			      row["adjoint_voxel_hit_count_s"] =
			          timing.adjointVoxelHitCountSeconds;
			      row["cache_lookup_s"] = timing.cacheLookupSeconds;
			      row["cache_admission_s"] = timing.cacheAdmissionSeconds;
			      row["cache_admission_gather_s"] =
			          timing.cacheAdmissionGatherSeconds;
			      row["cache_admission_pack_s"] =
			          timing.cacheAdmissionPackSeconds;
			      row["cache_admission_batch_upload_s"] =
			          timing.cacheAdmissionBatchUploadSeconds;
			      row["cache_admission_correction_build_s"] =
			          timing.cacheAdmissionCorrectionBuildSeconds;
			      row["cache_admission_correction_fill_s"] =
			          timing.cacheAdmissionCorrectionFillSeconds;
			      row["cache_admission_correction_upload_s"] =
			          timing.cacheAdmissionCorrectionUploadSeconds;
			      row["cache_admission_correction_measurement_s"] =
			          timing.cacheAdmissionCorrectionMeasurementSeconds;
			      row["cache_admission_correction_multiplicative_s"] =
			          timing.cacheAdmissionCorrectionMultiplicativeSeconds;
			      row["cache_admission_correction_additive_s"] =
			          timing.cacheAdmissionCorrectionAdditiveSeconds;
			      row["cache_admission_correction_in_vivo_s"] =
			          timing.cacheAdmissionCorrectionInVivoSeconds;
			      row["cache_insert_s"] = timing.cacheInsertSeconds;
			      row["forward_events"] = timing.forwardEvents;
			      row["forward_batches"] = timing.forwardBatches;
			      row["adjoint_events"] = timing.adjointEvents;
			      row["adjoint_nonzero_events"] =
			          timing.adjointNonzeroEvents;
			      row["adjoint_batches"] = timing.adjointBatches;
			      row["adjoint_voxel_updates"] =
			          timing.adjointVoxelUpdates;
			      row["adjoint_rays_with_updates"] =
			          timing.adjointRaysWithUpdates;
			      row["adjoint_max_updates_per_ray"] =
			          timing.adjointMaxUpdatesPerRay;
			      row["adjoint_voxel_hit_maps"] =
			          timing.adjointVoxelHitMaps;
			      row["adjoint_batch_hit_voxels"] =
			          timing.adjointBatchHitVoxels;
			      row["adjoint_voxel_hit_total_updates"] =
			          timing.adjointVoxelHitTotalUpdates;
			      row["adjoint_max_voxel_hits"] =
			          timing.adjointMaxVoxelHits;
			      row["adjoint_max_batch_p50_voxel_hits"] =
			          timing.adjointMaxBatchP50VoxelHits;
			      row["adjoint_max_batch_p90_voxel_hits"] =
			          timing.adjointMaxBatchP90VoxelHits;
			      row["adjoint_max_batch_p95_voxel_hits"] =
			          timing.adjointMaxBatchP95VoxelHits;
			      row["adjoint_max_batch_p99_voxel_hits"] =
			          timing.adjointMaxBatchP99VoxelHits;
			      row["adjoint_max_batch_p999_voxel_hits"] =
			          timing.adjointMaxBatchP999VoxelHits;
			      row["adjoint_max_batch_mean_voxel_hits"] =
			          timing.adjointMaxBatchMeanVoxelHits;
			      row["adjoint_max_batch_top_1pct_voxel_hit_fraction"] =
			          timing.adjointMaxBatchTop1PctVoxelHitFraction;
			      row["adjoint_max_batch_top_0_1pct_voxel_hit_fraction"] =
			          timing.adjointMaxBatchTop01PctVoxelHitFraction;
			      row["adjoint_tile_size"] = timing.adjointTileSize;
			      row["adjoint_voxel_hit_tiles"] =
			          timing.adjointVoxelHitTiles;
			      row["adjoint_voxel_hit_tile_total_updates"] =
			          timing.adjointVoxelHitTileTotalUpdates;
			      row["adjoint_max_tile_hits"] =
			          timing.adjointMaxTileHits;
			      row["adjoint_max_batch_p95_tile_hits"] =
			          timing.adjointMaxBatchP95TileHits;
			      row["adjoint_max_batch_p99_tile_hits"] =
			          timing.adjointMaxBatchP99TileHits;
			      row["adjoint_max_batch_mean_tile_hits"] =
			          timing.adjointMaxBatchMeanTileHits;
			      row["adjoint_max_batch_top_1pct_tile_hit_fraction"] =
			          timing.adjointMaxBatchTop1PctTileHitFraction;
			      row["adjoint_max_batch_top_0_1pct_tile_hit_fraction"] =
			          timing.adjointMaxBatchTop01PctTileHitFraction;
			      row["cache_lookups"] = timing.cacheLookups;
			      row["cache_hits"] = timing.cacheHits;
			      row["cache_misses"] = timing.cacheMisses;
			      row["cache_builds"] = timing.cacheBuilds;
			      row["cache_skips_over_budget"] =
			          timing.cacheSkipsOverBudget;
			      row["cache_used_bytes"] = timing.cacheUsedBytes;
			      row["cache_max_bytes"] = timing.cacheMaxBytes;
			      row["cache_correction_reserve_bytes"] =
			          timing.cacheCorrectionReserveBytes;
			      row["uncached_batches"] = timing.uncachedBatches;
			      row["ratio_correction_cache_builds"] =
			          timing.ratioCorrectionCacheBuilds;
			      row["ratio_correction_cache_hits"] =
			          timing.ratioCorrectionCacheHits;
			      row["ratio_correction_cache_misses"] =
			          timing.ratioCorrectionCacheMisses;
			      row["ratio_correction_cache_bytes"] =
			          timing.ratioCorrectionCacheBytes;
			      row["ratio_values"] = timing.ratioValues;
			      row["ratio_nonzero_values"] =
			          timing.ratioNonzeroValues;
			      row["ratio_zero_values"] = timing.ratioZeroValues;
			      row["ratio_nonzero_diagnostic_batches"] =
			          timing.ratioNonzeroDiagnosticBatches;
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
	c.def("setExperimentalMetalProjectorCorrectionCacheReserveBytes",
	      &OSEM_CPU::setExperimentalMetalProjectorCorrectionCacheReserveBytes,
	      "reserve_bytes"_a);
	c.def("getExperimentalMetalProjectorCorrectionCacheReserveBytes",
	      &OSEM_CPU::getExperimentalMetalProjectorCorrectionCacheReserveBytes);
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
	c.def("setExperimentalMetalProjectorLazyCorrectionsEnabled",
	      &OSEM_CPU::setExperimentalMetalProjectorLazyCorrectionsEnabled,
	      "enabled"_a);
	c.def("isExperimentalMetalProjectorLazyCorrectionsEnabled",
	      &OSEM_CPU::isExperimentalMetalProjectorLazyCorrectionsEnabled);
	c.def("setExperimentalMetalProjectorCachedCorrectionsEnabled",
	      &OSEM_CPU::setExperimentalMetalProjectorCachedCorrectionsEnabled,
	      "enabled"_a);
	c.def("isExperimentalMetalProjectorCachedCorrectionsEnabled",
	      &OSEM_CPU::isExperimentalMetalProjectorCachedCorrectionsEnabled);
	c.def("setExperimentalMetalProjectorImagePsfEnabled",
	      &OSEM_CPU::setExperimentalMetalProjectorImagePsfEnabled,
	      "enabled"_a);
	c.def("isExperimentalMetalProjectorImagePsfEnabled",
	      &OSEM_CPU::isExperimentalMetalProjectorImagePsfEnabled);
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

bool fitsMetalImageDimension(int value)
{
	return value > 0 &&
	       static_cast<std::uint64_t>(value) <=
	           std::numeric_limits<std::uint32_t>::max();
}

bool fitsMetalImageFrameCount(frame_t value)
{
	return value > 0 &&
	       static_cast<std::uint64_t>(value) <=
	           std::numeric_limits<std::uint32_t>::max();
}

bool makeMetalImageShape(const Image& image,
                         backend::metal::ImageShape& shape)
{
	if (!image.isMemoryValid())
	{
		return false;
	}
	const ImageParams& params = image.getParams();
	if (!fitsMetalImageDimension(params.nx) ||
	    !fitsMetalImageDimension(params.ny) ||
	    !fitsMetalImageDimension(params.nz) ||
	    !fitsMetalImageFrameCount(params.nt))
	{
		return false;
	}
	shape = {static_cast<std::uint32_t>(params.nx),
	         static_cast<std::uint32_t>(params.ny),
	         static_cast<std::uint32_t>(params.nz),
	         static_cast<std::uint32_t>(params.nt)};
	return true;
}

std::size_t metalImageByteCount(const backend::metal::ImageShape& shape)
{
	return sizeof(float) * shape.voxelCount();
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
	timings.setupContextSeconds += bridgeProfile.setupContextSeconds;
	timings.setupProjectorSeconds += bridgeProfile.setupProjectorSeconds;
	timings.setupCacheSeconds += bridgeProfile.setupCacheSeconds;
	timings.setupBridgeSeconds += bridgeProfile.setupBridgeSeconds;
	timings.setupCanRunSeconds += bridgeProfile.setupCanRunSeconds;
	timings.forwardSeconds += forwardSeconds;
	timings.ratioSeconds += ratioSeconds;
	timings.adjointSeconds += adjointSeconds;
	timings.totalSeconds += totalSeconds;
	timings.metalPathOverheadSeconds += bridgeProfile.metalPathOverheadSeconds;
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
	timings.ratioCorrectionCacheBuildSeconds +=
	    bridgeProfile.ratioCorrectionCacheBuildSeconds;
	timings.ratioNonzeroDiagnosticSeconds +=
	    bridgeProfile.ratioNonzeroDiagnosticSeconds;
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
	timings.cacheLookupSeconds += bridgeProfile.cacheLookupSeconds;
	timings.cacheAdmissionSeconds += bridgeProfile.cacheAdmissionSeconds;
	timings.cacheAdmissionGatherSeconds +=
	    bridgeProfile.cacheAdmissionGatherSeconds;
	timings.cacheAdmissionPackSeconds +=
	    bridgeProfile.cacheAdmissionPackSeconds;
	timings.cacheAdmissionBatchUploadSeconds +=
	    bridgeProfile.cacheAdmissionBatchUploadSeconds;
	timings.cacheAdmissionCorrectionBuildSeconds +=
	    bridgeProfile.cacheAdmissionCorrectionBuildSeconds;
	timings.cacheAdmissionCorrectionFillSeconds +=
	    bridgeProfile.cacheAdmissionCorrectionFillSeconds;
	timings.cacheAdmissionCorrectionUploadSeconds +=
	    bridgeProfile.cacheAdmissionCorrectionUploadSeconds;
	timings.cacheAdmissionCorrectionMeasurementSeconds +=
	    bridgeProfile.cacheAdmissionCorrectionMeasurementSeconds;
	timings.cacheAdmissionCorrectionMultiplicativeSeconds +=
	    bridgeProfile.cacheAdmissionCorrectionMultiplicativeSeconds;
	timings.cacheAdmissionCorrectionAdditiveSeconds +=
	    bridgeProfile.cacheAdmissionCorrectionAdditiveSeconds;
	timings.cacheAdmissionCorrectionInVivoSeconds +=
	    bridgeProfile.cacheAdmissionCorrectionInVivoSeconds;
	timings.cacheInsertSeconds += bridgeProfile.cacheInsertSeconds;
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
	timings.adjointMaxBatchP50VoxelHits =
	    std::max(timings.adjointMaxBatchP50VoxelHits,
	             static_cast<size_t>(
	                 bridgeProfile.adjointMaxBatchP50VoxelHits));
	timings.adjointMaxBatchP90VoxelHits =
	    std::max(timings.adjointMaxBatchP90VoxelHits,
	             static_cast<size_t>(
	                 bridgeProfile.adjointMaxBatchP90VoxelHits));
	timings.adjointMaxBatchP95VoxelHits =
	    std::max(timings.adjointMaxBatchP95VoxelHits,
	             static_cast<size_t>(
	                 bridgeProfile.adjointMaxBatchP95VoxelHits));
	timings.adjointMaxBatchP99VoxelHits =
	    std::max(timings.adjointMaxBatchP99VoxelHits,
	             static_cast<size_t>(
	                 bridgeProfile.adjointMaxBatchP99VoxelHits));
	timings.adjointMaxBatchP999VoxelHits =
	    std::max(timings.adjointMaxBatchP999VoxelHits,
	             static_cast<size_t>(
	                 bridgeProfile.adjointMaxBatchP999VoxelHits));
	timings.adjointMaxBatchMeanVoxelHits =
	    std::max(timings.adjointMaxBatchMeanVoxelHits,
	             bridgeProfile.adjointMaxBatchMeanVoxelHits);
	timings.adjointMaxBatchTop1PctVoxelHitFraction =
	    std::max(timings.adjointMaxBatchTop1PctVoxelHitFraction,
	             bridgeProfile.adjointMaxBatchTop1PctVoxelHitFraction);
	timings.adjointMaxBatchTop01PctVoxelHitFraction =
	    std::max(timings.adjointMaxBatchTop01PctVoxelHitFraction,
	             bridgeProfile.adjointMaxBatchTop01PctVoxelHitFraction);
	timings.adjointTileSize =
	    std::max(timings.adjointTileSize,
	             static_cast<size_t>(bridgeProfile.adjointTileSize));
	timings.adjointVoxelHitTiles += bridgeProfile.adjointVoxelHitTiles;
	timings.adjointVoxelHitTileTotalUpdates +=
	    bridgeProfile.adjointVoxelHitTileTotalUpdates;
	timings.adjointMaxTileHits =
	    std::max(timings.adjointMaxTileHits,
	             static_cast<size_t>(bridgeProfile.adjointMaxTileHits));
	timings.adjointMaxBatchP95TileHits =
	    std::max(timings.adjointMaxBatchP95TileHits,
	             static_cast<size_t>(
	                 bridgeProfile.adjointMaxBatchP95TileHits));
	timings.adjointMaxBatchP99TileHits =
	    std::max(timings.adjointMaxBatchP99TileHits,
	             static_cast<size_t>(
	                 bridgeProfile.adjointMaxBatchP99TileHits));
	timings.adjointMaxBatchMeanTileHits =
	    std::max(timings.adjointMaxBatchMeanTileHits,
	             bridgeProfile.adjointMaxBatchMeanTileHits);
	timings.adjointMaxBatchTop1PctTileHitFraction =
	    std::max(timings.adjointMaxBatchTop1PctTileHitFraction,
	             bridgeProfile.adjointMaxBatchTop1PctTileHitFraction);
	timings.adjointMaxBatchTop01PctTileHitFraction =
	    std::max(timings.adjointMaxBatchTop01PctTileHitFraction,
	             bridgeProfile.adjointMaxBatchTop01PctTileHitFraction);
	timings.cacheLookups += bridgeProfile.cacheLookups;
	timings.cacheHits += bridgeProfile.cacheHits;
	timings.cacheMisses += bridgeProfile.cacheMisses;
	timings.cacheBuilds += bridgeProfile.cacheBuilds;
	timings.cacheSkipsOverBudget += bridgeProfile.cacheSkipsOverBudget;
	timings.cacheUsedBytes = bridgeProfile.cacheUsedBytes;
	timings.cacheMaxBytes = bridgeProfile.cacheMaxBytes;
	timings.cacheCorrectionReserveBytes =
	    bridgeProfile.cacheCorrectionReserveBytes;
	timings.uncachedBatches += bridgeProfile.uncachedBatches;
	timings.ratioCorrectionCacheBuilds +=
	    bridgeProfile.ratioCorrectionCacheBuilds;
	timings.ratioCorrectionCacheHits +=
	    bridgeProfile.ratioCorrectionCacheHits;
	timings.ratioCorrectionCacheMisses +=
	    bridgeProfile.ratioCorrectionCacheMisses;
	timings.ratioCorrectionCacheBytes +=
	    bridgeProfile.ratioCorrectionCacheBytes;
	timings.ratioValues += bridgeProfile.ratioValues;
	timings.ratioNonzeroValues += bridgeProfile.ratioNonzeroValues;
	timings.ratioZeroValues += bridgeProfile.ratioZeroValues;
	timings.ratioNonzeroDiagnosticBatches +=
	    bridgeProfile.ratioNonzeroDiagnosticBatches;
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
#if BUILD_METAL
	mp_experimentalMetalImagePsf = nullptr;
	mp_experimentalMetalImagePsfContext = nullptr;
#endif
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
#if BUILD_METAL
	mp_experimentalMetalImagePsf = nullptr;
	mp_experimentalMetalImagePsfContext = nullptr;
#endif
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
#if BUILD_METAL
	mp_experimentalMetalImagePsf = nullptr;
	mp_experimentalMetalImagePsfContext = nullptr;
#endif
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

void OSEM_CPU::setExperimentalMetalProjectorResidentImagesEnabled(bool enabled)
{
	m_experimentalMetalProjectorResidentImagesEnabled = enabled;
	if (!enabled)
	{
#if BUILD_METAL
		mp_experimentalMetalResidentOsemState = nullptr;
#endif
	}
}

bool OSEM_CPU::isExperimentalMetalProjectorResidentImagesEnabled() const
{
	return m_experimentalMetalProjectorResidentImagesEnabled;
}

void OSEM_CPU::setExperimentalMetalProjectorKernel(const std::string& kernel)
{
	if (kernel != "siddon" && kernel != "joseph" &&
	    kernel != "joseph_texture_forward")
	{
		throw std::invalid_argument(
		    "Experimental Metal projector kernel must be 'siddon', 'joseph' "
		    "or 'joseph_texture_forward'");
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

bool OSEM_CPU::isReconstructionTimingEnabled() const
{
	return m_experimentalMetalProjectorEnabled &&
	       m_experimentalMetalProjectorProfilingEnabled;
}

void OSEM_CPU::recordReconstructionTiming(ReconstructionTimingPhase phase,
                                          double seconds)
{
	if (!isReconstructionTimingEnabled())
	{
		return;
	}

	switch (phase)
	{
	case ReconstructionTimingPhase::InitializeForRecon:
		m_experimentalMetalProjectorTimings.reconInitializeSeconds += seconds;
		break;
	case ReconstructionTimingPhase::SetupForDynamicRecon:
		m_experimentalMetalProjectorTimings.reconSetupDynamicSeconds += seconds;
		break;
	case ReconstructionTimingPhase::InitializeOutImage:
		m_experimentalMetalProjectorTimings.reconInitializeOutImageSeconds +=
		    seconds;
		break;
	case ReconstructionTimingPhase::InitializeSensImage:
		m_experimentalMetalProjectorTimings.reconInitializeSensImageSeconds +=
		    seconds;
		break;
	case ReconstructionTimingPhase::CorrectorSetup:
		m_experimentalMetalProjectorTimings.reconCorrectorSetupSeconds +=
		    seconds;
		break;
	case ReconstructionTimingPhase::InitializeBinIterators:
		m_experimentalMetalProjectorTimings
		    .reconInitializeBinIteratorsSeconds += seconds;
		break;
	case ReconstructionTimingPhase::CollectConstraints:
		m_experimentalMetalProjectorTimings.reconCollectConstraintsSeconds +=
		    seconds;
		break;
	case ReconstructionTimingPhase::SetupProjector:
		m_experimentalMetalProjectorTimings.reconSetupProjectorSeconds +=
		    seconds;
		break;
	case ReconstructionTimingPhase::PrepareBuffers:
		m_experimentalMetalProjectorTimings.reconPrepareBuffersSeconds +=
		    seconds;
		break;
	case ReconstructionTimingPhase::Iterate:
		m_experimentalMetalProjectorTimings.reconIterateSeconds += seconds;
		break;
	case ReconstructionTimingPhase::LoadSubset:
		m_experimentalMetalProjectorTimings.reconLoadSubsetSeconds += seconds;
		break;
	case ReconstructionTimingPhase::ResetUpdateImage:
		m_experimentalMetalProjectorTimings.reconResetUpdateSeconds += seconds;
		break;
	case ReconstructionTimingPhase::ComputeUpdateImage:
		m_experimentalMetalProjectorTimings.reconComputeUpdatePhaseSeconds +=
		    seconds;
		break;
	case ReconstructionTimingPhase::ApplyImageUpdate:
		m_experimentalMetalProjectorTimings.reconApplyUpdatePhaseSeconds +=
		    seconds;
		break;
	case ReconstructionTimingPhase::CompleteSubset:
		m_experimentalMetalProjectorTimings.reconCompleteSubsetSeconds +=
		    seconds;
		break;
	case ReconstructionTimingPhase::SaveIteration:
		m_experimentalMetalProjectorTimings.reconSaveIterationSeconds += seconds;
		break;
	case ReconstructionTimingPhase::CompleteMLEMIteration:
		m_experimentalMetalProjectorTimings.reconCompleteMLEMSeconds += seconds;
		break;
	case ReconstructionTimingPhase::EndRecon:
		m_experimentalMetalProjectorTimings.reconEndSeconds += seconds;
		break;
	}
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

void OSEM_CPU::setExperimentalMetalProjectorCorrectionCacheReserveBytes(
    size_t reserveBytes)
{
	m_experimentalMetalProjectorCorrectionCacheReserveBytes = reserveBytes;
#if BUILD_METAL
	if (mp_experimentalMetalProjectorCache != nullptr)
	{
		mp_experimentalMetalProjectorCache
		    ->setCorrectionCacheReserveBytes(reserveBytes);
	}
#endif
}

size_t OSEM_CPU::getExperimentalMetalProjectorCorrectionCacheReserveBytes()
    const
{
	return m_experimentalMetalProjectorCorrectionCacheReserveBytes;
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

void OSEM_CPU::setExperimentalMetalProjectorLazyCorrectionsEnabled(
    bool enabled)
{
	m_experimentalMetalProjectorLazyCorrectionsEnabled = enabled;
}

bool OSEM_CPU::isExperimentalMetalProjectorLazyCorrectionsEnabled() const
{
	return m_experimentalMetalProjectorLazyCorrectionsEnabled;
}

void OSEM_CPU::setExperimentalMetalProjectorCachedCorrectionsEnabled(
    bool enabled)
{
	m_experimentalMetalProjectorCachedCorrectionsEnabled = enabled;
}

bool OSEM_CPU::isExperimentalMetalProjectorCachedCorrectionsEnabled() const
{
	return m_experimentalMetalProjectorCachedCorrectionsEnabled;
}

void OSEM_CPU::setExperimentalMetalProjectorImagePsfEnabled(bool enabled)
{
	m_experimentalMetalProjectorImagePsfEnabled = enabled;
}

bool OSEM_CPU::isExperimentalMetalProjectorImagePsfEnabled() const
{
	return m_experimentalMetalProjectorImagePsfEnabled;
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
#if BUILD_METAL
	auto timePreparePhase = [this](double& destination, auto&& fn)
	{
		if (!isReconstructionTimingEnabled())
		{
			fn();
			return;
		}
		const auto start = Clock::now();
		fn();
		destination += getElapsedSeconds(start, Clock::now());
	};
#endif

	// Allocate for projection-space buffers
	const ProjectionData* dataInput = getDataInput();

	// Allocate for image-space buffers
	auto allocateImages = [this]()
	{
		mp_mlemImageTmpEMRatio = std::make_unique<ImageOwned>(getImageParams());
		reinterpret_cast<ImageOwned*>(mp_mlemImageTmpEMRatio.get())->allocate();
		if (flagImagePSF)
		{
			mp_imageTmpPsf = std::make_unique<ImageOwned>(getImageParams());
			reinterpret_cast<ImageOwned*>(mp_imageTmpPsf.get())->allocate();
		}
	};
#if BUILD_METAL
	timePreparePhase(
	    m_experimentalMetalProjectorTimings.prepareAllocateImagesSeconds,
	    allocateImages);
#else
	allocateImages();
#endif

	// Initialize output image
	auto initializeOutput = [this]()
	{
		if (initialEstimate != nullptr)
		{
			outImage->copyFromImage(initialEstimate);
		}
		else
		{
			outImage->fill(INITIAL_VALUE_MLEM);
		}
	};
#if BUILD_METAL
	timePreparePhase(
	    m_experimentalMetalProjectorTimings.prepareInitializeOutputSeconds,
	    initializeOutput);
#else
	initializeOutput();
#endif

	// Apply mask function
	auto applyMask = [this](const Image* maskImage) -> void
	{
		outImage->applyThresholdBroadcast(maskImage, 0.0f, 0.0f, 0.0f, 1.0f,
		                                  0.0f);
	};

	// Apply mask image
	std::cout << "Applying threshold..." << std::endl;
	auto applyMaskPhase = [this, &applyMask]()
	{
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
	};
#if BUILD_METAL
	timePreparePhase(m_experimentalMetalProjectorTimings.prepareApplyMaskSeconds,
	                 applyMaskPhase);
#else
	applyMaskPhase();
#endif

	auto clearUpdate = [this]() { mp_mlemImageTmpEMRatio->fill(0.0f); };
#if BUILD_METAL
	timePreparePhase(m_experimentalMetalProjectorTimings.prepareClearUpdateSeconds,
	                 clearUpdate);
#else
	clearUpdate();
#endif

	auto precomputeCorrections = [this, dataInput]()
	{ mp_corrector->precomputeCorrectionFactors(*dataInput); };
#if BUILD_METAL
	if (!(m_experimentalMetalProjectorEnabled &&
	      (m_experimentalMetalProjectorLazyCorrectionsEnabled ||
	       m_experimentalMetalProjectorCachedCorrectionsEnabled)))
	{
		timePreparePhase(
		    m_experimentalMetalProjectorTimings
		        .preparePrecomputeCorrectionsSeconds,
		    precomputeCorrections);
	}
#else
	precomputeCorrections();
#endif

	auto initializeBinLoader = [this]() { initBinLoader(true); };
#if BUILD_METAL
	timePreparePhase(
	    m_experimentalMetalProjectorTimings.prepareInitBinLoaderSeconds,
	    initializeBinLoader);
#else
	initializeBinLoader();
#endif

#if BUILD_METAL
	auto clearMetalCache = [this]()
	{
		if (mp_experimentalMetalProjectorCache != nullptr)
		{
			mp_experimentalMetalProjectorCache->clear();
		}
		mp_experimentalMetalResidentOsemState = nullptr;
	};
	timePreparePhase(
	    m_experimentalMetalProjectorTimings.prepareClearMetalCacheSeconds,
	    clearMetalCache);
#endif
}

#if BUILD_METAL
bool OSEM_CPU::canUseExperimentalMetalImagePsf() const
{
	return m_experimentalMetalProjectorImagePsfEnabled && flagImagePSF &&
	       m_imagePSFMode == ImagePSFMode::UNIFORM &&
	       dynamic_cast<const OperatorPsf*>(imagePsf.get()) != nullptr;
}

backend::metal::OperatorPsfMetal* OSEM_CPU::getExperimentalMetalImagePsf(
    const backend::metal::Context* context)
{
	if (!canUseExperimentalMetalImagePsf())
	{
		return nullptr;
	}
	if (mp_experimentalMetalImagePsfContext != context)
	{
		mp_experimentalMetalImagePsf = nullptr;
		mp_experimentalMetalImagePsfContext = context;
	}
	if (mp_experimentalMetalImagePsf == nullptr)
	{
		const auto* uniformPsf =
		    dynamic_cast<const OperatorPsf*>(imagePsf.get());
		if (uniformPsf == nullptr)
		{
			return nullptr;
		}
		if (context != nullptr)
		{
			mp_experimentalMetalImagePsf =
			    std::make_unique<backend::metal::OperatorPsfMetal>(
			        *context, uniformPsf->getKernelX(), uniformPsf->getKernelY(),
			        uniformPsf->getKernelZ());
		}
		else
		{
			mp_experimentalMetalImagePsf =
			    std::make_unique<backend::metal::OperatorPsfMetal>(
			        uniformPsf->getKernelX(), uniformPsf->getKernelY(),
			        uniformPsf->getKernelZ());
		}
	}
	return mp_experimentalMetalImagePsf != nullptr &&
	           mp_experimentalMetalImagePsf->isValid() ?
	           mp_experimentalMetalImagePsf.get() :
	           nullptr;
}

std::string OSEM_CPU::describeExperimentalMetalImagePsfState(
    const Image& input, const Image& output)
{
	std::ostringstream stream;
	const auto& inputParams = input.getParams();
	const auto& outputParams = output.getParams();
	stream << "input_valid=" << input.isMemoryValid()
	       << " output_valid=" << output.isMemoryValid() << " input_dims="
	       << inputParams.nx << "x" << inputParams.ny << "x"
	       << inputParams.nz << "x" << inputParams.nt << " output_dims="
	       << outputParams.nx << "x" << outputParams.ny << "x"
	       << outputParams.nz << "x" << outputParams.nt
	       << " input_frames=" << input.getNumFrames()
	       << " output_frames=" << output.getNumFrames()
	       << " uniform_psf=" << canUseExperimentalMetalImagePsf();
	auto* psf = getExperimentalMetalImagePsf();
	if (psf == nullptr)
	{
		stream << " psf_valid=0";
		if (mp_experimentalMetalImagePsf != nullptr &&
		    !mp_experimentalMetalImagePsf->errorMessage().empty())
		{
			stream << " psf_error="
			       << mp_experimentalMetalImagePsf->errorMessage();
		}
	}
	else
	{
		stream << " psf_valid=" << psf->isValid()
		       << " kernel_sizes=" << psf->getKernelX().size() << "x"
		       << psf->getKernelY().size() << "x"
		       << psf->getKernelZ().size();
		if (!psf->errorMessage().empty())
		{
			stream << " psf_error=" << psf->errorMessage();
		}
	}
	return stream.str();
}

bool OSEM_CPU::applyExperimentalMetalImagePsfForward(const Image& input,
                                                     Image& output)
{
	auto* psf = getExperimentalMetalImagePsf();
	return psf != nullptr && psf->applyA(input, output);
}

bool OSEM_CPU::applyExperimentalMetalImagePsfAdjoint(const Image& input,
                                                     Image& output)
{
	auto* psf = getExperimentalMetalImagePsf();
	return psf != nullptr && psf->applyAH(input, output);
}

bool OSEM_CPU::applyExperimentalMetalResidentImagePsfForward(
    const backend::metal::Context& context,
    const backend::metal::Buffer& input, backend::metal::Buffer& output,
    const backend::metal::ImageShape& shape)
{
	auto* psf = getExperimentalMetalImagePsf(&context);
	return psf != nullptr && psf->applyA(input, output, shape);
}

bool OSEM_CPU::applyExperimentalMetalResidentImagePsfAdjoint(
    const backend::metal::Context& context,
    const backend::metal::Buffer& input, backend::metal::Buffer& output,
    const backend::metal::ImageShape& shape)
{
	auto* psf = getExperimentalMetalImagePsf(&context);
	return psf != nullptr && psf->applyAH(input, output, shape);
}

bool OSEM_CPU::isExperimentalMetalResidentImagesAllowedForCurrentState() const
{
	if (!m_experimentalMetalProjectorResidentImagesEnabled ||
	    !m_experimentalMetalProjectorEnabled ||
	    m_experimentalMetalProjectorFusedRatioEnabled ||
	    (flagImagePSF && !canUseExperimentalMetalImagePsf()) ||
	    !saveIterRanges.empty() || outImage == nullptr ||
	    mp_mlemImageTmpEMRatio == nullptr)
	{
		return false;
	}

	const Image* image = dynamic_cast<const Image*>(outImage.get());
	const Image* updateImage =
	    dynamic_cast<const Image*>(mp_mlemImageTmpEMRatio.get());
	if (image == nullptr || updateImage == nullptr ||
	    !image->getParams().isSameAs(updateImage->getParams()))
	{
		return false;
	}

	backend::metal::ImageShape shape{};
	backend::metal::ImageShape updateShape{};
	return makeMetalImageShape(*image, shape) &&
	       makeMetalImageShape(*updateImage, updateShape);
}

bool OSEM_CPU::tryResetExperimentalMetalResidentUpdateImage()
{
	if (!isExperimentalMetalResidentImagesAllowedForCurrentState())
	{
		return false;
	}

	Image* updateImage = dynamic_cast<Image*>(mp_mlemImageTmpEMRatio.get());
	if (updateImage == nullptr)
	{
		return false;
	}

	if (mp_experimentalMetalResidentOsemState == nullptr)
	{
		mp_experimentalMetalResidentOsemState =
		    std::make_unique<ExperimentalMetalResidentOsemState>();
	}
	auto& state = *mp_experimentalMetalResidentOsemState;
	if (!state.context.isValid())
	{
		return false;
	}

	backend::metal::ImageShape updateShape{};
	if (!makeMetalImageShape(*updateImage, updateShape))
	{
		return false;
	}

	const std::size_t byteCount = metalImageByteCount(updateShape);
	if (!state.updateAllocated ||
	    !state.updateParams.isSameAs(updateImage->getParams()) ||
	    state.updateBuffer.byteCount() < byteCount)
	{
		state.updateBuffer =
		    backend::metal::Buffer::allocate(state.context.device(), byteCount);
		state.updateAllocated = state.updateBuffer.isValid();
		state.updateParams = updateImage->getParams();
		state.updateReady = false;
	}
	if (!state.updateAllocated ||
	    !backend::metal::launchImageFill(state.context.device(),
	        state.context.library(), state.context.commandQueue(),
	        state.updateBuffer, updateShape, 0.0f))
	{
		state.updateReady = false;
		return false;
	}

	state.updateReady = true;
	return true;
}

bool OSEM_CPU::downloadExperimentalMetalResidentImage()
{
	if (mp_experimentalMetalResidentOsemState == nullptr ||
	    !mp_experimentalMetalResidentOsemState->hostImageStale)
	{
		return true;
	}

	auto& state = *mp_experimentalMetalResidentOsemState;
	Image* image = dynamic_cast<Image*>(outImage.get());
	if (image == nullptr || !state.context.isValid() ||
	    !state.imageBuffer.isValid())
	{
		return false;
	}

	backend::metal::SiddonProjectorKernelProfile transferProfile;
	const bool didDownload = backend::metal::downloadSiddonImageBuffer(
	    state.context, state.imageBuffer, *image,
	    m_experimentalMetalProjectorProfilingEnabled ? &transferProfile :
	                                                   nullptr);
	if (!didDownload)
	{
		return false;
	}

	if (m_experimentalMetalProjectorProfilingEnabled)
	{
		m_experimentalMetalProjectorTimings.adjointImageDownloadSeconds +=
		    transferProfile.imageDownloadSeconds;
	}
	state.hostImageStale = false;
	state.imageUploaded = true;
	state.imageParams = image->getParams();
	return true;
}

bool OSEM_CPU::ensureExperimentalMetalResidentProjectorBuffers(
    const Image& inputImage, const Image& updateImage,
    backend::metal::OperatorProjectorMetalProfile& bridgeProfile)
{
	if (!isExperimentalMetalResidentImagesAllowedForCurrentState())
	{
		return false;
	}
	if (mp_experimentalMetalResidentOsemState == nullptr)
	{
		mp_experimentalMetalResidentOsemState =
		    std::make_unique<ExperimentalMetalResidentOsemState>();
	}
	auto& state = *mp_experimentalMetalResidentOsemState;
	if (!state.context.isValid())
	{
		return false;
	}

	backend::metal::ImageShape inputShape{};
	if (!makeMetalImageShape(inputImage, inputShape) ||
	    !inputImage.getParams().isSameAs(updateImage.getParams()))
	{
		return false;
	}

	const std::size_t inputByteCount = metalImageByteCount(inputShape);
	const bool needsUpload =
	    !state.imageUploaded || !state.imageBuffer.isValid() ||
	    state.imageBuffer.byteCount() < inputByteCount ||
	    !state.imageParams.isSameAs(inputImage.getParams());
	if (needsUpload)
	{
		if (state.hostImageStale)
		{
			return false;
		}
		backend::metal::SiddonProjectorKernelProfile transferProfile;
		if (!backend::metal::uploadSiddonImageBuffer(state.context,
		        inputImage, state.imageBuffer,
		        m_experimentalMetalProjectorProfilingEnabled ?
		            &transferProfile :
		            nullptr))
		{
			return false;
		}
		if (m_experimentalMetalProjectorProfilingEnabled)
		{
			bridgeProfile.forwardImageUploadSeconds +=
			    transferProfile.imageUploadSeconds;
		}
		state.imageUploaded = true;
		state.imageParams = inputImage.getParams();
	}

	if (!state.updateReady)
	{
		return tryResetExperimentalMetalResidentUpdateImage();
	}
	return state.updateBuffer.isValid();
}

bool OSEM_CPU::applyExperimentalMetalResidentImageUpdate(
    const ImageBase* sensitivityImage)
{
	if (!isExperimentalMetalResidentImagesAllowedForCurrentState() ||
	    mp_experimentalMetalResidentOsemState == nullptr ||
	    sensitivityImage == nullptr)
	{
		return false;
	}

	auto& state = *mp_experimentalMetalResidentOsemState;
	Image* image = dynamic_cast<Image*>(outImage.get());
	const Image* sensitivity = dynamic_cast<const Image*>(sensitivityImage);
	if (image == nullptr || sensitivity == nullptr || !state.context.isValid() ||
	    !state.imageBuffer.isValid() || !state.updateBuffer.isValid() ||
	    !state.updateReady)
	{
		return false;
	}
	if (image->getParams().nt > 1 && sensitivity->getParams().nt > 1)
	{
		return false;
	}

	backend::metal::ImageShape imageShape{};
	if (!makeMetalImageShape(*image, imageShape))
	{
		return false;
	}

	const bool needsSensitivityUpload =
	    !state.sensitivityUploaded || !state.sensitivityBuffer.isValid() ||
	    state.sensitivityImage != sensitivityImage ||
	    !state.sensitivityParams.isSameAs(sensitivity->getParams());
	if (needsSensitivityUpload)
	{
		if (!backend::metal::uploadSiddonImageBuffer(
		        state.context, *sensitivity, state.sensitivityBuffer, nullptr))
		{
			return false;
		}
		state.sensitivityUploaded = true;
		state.sensitivityImage = sensitivityImage;
		state.sensitivityParams = sensitivity->getParams();
	}

	const bool didUpdate =
	    imageShape.nt == 1 ?
	        backend::metal::launchImageUpdateEMStatic(
	            state.context.device(), state.context.library(),
	            state.context.commandQueue(), state.updateBuffer,
	            state.imageBuffer, state.sensitivityBuffer, imageShape,
	            EPS_FLT) :
	        backend::metal::launchImageUpdateEMDynamic(
	            state.context.device(), state.context.library(),
	            state.context.commandQueue(), state.updateBuffer,
	            state.imageBuffer, state.sensitivityBuffer, imageShape,
	            EPS_FLT);
	if (!didUpdate)
	{
		return false;
	}

	state.hostImageStale = true;
	state.updateReady = false;
	return true;
}
#endif

void OSEM_CPU::loadCurrentSubset(bool /*forRecon*/) {}

void OSEM_CPU::resetEMUpdateImage()
{
#if BUILD_METAL
	if (tryResetExperimentalMetalResidentUpdateImage())
	{
		return;
	}
#endif
	mp_mlemImageTmpEMRatio->fill(0.0);
}

void OSEM_CPU::computeEMUpdateImage()
{
	m_experimentalMetalProjectorRanLastCompute = false;
#if BUILD_METAL
	const auto computeUpdateStart = Clock::now();
	const bool useExperimentalMetalImagePsf =
	    canUseExperimentalMetalImagePsf();
	const bool useExperimentalMetalResidentImagePsf =
	    useExperimentalMetalImagePsf &&
	    isExperimentalMetalResidentImagesAllowedForCurrentState();
#else
	const bool useExperimentalMetalImagePsf = false;
	const bool useExperimentalMetalResidentImagePsf = false;
#endif

	if (flagImagePSF)
	{
		if (!useExperimentalMetalResidentImagePsf)
		{
			mp_imageTmpPsf->fill(0.0);
#if BUILD_METAL
			if (useExperimentalMetalImagePsf)
			{
				const Image* psfInput =
				    dynamic_cast<const Image*>(outImage.get());
				Image* psfOutput = dynamic_cast<Image*>(mp_imageTmpPsf.get());
				if (psfInput == nullptr || psfOutput == nullptr ||
				    !applyExperimentalMetalImagePsfForward(
				        *psfInput, *psfOutput))
				{
					std::string details;
					if (psfInput != nullptr && psfOutput != nullptr)
					{
						details = describeExperimentalMetalImagePsfState(
						    *psfInput, *psfOutput);
					}
					throw std::runtime_error(
					    "Experimental Metal image PSF forward apply failed: " +
					    details);
				}
			}
			else
#endif
			{
				imagePsf->applyA(outImage.get(), mp_imageTmpPsf.get());
			}

			// We swap here so that the outImage buffer stores the MLEM image
			// with the PSF applied to it and the imageTmpPsf buffer stores the
			// original MLEM image.
			outImage.swap(mp_imageTmpPsf);
		}
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
	const bool useNonPrecomputedMetalCorrections =
	    m_experimentalMetalProjectorEnabled &&
	    (m_experimentalMetalProjectorLazyCorrectionsEnabled ||
	     m_experimentalMetalProjectorCachedCorrectionsEnabled) &&
	    (!flagImagePSF || useExperimentalMetalImagePsf);

	if (!useNonPrecomputedMetalCorrections)
	{
		corrector.assertMeasurementsMatchCache(measurements);
	}

	bool computedWithExperimentalMetalProjector = false;
	if (m_experimentalMetalProjectorEnabled &&
	    (!flagImagePSF || useExperimentalMetalImagePsf))
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

	if (!computedWithExperimentalMetalProjector &&
	    useExperimentalMetalResidentImagePsf)
	{
		throw std::runtime_error(
		    "Experimental Metal resident image PSF path failed");
	}

#if BUILD_METAL
	if (!computedWithExperimentalMetalProjector &&
	    mp_experimentalMetalResidentOsemState != nullptr &&
	    mp_experimentalMetalResidentOsemState->hostImageStale)
	{
		if (!downloadExperimentalMetalResidentImage())
		{
			throw std::runtime_error(
			    "Experimental Metal resident image path could not download "
			    "the stale host image before CPU fallback");
		}
	}
#endif

	if (!computedWithExperimentalMetalProjector)
	{
#if BUILD_METAL
		if (m_experimentalMetalProjectorResidentImagesEnabled)
		{
			mp_mlemImageTmpEMRatio->fill(0.0f);
		}
#endif
		if (useNonPrecomputedMetalCorrections)
		{
			mp_corrector->precomputeCorrectionFactors(*measurements);
			corrector.assertMeasurementsMatchCache(measurements);
		}
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
		if (!useExperimentalMetalResidentImagePsf)
		{
			// We swap again here so that outImage gets back to storing the
			// original MLEM image. This way, we can use the imageTmpPsf buffer
			// to compute the PSF.
			mp_imageTmpPsf.swap(outImage);

			// YN: Is this initialization necessary ?
			mp_imageTmpPsf->fill(0.0);
#if BUILD_METAL
			if (useExperimentalMetalImagePsf)
			{
				const Image* psfInput =
				    dynamic_cast<const Image*>(mp_mlemImageTmpEMRatio.get());
				Image* psfOutput = dynamic_cast<Image*>(mp_imageTmpPsf.get());
				if (psfInput == nullptr || psfOutput == nullptr ||
				    !applyExperimentalMetalImagePsfAdjoint(
				        *psfInput, *psfOutput))
				{
					std::string details;
					if (psfInput != nullptr && psfOutput != nullptr)
					{
						details = describeExperimentalMetalImagePsfState(
						    *psfInput, *psfOutput);
					}
					throw std::runtime_error(
					    "Experimental Metal image PSF adjoint apply failed: " +
					    details);
				}
			}
			else
#endif
			{
				imagePsf->applyAH(mp_mlemImageTmpEMRatio.get(),
				                  mp_imageTmpPsf.get());
			}

			// We swap these two buffers so that we can use mlemImageTmpEMRatio
			// to apply the image update.
			mp_mlemImageTmpEMRatio.swap(mp_imageTmpPsf);
		}
	}

#if BUILD_METAL
	if (computedWithExperimentalMetalProjector &&
	    m_experimentalMetalProjectorProfilingEnabled)
	{
		const double computeUpdateSeconds =
		    getElapsedSeconds(computeUpdateStart, Clock::now());
		m_experimentalMetalProjectorTimings.computeUpdateImageSeconds +=
		    computeUpdateSeconds;
		if (!m_experimentalMetalProjectorSubsetTimings.empty())
		{
			auto& subsetTiming = m_experimentalMetalProjectorSubsetTimings.back();
			if (subsetTiming.iteration == getCurrentMLEMIteration() &&
			    subsetTiming.subset == getCurrentOSEMSubset())
			{
				subsetTiming.computeUpdateImageSeconds += computeUpdateSeconds;
			}
		}
	}
#endif
}

bool OSEM_CPU::computeEMUpdateImageWithExperimentalMetalProjector(
    const Image& inputImageForForwardProj, Image& destImageForBackproj,
    const ProjectionData& measurements, const BinIterator& binIter,
    const Corrector_CPU& corrector, float globalScaleFactor,
    bool hasSensitivity, bool hasAttenuation, bool hasScatterEstimates,
    bool hasRandomsEstimates, bool hasInVivoAttenuation)
{
#if BUILD_METAL
	if ((flagImagePSF && !canUseExperimentalMetalImagePsf()) ||
	    mp_binLoader == nullptr)
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
	backend::metal::OperatorProjectorMetalProfile bridgeProfile;
	bridgeProfile.diagnoseAdjointUpdateCounts =
	    m_experimentalMetalProjectorAdjointDiagnosticsEnabled;
	bridgeProfile.diagnoseAdjointVoxelHits =
	    m_experimentalMetalProjectorAdjointHitDiagnosticsEnabled;

	const auto setupStart = Clock::now();
	const bool useResidentImages =
	    isExperimentalMetalResidentImagesAllowedForCurrentState();
	const auto contextStart = Clock::now();
	std::unique_ptr<backend::metal::Context> localContext;
	const backend::metal::Context* context = nullptr;
	if (useResidentImages)
	{
		if (mp_experimentalMetalResidentOsemState == nullptr)
		{
			mp_experimentalMetalResidentOsemState =
			    std::make_unique<ExperimentalMetalResidentOsemState>();
		}
		context = &mp_experimentalMetalResidentOsemState->context;
	}
	else
	{
		localContext = std::make_unique<backend::metal::Context>();
		context = localContext.get();
	}
	bridgeProfile.setupContextSeconds =
	    getElapsedSeconds(contextStart, Clock::now());
	if (context == nullptr || !context->isValid())
	{
		return false;
	}

	const auto projectorStart = Clock::now();
	std::vector<Constraint*> constraints = getConstraintsAsVectorOfPointers();
	OperatorProjector projector(projectorParams, &binIter, constraints);
	const BinLoader* bridgeBinLoader = projector.getBinLoader();
	bridgeProfile.setupProjectorSeconds =
	    getElapsedSeconds(projectorStart, Clock::now());
	if (bridgeBinLoader == nullptr)
	{
		return false;
	}

	const auto cacheStart = Clock::now();
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
		mp_experimentalMetalProjectorCache->setCorrectionCacheReserveBytes(
		    m_experimentalMetalProjectorCorrectionCacheReserveBytes);
		mp_experimentalMetalProjectorCache->setMaxBatchEvents(
		    m_experimentalMetalProjectorMaxBatchEvents);
		bridgeCache = mp_experimentalMetalProjectorCache.get();
	}
	bridgeProfile.setupCacheSeconds =
	    getElapsedSeconds(cacheStart, Clock::now());
	const auto bridgeStart = Clock::now();
	const backend::metal::OperatorProjectorMetalBridge bridge(
	    *context, m_experimentalMetalProjectorProfilingEnabled ? &bridgeProfile
	                                                           : nullptr,
	    bridgeCache);
	bridgeProfile.setupBridgeSeconds =
	    getElapsedSeconds(bridgeStart, Clock::now());
	const auto canRunStart = Clock::now();
	const auto support = bridge.canRunSiddon(projector);
	bridgeProfile.setupCanRunSeconds =
	    getElapsedSeconds(canRunStart, Clock::now());
	if (!support.supported)
	{
		return false;
	}

	setupSeconds = getElapsedSeconds(setupStart, Clock::now());
	const bool residentImagesActive =
	    useResidentImages &&
	    ensureExperimentalMetalResidentProjectorBuffers(
	        inputImageForForwardProj, destImageForBackproj, bridgeProfile);
	const backend::metal::Buffer* residentInputImageBuffer = nullptr;
	backend::metal::ImageShape residentImageShape{};
	if (residentImagesActive && mp_experimentalMetalResidentOsemState != nullptr)
	{
		auto& state = *mp_experimentalMetalResidentOsemState;
		if (!makeMetalImageShape(inputImageForForwardProj, residentImageShape))
		{
			return false;
		}
		if (flagImagePSF)
		{
			if (!applyExperimentalMetalResidentImagePsfForward(
			        state.context, state.imageBuffer, state.psfForwardBuffer,
			        residentImageShape))
			{
				return false;
			}
			residentInputImageBuffer = &state.psfForwardBuffer;
		}
		else
		{
			residentInputImageBuffer = &state.imageBuffer;
		}
	}

	auto metalProjectorKernel =
	    backend::metal::OperatorProjectorMetalKernel::Siddon;
	if (m_experimentalMetalProjectorKernel == "joseph")
	{
		metalProjectorKernel =
		    backend::metal::OperatorProjectorMetalKernel::Joseph;
	}
	else if (m_experimentalMetalProjectorKernel == "joseph_texture_forward")
	{
		metalProjectorKernel =
		    backend::metal::OperatorProjectorMetalKernel::JosephTextureForward;
	}
	bool didRun = false;
	const bool usePrecomputedCorrections =
	    !(m_experimentalMetalProjectorLazyCorrectionsEnabled ||
	      m_experimentalMetalProjectorCachedCorrectionsEnabled);
	const bool cacheCorrectionFactors =
	    m_experimentalMetalProjectorCachedCorrectionsEnabled;
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
		    usePrecomputedCorrections,
		    cacheCorrectionFactors,
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
		    usePrecomputedCorrections,
		    cacheCorrectionFactors,
		    metalProjectorKernel};
		if (residentImagesActive &&
		    mp_experimentalMetalResidentOsemState != nullptr)
		{
			if (residentInputImageBuffer == nullptr)
			{
				return false;
			}
			didRun = bridge.applyOsemEMUpdateHostRatioWithBuffers(projector,
			    inputImageForForwardProj, *residentInputImageBuffer,
			    destImageForBackproj,
			    mp_experimentalMetalResidentOsemState->updateBuffer,
			    measurements, binIter, *bridgeBinLoader, corrector,
			    metalOsemConfig);
			if (didRun && flagImagePSF)
			{
				auto& state = *mp_experimentalMetalResidentOsemState;
				if (!applyExperimentalMetalResidentImagePsfAdjoint(
				        state.context, state.updateBuffer, state.psfUpdateBuffer,
				        residentImageShape))
				{
					return false;
				}
				std::swap(state.updateBuffer, state.psfUpdateBuffer);
				state.updateReady = true;
			}
		}
		else
		{
			didRun = bridge.applyOsemEMUpdateHostRatio(projector,
			    inputImageForForwardProj, destImageForBackproj, measurements,
			    binIter, *bridgeBinLoader, corrector, metalOsemConfig);
		}

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
	if (!didRun && !m_experimentalMetalProjectorFusedRatioEnabled &&
	    (m_experimentalMetalProjectorLazyCorrectionsEnabled ||
	     m_experimentalMetalProjectorCachedCorrectionsEnabled))
	{
		return false;
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
		const double accountedSeconds =
		    setupSeconds + forwardSeconds + ratioSeconds + adjointSeconds +
		    bridgeProfile.ratioNonzeroDiagnosticSeconds;
		bridgeProfile.metalPathOverheadSeconds =
		    std::max(0.0, totalSeconds - accountedSeconds);
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
#if BUILD_METAL
	const bool profileExperimentalMetalImageUpdate =
	    m_experimentalMetalProjectorProfilingEnabled &&
	    m_experimentalMetalProjectorRanLastCompute;
	const auto imageUpdateStart = Clock::now();
#endif

	// Apply update using the correct sensitivity image
	const ImageBase* sensImage = getSensImageBuffer();

#if BUILD_METAL
	if (m_experimentalMetalProjectorRanLastCompute &&
	    mp_experimentalMetalResidentOsemState != nullptr &&
	    mp_experimentalMetalResidentOsemState->updateReady)
	{
		if (!applyExperimentalMetalResidentImageUpdate(sensImage))
		{
			throw std::runtime_error(
			    "Experimental Metal resident image update failed");
		}
		if (profileExperimentalMetalImageUpdate)
		{
			const double imageUpdateSeconds =
			    getElapsedSeconds(imageUpdateStart, Clock::now());
			m_experimentalMetalProjectorTimings.imageUpdateSeconds +=
			    imageUpdateSeconds;
			if (!m_experimentalMetalProjectorSubsetTimings.empty())
			{
				auto& subsetTiming =
				    m_experimentalMetalProjectorSubsetTimings.back();
				if (subsetTiming.iteration == getCurrentMLEMIteration() &&
				    subsetTiming.subset == getCurrentOSEMSubset())
				{
					subsetTiming.imageUpdateSeconds += imageUpdateSeconds;
				}
			}
		}
		return;
	}
#endif

	// Apply the update on the outImage buffer
	outImage->updateEMThresholdDynamic(mp_mlemImageTmpEMRatio.get(), sensImage,
	                                   EPS_FLT);

#if BUILD_METAL
	if (profileExperimentalMetalImageUpdate)
	{
		const double imageUpdateSeconds =
		    getElapsedSeconds(imageUpdateStart, Clock::now());
		m_experimentalMetalProjectorTimings.imageUpdateSeconds +=
		    imageUpdateSeconds;
		if (!m_experimentalMetalProjectorSubsetTimings.empty())
		{
			auto& subsetTiming = m_experimentalMetalProjectorSubsetTimings.back();
			if (subsetTiming.iteration == getCurrentMLEMIteration() &&
			    subsetTiming.subset == getCurrentOSEMSubset())
			{
				subsetTiming.imageUpdateSeconds += imageUpdateSeconds;
			}
		}
	}
#endif
}


void OSEM_CPU::completeSubset() {}

void OSEM_CPU::completeMLEMIteration() {}

void OSEM_CPU::endRecon()
{
#if BUILD_METAL
	if (!downloadExperimentalMetalResidentImage())
	{
		throw std::runtime_error(
		    "Experimental Metal resident image path could not download the "
		    "final image");
	}
	mp_experimentalMetalResidentOsemState = nullptr;
#endif
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
