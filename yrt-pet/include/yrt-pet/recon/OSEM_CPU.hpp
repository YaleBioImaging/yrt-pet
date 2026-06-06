/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/BinLoader.hpp"
#include "yrt-pet/operators/Projector.hpp"
#include "yrt-pet/recon/Corrector_CPU.hpp"
#include "yrt-pet/recon/OSEM.hpp"

#include <string>
#include <vector>

#if BUILD_METAL
#include "yrt-pet/backends/metal/OperatorProjectorMetalBridge.hpp"
#endif

namespace yrt
{
#if BUILD_METAL
namespace backend::metal
{
class Context;
class Buffer;
class OperatorPsfMetal;
struct ImageShape;
}
#endif

class OSEM_CPU : public OSEM
{
public:
	struct ExperimentalMetalProjectorTimings
	{
		double setupSeconds = 0.0;
		double setupContextSeconds = 0.0;
		double setupProjectorSeconds = 0.0;
		double setupCacheSeconds = 0.0;
		double setupBridgeSeconds = 0.0;
		double setupCanRunSeconds = 0.0;
		double forwardSeconds = 0.0;
		double ratioSeconds = 0.0;
		double adjointSeconds = 0.0;
		double totalSeconds = 0.0;
		double metalPathOverheadSeconds = 0.0;
		double computeUpdateImageSeconds = 0.0;
		double imageUpdateSeconds = 0.0;
		double reconInitializeSeconds = 0.0;
		double reconSetupDynamicSeconds = 0.0;
		double reconInitializeOutImageSeconds = 0.0;
		double reconInitializeSensImageSeconds = 0.0;
		double reconCorrectorSetupSeconds = 0.0;
		double reconInitializeBinIteratorsSeconds = 0.0;
		double reconCollectConstraintsSeconds = 0.0;
		double reconSetupProjectorSeconds = 0.0;
		double reconPrepareBuffersSeconds = 0.0;
		double reconIterateSeconds = 0.0;
		double reconLoadSubsetSeconds = 0.0;
		double reconResetUpdateSeconds = 0.0;
		double reconComputeUpdatePhaseSeconds = 0.0;
		double reconApplyUpdatePhaseSeconds = 0.0;
		double reconCompleteSubsetSeconds = 0.0;
		double reconSaveIterationSeconds = 0.0;
		double reconCompleteMLEMSeconds = 0.0;
		double reconEndSeconds = 0.0;
		double prepareAllocateImagesSeconds = 0.0;
		double prepareInitializeOutputSeconds = 0.0;
		double prepareApplyMaskSeconds = 0.0;
		double prepareClearUpdateSeconds = 0.0;
		double preparePrecomputeCorrectionsSeconds = 0.0;
		double prepareInitBinLoaderSeconds = 0.0;
		double prepareClearMetalCacheSeconds = 0.0;
		size_t calls = 0;
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
		double imagePsfForwardSeconds = 0.0;
		double imagePsfAdjointSeconds = 0.0;
		double ratioPackSeconds = 0.0;
		double ratioBatchUploadSeconds = 0.0;
		double ratioKernelSeconds = 0.0;
		double ratioCorrectionCacheBuildSeconds = 0.0;
		double ratioNonzeroDiagnosticSeconds = 0.0;
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
		size_t adjointDiagnosticBatchesSeen = 0;
		size_t adjointDiagnosticBatchesProfiled = 0;
		size_t adjointDiagnosticBatchesSkipped = 0;
		size_t adjointDiagnosticMaxBatches = 0;
		size_t adjointDiagnosticStride = 1;
		double cacheLookupSeconds = 0.0;
		double cacheAdmissionSeconds = 0.0;
		double cacheAdmissionGatherSeconds = 0.0;
		double cacheAdmissionPackSeconds = 0.0;
		double cacheAdmissionBatchUploadSeconds = 0.0;
		double cacheAdmissionCorrectionBuildSeconds = 0.0;
		double cacheAdmissionCorrectionFillSeconds = 0.0;
		double cacheAdmissionCorrectionUploadSeconds = 0.0;
		double cacheAdmissionCorrectionMeasurementSeconds = 0.0;
		double cacheAdmissionCorrectionMultiplicativeSeconds = 0.0;
		double cacheAdmissionCorrectionAdditiveSeconds = 0.0;
		double cacheAdmissionCorrectionInVivoSeconds = 0.0;
		double cacheInsertSeconds = 0.0;
		size_t forwardEvents = 0;
		size_t forwardBatches = 0;
		size_t adjointEvents = 0;
		size_t adjointNonzeroEvents = 0;
		size_t adjointBatches = 0;
		size_t adjointVoxelUpdates = 0;
		size_t adjointRaysWithUpdates = 0;
		size_t adjointMaxUpdatesPerRay = 0;
		size_t adjointVoxelHitMaps = 0;
		size_t adjointBatchHitVoxels = 0;
		size_t adjointVoxelHitTotalUpdates = 0;
		size_t adjointMaxVoxelHits = 0;
		size_t adjointMaxBatchP50VoxelHits = 0;
		size_t adjointMaxBatchP90VoxelHits = 0;
		size_t adjointMaxBatchP95VoxelHits = 0;
		size_t adjointMaxBatchP99VoxelHits = 0;
		size_t adjointMaxBatchP999VoxelHits = 0;
		double adjointMaxBatchMeanVoxelHits = 0.0;
		double adjointMaxBatchTop1PctVoxelHitFraction = 0.0;
		double adjointMaxBatchTop01PctVoxelHitFraction = 0.0;
		size_t adjointTileSize = 0;
		size_t adjointVoxelHitTiles = 0;
		size_t adjointVoxelHitTileTotalUpdates = 0;
		size_t adjointMaxTileHits = 0;
		size_t adjointMaxBatchP95TileHits = 0;
		size_t adjointMaxBatchP99TileHits = 0;
		double adjointMaxBatchMeanTileHits = 0.0;
		double adjointMaxBatchTop1PctTileHitFraction = 0.0;
		double adjointMaxBatchTop01PctTileHitFraction = 0.0;
		size_t cacheLookups = 0;
		size_t cacheHits = 0;
		size_t cacheMisses = 0;
		size_t cacheBuilds = 0;
		size_t cacheSkipsOverBudget = 0;
		size_t cacheUsedBytes = 0;
		size_t cacheMaxBytes = 0;
		size_t cacheCorrectionReserveBytes = 0;
		size_t uncachedBatches = 0;
		size_t ratioCorrectionCacheBuilds = 0;
		size_t ratioCorrectionCacheHits = 0;
		size_t ratioCorrectionCacheMisses = 0;
		size_t ratioCorrectionCacheBytes = 0;
		size_t ratioValues = 0;
		size_t ratioNonzeroValues = 0;
		size_t ratioZeroValues = 0;
		size_t ratioNonzeroDiagnosticBatches = 0;
	};

	struct ExperimentalMetalProjectorMemorySnapshot
	{
		bool available = false;
		size_t totalBytes = 0;
		size_t availableBytes = 0;
		size_t usedBytes = 0;
		size_t freeBytes = 0;
		size_t speculativeBytes = 0;
		size_t activeBytes = 0;
		size_t inactiveBytes = 0;
		size_t wiredBytes = 0;
		size_t compressedBytes = 0;
		size_t pageins = 0;
		size_t pageouts = 0;
		size_t decompressions = 0;
		size_t compressions = 0;
		size_t swapins = 0;
		size_t swapouts = 0;
		double availableRatio = 0.0;
		double freeRatio = 0.0;
		double compressedRatio = 0.0;
		std::string pressureLevel;
	};

	struct ExperimentalMetalProjectorSubsetTiming
	    : public ExperimentalMetalProjectorTimings
	{
		int iteration = -1;
		int subset = -1;
		size_t events = 0;
		bool metalRan = false;
		ExperimentalMetalProjectorMemorySnapshot memoryBefore;
		ExperimentalMetalProjectorMemorySnapshot memoryAfter;
	};

	struct ExperimentalMetalProjectorOptions
	{
		bool enabled = false;
		bool fusedRatio = false;
		bool residentImages = false;
		std::string kernel = "siddon";
		bool profiling = false;
		bool adjointDiagnostics = false;
		bool adjointHitDiagnostics = false;
		bool cacheEnabled = true;
		bool lazyCorrections = false;
		bool cachedCorrections = false;
		bool imagePsf = false;
		size_t cacheMaxBytes = static_cast<size_t>(1024) * 1024 * 1024;
		size_t correctionCacheReserveBytes = 0;
		size_t maxBatchEvents = 1000000;
		bool directFrameBatchesExplicit = false;
		bool directFrameBatches = false;
		bool nativeFloatAtomicsExplicit = false;
		bool nativeFloatAtomics = false;
		bool josephAdjointAxisSwitchOnceExplicit = false;
		bool josephAdjointAxisSwitchOnce = false;
		bool threadsPerThreadgroupExplicit = false;
		size_t threadsPerThreadgroup = 0;
	};

	explicit OSEM_CPU(const Scanner& pr_scanner);
	~OSEM_CPU() override;

	void addImagePSF(const std::string& p_imagePsf_fname,
	                 ImagePSFMode p_imagePSFMode) override;
	void addUniformGaussianImagePSFFromFWHM(
	    float fwhmX, float fwhmY, float fwhmZ, const size_t* kerSizeX = nullptr,
	    const size_t* kerSizeY = nullptr,
	    const size_t* kerSizeZ = nullptr) override;
	void addUniformGaussianImagePSFFromSigma(
	    float sigmaX, float sigmaY, float sigmaZ,
	    const size_t* kerSizeX = nullptr, const size_t* kerSizeY = nullptr,
	    const size_t* kerSizeZ = nullptr) override;

	void setExperimentalMetalProjectorEnabled(bool enabled);
	bool isExperimentalMetalProjectorEnabled() const;
	bool didLastExperimentalMetalProjectorRun() const;
	void setExperimentalMetalProjectorFusedRatioEnabled(bool enabled);
	bool isExperimentalMetalProjectorFusedRatioEnabled() const;
	void setExperimentalMetalProjectorResidentImagesEnabled(bool enabled);
	bool isExperimentalMetalProjectorResidentImagesEnabled() const;
	void setExperimentalMetalProjectorKernel(const std::string& kernel);
	std::string getExperimentalMetalProjectorKernel() const;
	void setExperimentalMetalProjectorProfilingEnabled(bool enabled);
	bool isExperimentalMetalProjectorProfilingEnabled() const;
	void setExperimentalMetalProjectorAdjointDiagnosticsEnabled(bool enabled);
	bool isExperimentalMetalProjectorAdjointDiagnosticsEnabled() const;
	void setExperimentalMetalProjectorAdjointHitDiagnosticsEnabled(
	    bool enabled);
	bool isExperimentalMetalProjectorAdjointHitDiagnosticsEnabled() const;
	void resetExperimentalMetalProjectorTimings();
	ExperimentalMetalProjectorTimings getExperimentalMetalProjectorTimings()
	    const;
	std::vector<ExperimentalMetalProjectorSubsetTiming>
	    getExperimentalMetalProjectorSubsetTimings() const;
	void setExperimentalMetalProjectorCacheEnabled(bool enabled);
	bool isExperimentalMetalProjectorCacheEnabled() const;
	void setExperimentalMetalProjectorCacheMaxBytes(size_t maxBytes);
	size_t getExperimentalMetalProjectorCacheMaxBytes() const;
	void setExperimentalMetalProjectorCorrectionCacheReserveBytes(
	    size_t reserveBytes);
	size_t getExperimentalMetalProjectorCorrectionCacheReserveBytes() const;
	void setExperimentalMetalProjectorMaxBatchEvents(size_t maxBatchEvents);
	size_t getExperimentalMetalProjectorMaxBatchEvents() const;
	void setExperimentalMetalProjectorMaxChunkEvents(size_t maxChunkEvents);
	size_t getExperimentalMetalProjectorMaxChunkEvents() const;
	void setExperimentalMetalProjectorLazyCorrectionsEnabled(bool enabled);
	bool isExperimentalMetalProjectorLazyCorrectionsEnabled() const;
	void setExperimentalMetalProjectorCachedCorrectionsEnabled(bool enabled);
	bool isExperimentalMetalProjectorCachedCorrectionsEnabled() const;
	void setExperimentalMetalProjectorImagePsfEnabled(bool enabled);
	bool isExperimentalMetalProjectorImagePsfEnabled() const;
	void setExperimentalMetalProjectorOptions(
	    const ExperimentalMetalProjectorOptions& options);
	ExperimentalMetalProjectorOptions getExperimentalMetalProjectorOptions()
	    const;

protected:
	bool isReconstructionTimingEnabled() const override;
	void recordReconstructionTiming(ReconstructionTimingPhase phase,
	                                double seconds) override;

	// Sens Image generator driver
	void setupProjectorForSensImgGen() override;
	void prepareBuffersForSensImgGen() override;
	std::unique_ptr<Image> generateSensitivityImageForCurrentSubset() override;
	void endSensImgGen() override;

	// Reconstruction driver
	void setupForDynamicRecon() override;
	void setupProjectorForRecon() override;
	void prepareBuffersForRecon() override;
	void loadCurrentSubset(bool forRecon) override;
	void resetEMUpdateImage() override;
	void computeEMUpdateImage() override;
	void applyImageUpdate() override;
	void completeSubset() override;
	void completeMLEMIteration() override;
	void endRecon() override;

	// Abstract Getters
	ImageBase* getSensImageBuffer() override;
	ImageBase* getMLEMImageBuffer() override;
	ImageBase* getEMUpdateImageBuffer() override;
	const Corrector& getCorrector() const override;
	Corrector& getCorrector() override;
	const Corrector_CPU& getCorrector_CPU() const;

	std::unique_ptr<Projector> mp_projector;
	std::unique_ptr<BinLoader> mp_binLoader;
	// For sensitivity image generation
	std::unique_ptr<Image> mp_tempSensImageBuffer;
	// For reconstruction
	std::unique_ptr<Image> mp_mlemImageTmpEMRatio;
	std::unique_ptr<Image> mp_imageTmpPsf;
	// Corrector
	std::unique_ptr<Corrector_CPU> mp_corrector;
	bool m_experimentalMetalProjectorEnabled = false;
	bool m_experimentalMetalProjectorRanLastCompute = false;
	bool m_experimentalMetalProjectorFusedRatioEnabled = false;
	bool m_experimentalMetalProjectorResidentImagesEnabled = false;
	std::string m_experimentalMetalProjectorKernel = "siddon";
	bool m_experimentalMetalProjectorProfilingEnabled = false;
	bool m_experimentalMetalProjectorAdjointDiagnosticsEnabled = false;
	bool m_experimentalMetalProjectorAdjointHitDiagnosticsEnabled = false;
	bool m_experimentalMetalProjectorCacheEnabled = true;
	bool m_experimentalMetalProjectorLazyCorrectionsEnabled = false;
	bool m_experimentalMetalProjectorCachedCorrectionsEnabled = false;
	bool m_experimentalMetalProjectorImagePsfEnabled = false;
	size_t m_experimentalMetalProjectorCacheMaxBytes =
	    static_cast<size_t>(1024) * 1024 * 1024;
	size_t m_experimentalMetalProjectorCorrectionCacheReserveBytes = 0;
	size_t m_experimentalMetalProjectorMaxBatchEvents = 1000000;
	bool m_experimentalMetalProjectorDirectFrameBatchesExplicit = false;
	bool m_experimentalMetalProjectorDirectFrameBatches = false;
	bool m_experimentalMetalProjectorNativeFloatAtomicsExplicit = false;
	bool m_experimentalMetalProjectorNativeFloatAtomics = false;
	bool m_experimentalMetalProjectorJosephAdjointAxisSwitchOnceExplicit =
	    false;
	bool m_experimentalMetalProjectorJosephAdjointAxisSwitchOnce = false;
	bool m_experimentalMetalProjectorThreadsPerThreadgroupExplicit = false;
	size_t m_experimentalMetalProjectorThreadsPerThreadgroup = 0;
	ExperimentalMetalProjectorTimings m_experimentalMetalProjectorTimings;
	std::vector<ExperimentalMetalProjectorSubsetTiming>
	    m_experimentalMetalProjectorSubsetTimings;
#if BUILD_METAL
	struct ExperimentalMetalResidentOsemState
	{
		backend::metal::Context context;
		backend::metal::Buffer imageBuffer;
		backend::metal::Buffer updateBuffer;
		backend::metal::Buffer sensitivityBuffer;
		backend::metal::Buffer psfForwardBuffer;
		backend::metal::Buffer psfUpdateBuffer;
		ImageParams imageParams;
		ImageParams updateParams;
		ImageParams sensitivityParams;
		const ImageBase* sensitivityImage = nullptr;
		bool imageUploaded = false;
		bool updateAllocated = false;
		bool updateReady = false;
		bool sensitivityUploaded = false;
		bool hostImageStale = false;
	};

	std::unique_ptr<backend::metal::OperatorProjectorMetalCache>
	    mp_experimentalMetalProjectorCache;
	std::unique_ptr<ExperimentalMetalResidentOsemState>
	    mp_experimentalMetalResidentOsemState;
	std::unique_ptr<backend::metal::OperatorPsfMetal>
	    mp_experimentalMetalImagePsf;
	const backend::metal::Context* mp_experimentalMetalImagePsfContext =
	    nullptr;
#endif

private:
	void initBinLoader(bool forRecon);
	std::set<ProjectionPropertyType> getNeededProperties(bool forRecon) const;
	bool computeEMUpdateImageWithExperimentalMetalProjector(
	    const Image& inputImageForForwardProj, Image& destImageForBackproj,
	    const ProjectionData& measurements, const BinIterator& binIter,
	    const Corrector_CPU& corrector, float globalScaleFactor,
	    bool hasSensitivity, bool hasAttenuation, bool hasScatterEstimates,
	    bool hasRandomsEstimates, bool hasInVivoAttenuation);
#if BUILD_METAL
	bool canUseExperimentalMetalImagePsf() const;
	backend::metal::OperatorPsfMetal* getExperimentalMetalImagePsf(
	    const backend::metal::Context* context = nullptr);
	std::string describeExperimentalMetalImagePsfState(
	    const Image& input, const Image& output);
	bool applyExperimentalMetalImagePsfForward(const Image& input,
	                                           Image& output);
	bool applyExperimentalMetalImagePsfAdjoint(const Image& input,
	                                           Image& output);
	bool applyExperimentalMetalResidentImagePsfForward(
	    const backend::metal::Context& context,
	    const backend::metal::Buffer& input,
	    backend::metal::Buffer& output,
	    const backend::metal::ImageShape& shape);
	bool applyExperimentalMetalResidentImagePsfAdjoint(
	    const backend::metal::Context& context,
	    const backend::metal::Buffer& input,
	    backend::metal::Buffer& output,
	    const backend::metal::ImageShape& shape);
	bool isExperimentalMetalResidentImagesAllowedForCurrentState() const;
	bool tryResetExperimentalMetalResidentUpdateImage();
	bool downloadExperimentalMetalResidentImage();
	bool ensureExperimentalMetalResidentProjectorBuffers(
	    const Image& inputImage, const Image& updateImage,
	    backend::metal::OperatorProjectorMetalProfile& bridgeProfile);
	bool applyExperimentalMetalResidentImageUpdate(
	    const ImageBase* sensitivityImage);
#endif
};
}  // namespace yrt
