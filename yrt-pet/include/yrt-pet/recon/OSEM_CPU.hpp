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
class OSEM_CPU : public OSEM
{
public:
	struct ExperimentalMetalProjectorTimings
	{
		double setupSeconds = 0.0;
		double forwardSeconds = 0.0;
		double ratioSeconds = 0.0;
		double adjointSeconds = 0.0;
		double totalSeconds = 0.0;
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
		size_t adjointMaxBatchP95VoxelHits = 0;
		size_t adjointMaxBatchP99VoxelHits = 0;
		size_t cacheLookups = 0;
		size_t cacheHits = 0;
		size_t cacheMisses = 0;
		size_t cacheBuilds = 0;
		size_t cacheSkipsOverBudget = 0;
		size_t cacheUsedBytes = 0;
		size_t cacheMaxBytes = 0;
		size_t uncachedBatches = 0;
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
	void setExperimentalMetalProjectorMaxBatchEvents(size_t maxBatchEvents);
	size_t getExperimentalMetalProjectorMaxBatchEvents() const;
	void setExperimentalMetalProjectorMaxChunkEvents(size_t maxChunkEvents);
	size_t getExperimentalMetalProjectorMaxChunkEvents() const;

protected:
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
	std::string m_experimentalMetalProjectorKernel = "siddon";
	bool m_experimentalMetalProjectorProfilingEnabled = false;
	bool m_experimentalMetalProjectorAdjointDiagnosticsEnabled = false;
	bool m_experimentalMetalProjectorAdjointHitDiagnosticsEnabled = false;
	bool m_experimentalMetalProjectorCacheEnabled = true;
	size_t m_experimentalMetalProjectorCacheMaxBytes =
	    static_cast<size_t>(1024) * 1024 * 1024;
	size_t m_experimentalMetalProjectorMaxBatchEvents = 1000000;
	ExperimentalMetalProjectorTimings m_experimentalMetalProjectorTimings;
	std::vector<ExperimentalMetalProjectorSubsetTiming>
	    m_experimentalMetalProjectorSubsetTimings;
#if BUILD_METAL
	std::unique_ptr<backend::metal::OperatorProjectorMetalCache>
	    mp_experimentalMetalProjectorCache;
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
};
}  // namespace yrt
