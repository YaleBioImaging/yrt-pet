/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionListDevice.cuh"
#include "yrt-pet/operators/ProjectionPsfManagerDevice.cuh"
#include "yrt-pet/operators/Projector.hpp"
#include "yrt-pet/operators/ProjectorUpdaterDevice.cuh"
#include "yrt-pet/recon/Corrector_GPU.cuh"
#include "yrt-pet/recon/OSEM.hpp"
#include "yrt-pet/utils/GPUStream.cuh"


namespace yrt
{

struct CUScannerParams;
struct CUImage;

class OSEM_GPU : public OSEM
{
public:
	explicit OSEM_GPU(const Scanner& pr_scanner);
	~OSEM_GPU() override = default;

	void addImagePSF(const std::string& p_imagePsf_fname,
	                 ImagePSFMode p_imagePSFMode) override;

protected:
	// Sens Image generator driver
	void setupProjectorForSensImgGen() override;
	void allocateForSensImgGen() override;
	std::unique_ptr<Image> generateSensitivityImageForCurrentSubset() override;
	void endSensImgGen() override;

	// Reconstruction driver
	void setupForDynamicRecon() override;
	void setupProjectorForRecon() override;
	void allocateForRecon() override;
	void loadCurrentSubset(bool forRecon) override;
	void resetEMUpdateImage() override;
	void computeEMUpdateImage() override;
	void applyImageUpdate() override;
	void completeSubset() override;
	void completeMLEMIteration() override;
	void endRecon() override;

	// Getters for internal objects
	const cudaStream_t* getAuxStream() const;
	const cudaStream_t* getMainStream() const;

	// Overridden abstract getters
	ImageBase* getSensImageBuffer() override;
	ImageBase* getMLEMImageBuffer() override;
	ImageBase* getEMUpdateImageBuffer() override;
	const Corrector& getCorrector() const override;
	Corrector& getCorrector() override;

	// Abstract member functions
	virtual void setupProjectorUpdater(const ProjectorParams& params);

	// Use 90% of what is available
	static constexpr float DefaultMemoryShare = 0.9f;

	// Helpers
	size_t getMemAvailable(float shareOfMemoryToUse) const;
	size_t getBatchSize(int subsetId, int batchId) const;
	size_t getMaxBatchSize() const;
	int getNumBatchesInSubset(int subsetId) const;

	// Precompute values for batch loading for sensitivity image generation
	void precomputeBatchPropsForSensImgGen(int subsetId, int batchId);
	// Precompute values for batch loading for reconstruction
	void precomputeBatchPropsForRecon(int subsetId, int batchId);
	// Transfer projection properties to the device strcture
	void loadPrecomputedBatchPropsToDevice(int subsetId, int batchId,
	                                       GPULaunchConfig launchConfig);
	// Add to sensitivity image for the currently-loaded batch
	void generateSensImageForLoadedBatch(int subsetId, int batchId,
	                                     GPULaunchConfig launchConfig);
	// Add to EM update image for the currently-loaded batch
	void computeEMUpdateImageForLoadedBatch(int subsetId, int batchId,
	                                        GPULaunchConfig launchConfig);

	std::unique_ptr<ImageDeviceOwned> mpd_sensImageBuffer;
	std::unique_ptr<ImageDeviceOwned> mpd_mlemImage;
	// Temporary buffers
	std::unique_ptr<ImageDeviceOwned> mpd_tmpImage1;
	std::unique_ptr<ImageDeviceOwned> mpd_tmpImage2;

	// Buffers used for sensitivity image generation and reconstruction
	std::unique_ptr<BinLoader> mp_binLoader;
	std::unique_ptr<PropStructDevice<ProjectionPropertyType>> mpd_propStruct;
	std::vector<GPUBatchSetup> m_batchSetups;  // One batch setup per subset
	// Corrector
	std::unique_ptr<Corrector_GPU> mp_corrector;
	// Objects used for projections
	std::unique_ptr<ProjectionPsfManagerDevice> mp_projPsfManager;
	ProjectorUpdaterDeviceWrapper m_updaterContainer;
	std::unique_ptr<DeviceSynchronizedObject<TimeOfFlightHelper>> mp_tofHelper;

	// This projector is only used as a shortcut to gather the correct
	//  projection properties, not to actually project
	std::unique_ptr<Projector> mp_projector;

private:
	std::set<ProjectionPropertyType> getNeededProperties(bool forRecon) const;

	GPUStream m_mainStream;
	GPUStream m_auxStream;
};

template <bool UseUpdater>
__global__ void computeEMUpdateImage_kernel(
    CUImage forwImage, CUImage emImage, float globalScaleFactor,
    float measurementUniformValue,
    const ProjectionPropertyManager* pd_projPropManager,
    const PropertyUnit* pd_projectionProperties, UpdaterPointer pd_updater,
    const TimeOfFlightHelper* pd_tofHelper,
    ProjectionPsfKernelStruct projPsfKernelStruct,
    CUScannerParams scannerParams, int numRays, ProjectorType projectorType,
    size_t batchSize);

__global__ void generateSensImage_kernel(
    CUImage sensImage, CUImage attImage,
    const ProjectionPropertyManager* pd_projPropManager,
    const PropertyUnit* pd_projectionProperties,
    ProjectionPsfKernelStruct projPsfKernelStruct,
    CUScannerParams scannerParams, int numRays, ProjectorType projectorType,
    size_t batchSize);

}  // namespace yrt
