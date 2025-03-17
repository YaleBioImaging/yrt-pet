/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "datastruct/projection/LORsDevice.cuh"
#include "datastruct/projection/ProjectionData.hpp"
#include "datastruct/projection/ProjectionList.hpp"
#include "utils/DeviceArray.cuh"
#include "utils/GPUTypes.cuh"
#include "utils/PageLockedBuffer.cuh"

#include <memory>

class Histogram;

class ProjectionDataDevice : public ProjectionList
{
public:
	// The Scanner LUT has to be loaded to device, but the BinIterators have
	// already been generated
	ProjectionDataDevice(const Scanner& pr_scanner,
	                     const ProjectionData* pp_reference,
	                     std::vector<const BinIterator*> pp_binIteratorList,
	                     float shareOfMemoryToUse = DefaultMemoryShare);
	// The Scanner LUT has to be loaded to device AND the BinIterators have to
	// be generated
	ProjectionDataDevice(const Scanner& pr_scanner,
	                     const ProjectionData* pp_reference,
	                     int num_OSEM_subsets = 1,
	                     float shareOfMemoryToUse = DefaultMemoryShare);
	// The Scanner LUT AND the lines of responses are already on device, but the
	// BinIterators have already been generated
	ProjectionDataDevice(std::shared_ptr<LORsDevice> pp_LORs,
	                     const ProjectionData* pp_reference,
	                     int num_OSEM_subsets = 1,
	                     float shareOfMemoryToUse = DefaultMemoryShare);
	// The Scanner LUT AND the lines of responses are already on the device, and
	// no need to generate the BinIterators
	ProjectionDataDevice(std::shared_ptr<LORsDevice> pp_LORs,
	                     const ProjectionData* pp_reference,
	                     std::vector<const BinIterator*> pp_binIteratorList,
	                     float shareOfMemoryToUse = DefaultMemoryShare);
	// Proxy for the above
	explicit ProjectionDataDevice(const ProjectionDataDevice* orig);

	// Load the events' detector ids from a specific subset&batch id and prepare
	// the projection values buffer
	void prepareBatchLORs(int subsetId, int batchId,
	                      GPULaunchConfig launchConfig);
	void precomputeBatchLORs(int subsetId, int batchId);
	void loadPrecomputedLORsToDevice(GPULaunchConfig launchConfig);

	void loadProjValuesFromReference(GPULaunchConfig launchConfig);
	void loadProjValuesFromHost(const ProjectionData* src,
	                            GPULaunchConfig launchConfig);
	void loadProjValuesFromHostHistogram(const Histogram* histo,
	                                     GPULaunchConfig launchConfig);
	void transferProjValuesToHost(ProjectionData* projDataDest,
	                              const cudaStream_t* stream = nullptr) const;

	// Gets the size of the last precomputed batch
	size_t getPrecomputedBatchSize() const;
	// Gets the index of the last precomputed batch
	size_t getPrecomputedBatchId() const;
	// Get the index of the last precomputed subset
	size_t getPrecomputedSubsetId() const;
	// Gets the size of the last-loaded batch
	size_t getLoadedBatchSize() const;
	// Gets the index of the last-loaded batch
	size_t getLoadedBatchId() const;
	// Get the index of the last-loaded subset
	size_t getLoadedSubsetId() const;

	virtual float* getProjValuesDevicePointer() = 0;
	virtual const float* getProjValuesDevicePointer() const = 0;
	const float4* getLorDet1PosDevicePointer() const;
	const float4* getLorDet1OrientDevicePointer() const;
	const float4* getLorDet2PosDevicePointer() const;
	const float4* getLorDet2OrientDevicePointer() const;
	const float* getLorTOFValueDevicePointer() const;

	float getProjectionValue(bin_t id) const override;
	void setProjectionValue(bin_t id, float val) override;
	void clearProjections(float value) override;
	void clearProjectionsDevice(float value, GPULaunchConfig launchConfig);
	void clearProjectionsDevice(GPULaunchConfig launchConfig);
	void divideMeasurements(const ProjectionData* measurements,
	                        const BinIterator* binIter) override;
	void divideMeasurementsDevice(const ProjectionData* measurements,
	                              GPULaunchConfig launchConfig);
	void invertProjValuesDevice(GPULaunchConfig launchConfig);
	void addProjValues(const ProjectionDataDevice* projValues,
	                   GPULaunchConfig launchConfig);
	void convertToACFsDevice(GPULaunchConfig launchConfig);
	void multiplyProjValues(const ProjectionDataDevice* projValues,
	                        GPULaunchConfig launchConfig);
	void multiplyProjValues(float scalar, GPULaunchConfig launchConfig);
	const GPUBatchSetup& getBatchSetup(size_t subsetId) const;
	size_t getNumBatches(size_t subsetId) const;
	bool areLORsGathered() const;

	// Use 90% of what is available
	static constexpr float DefaultMemoryShare = 0.9f;

protected:
	virtual void loadProjValuesFromHostInternal(const ProjectionData* src,
	                                            const Histogram* histo,
	                                            GPULaunchConfig launchConfig);

	// For Host->Device data transfers
	mutable PageLockedBuffer<float> m_tempBuffer;

	// We need all the BinIterators in order to be able to properly load the
	// data from Host to device (and vice-verse)
	std::vector<const BinIterator*> mp_binIteratorList;

private:
	void createBinIterators(int num_OSEM_subsets);
	void createBatchSetups(float shareOfMemoryToUse);

	std::shared_ptr<LORsDevice> mp_LORs;
	const Scanner& mr_scanner;
	std::vector<GPUBatchSetup> m_batchSetups;  // One batch setup per subset

	// In case we need to compute our own BinIterators
	std::vector<std::unique_ptr<BinIterator>> m_binIterators;
};

class ProjectionDataDeviceOwned : public ProjectionDataDevice
{
public:
	ProjectionDataDeviceOwned(
	    const Scanner& pr_scanner, const ProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    float shareOfMemoryToUse = DefaultMemoryShare);
	ProjectionDataDeviceOwned(const Scanner& pr_scanner,
	                          const ProjectionData* pp_reference,
	                          int num_OSEM_subsets = 1,
	                          float shareOfMemoryToUse = DefaultMemoryShare);
	ProjectionDataDeviceOwned(std::shared_ptr<LORsDevice> pp_LORs,
	                          const ProjectionData* pp_reference,
	                          int num_OSEM_subsets = 1,
	                          float shareOfMemoryToUse = DefaultMemoryShare);
	ProjectionDataDeviceOwned(
	    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    float shareOfMemoryToUse = DefaultMemoryShare);
	explicit ProjectionDataDeviceOwned(const ProjectionDataDevice* orig);

	~ProjectionDataDeviceOwned() override = default;

	bool allocateForProjValues(GPULaunchConfig launchConfig);

	float* getProjValuesDevicePointer() override;
	const float* getProjValuesDevicePointer() const override;

protected:
	void loadProjValuesFromHostInternal(const ProjectionData* src,
	                                    const Histogram* histo,
	                                    GPULaunchConfig launchConfig) override;

private:
	std::unique_ptr<DeviceArray<float>> mp_projValues;
};

class ProjectionDataDeviceAlias : public ProjectionDataDevice
{
public:
	ProjectionDataDeviceAlias(
	    const Scanner& pr_scanner, const ProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    float shareOfMemoryToUse = DefaultMemoryShare);
	ProjectionDataDeviceAlias(const Scanner& pr_scanner,
	                          const ProjectionData* pp_reference,
	                          int num_OSEM_subsets = 1,
	                          float shareOfMemoryToUse = DefaultMemoryShare);
	ProjectionDataDeviceAlias(std::shared_ptr<LORsDevice> pp_LORs,
	                          const ProjectionData* pp_reference,
	                          int num_OSEM_subsets = 1,
	                          float shareOfMemoryToUse = DefaultMemoryShare);
	ProjectionDataDeviceAlias(
	    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    float shareOfMemoryToUse = DefaultMemoryShare);
	explicit ProjectionDataDeviceAlias(const ProjectionDataDevice* orig);

	float* getProjValuesDevicePointer() override;
	const float* getProjValuesDevicePointer() const override;
	size_t getProjValuesDevicePointerInULL() const;

	void setProjValuesDevicePointer(float* ppd_devicePointer);
	void setProjValuesDevicePointer(size_t ppd_pointerInULL);
	bool isDevicePointerSet() const;

private:
	float* mpd_devicePointer;
};
