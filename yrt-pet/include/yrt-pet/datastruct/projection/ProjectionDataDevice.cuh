/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "BinFilter.hpp"
#include "yrt-pet/datastruct/image/ImageBase.hpp"
#include "yrt-pet/datastruct/projection/LORsDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionList.hpp"
#include "yrt-pet/utils/DeviceArray.cuh"
#include "yrt-pet/utils/GPUTypes.cuh"
#include "yrt-pet/utils/PageLockedBuffer.cuh"

#include <memory>

namespace yrt
{
class Histogram;

class ProjectionDataDevice : public ProjectionList
{
public:
	// The Scanner LUT has to be loaded to device, but the BinIterators have
	// already been generated
	ProjectionDataDevice(const Scanner& pr_scanner,
	                     const ProjectionData* pp_reference,
	                     std::vector<const BinIterator*> pp_binIteratorList,
	                     size_t memoryUsagePerLOR, size_t memAvailable);
	// The Scanner LUT has to be loaded to device AND the BinIterators have to
	// be generated
	ProjectionDataDevice(const Scanner& pr_scanner,
	                     const ProjectionData* pp_reference,
	                     size_t memoryUsagePerLOR, size_t memAvailable,
	                     int num_OSEM_subsets = 1);
	// The Scanner LUT AND the lines of responses are already on device, but the
	// BinIterators have already been generated
	ProjectionDataDevice(std::shared_ptr<LORsDevice> pp_LORs,
	                     const ProjectionData* pp_reference,
	                     size_t memoryUsagePerLOR, size_t memAvailable,
	                     int num_OSEM_subsets = 1);
	// The Scanner LUT AND the lines of responses are already on the device, and
	// no need to generate the BinIterators
	ProjectionDataDevice(std::shared_ptr<LORsDevice> pp_LORs,
	                     const ProjectionData* pp_reference,
	                     std::vector<const BinIterator*> pp_binIteratorList,
	                     size_t memoryUsagePerLOR, size_t memAvailable);
	// Proxy for the above
	explicit ProjectionDataDevice(const ProjectionDataDevice* orig);

	// Load the events' detector ids from a specific subset&batch id and prepare
	// the projection values buffer
	void prepareBatchLORs(int subsetId, int batchId,
	                      GPULaunchConfig launchConfig,
	                      const BinFilter& binFilter);
	void precomputeBatchLORs(int subsetId, int batchId,
	                         const BinFilter& binFilter);
	void loadPrecomputedLORsToDevice(GPULaunchConfig launchConfig);

	// Gather the projection values from the reference ProjectionData object and
	// store them on the GPU buffer
	void loadProjValuesFromReference(GPULaunchConfig launchConfig);
	// Gather the projection values from any given ProjectionData object and
	// store them on the GPU buffer
	void loadProjValuesFromHost(const ProjectionData* src,
	                            GPULaunchConfig launchConfig);
	// Gather the randoms estimates from any given ProjectionData object and
	// store them on the GPU buffer
	void loadProjValuesFromHostRandoms(const ProjectionData* src,
	                                   GPULaunchConfig launchConfig);
	// Gather the projection values from any given Histogram object (using the
	// associated histogram bins) and store them on the GPU buffer
	void loadProjValuesFromHostHistogram(const Histogram* histo,
	                                     GPULaunchConfig launchConfig);
	// Transfer the projection values on the device into a host-side
	// ProjectionData object
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
	const char* getProjectionPropertiesDevicePointer() const;

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

protected:
	// Function overridden by the Owned vs Alias pattern
	virtual void loadProjValuesFromHostInternal(const ProjectionData* src,
	                                            const Histogram* histo,
	                                            bool gatherRandoms,
	                                            GPULaunchConfig launchConfig);

	// For Host->Device data transfers
	mutable PageLockedBuffer<float> m_tempBuffer;

	// We need all the BinIterators in order to be able to properly load the
	// data from Host to device (and vice-verse)
	std::vector<const BinIterator*> mp_binIteratorList;

private:
	// Helper with all the logic
	template <bool GatherRandoms = false>
	void loadProjValuesFromHostInternal(const ProjectionData* src,
	                                    const Histogram* histo,
	                                    GPULaunchConfig launchConfig);
	void createBinIterators(int num_OSEM_subsets);
	void createBatchSetups(size_t memoryUsagePerLOR, size_t memAvailable);

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
	    size_t memoryUsagePerLOR, size_t memAvailable);
	ProjectionDataDeviceOwned(const Scanner& pr_scanner,
	                          const ProjectionData* pp_reference,
	                          size_t memoryUsagePerLOR, size_t memAvailable,
	                          int num_OSEM_subsets = 1);
	ProjectionDataDeviceOwned(std::shared_ptr<LORsDevice> pp_LORs,
	                          const ProjectionData* pp_reference,
	                          size_t memoryUsagePerLOR, size_t memAvailable,
	                          int num_OSEM_subsets = 1);
	ProjectionDataDeviceOwned(
	    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    size_t memoryUsagePerLOR, size_t memAvailable);
	explicit ProjectionDataDeviceOwned(const ProjectionDataDevice* orig);

	~ProjectionDataDeviceOwned() override = default;

	bool allocateForProjValues(GPULaunchConfig launchConfig);

	float* getProjValuesDevicePointer() override;
	const float* getProjValuesDevicePointer() const override;

protected:
	void loadProjValuesFromHostInternal(const ProjectionData* src,
	                                    const Histogram* histo,
	                                    bool gatherRandoms,
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
	    size_t memoryUsagePerLOR, size_t memAvailable);
	ProjectionDataDeviceAlias(const Scanner& pr_scanner,
	                          const ProjectionData* pp_reference,
	                          size_t memoryUsagePerLOR, size_t memAvailable,
	                          int num_OSEM_subsets = 1);
	ProjectionDataDeviceAlias(std::shared_ptr<LORsDevice> pp_LORs,
	                          const ProjectionData* pp_reference,
	                          size_t memoryUsagePerLOR, size_t memAvailable,
	                          int num_OSEM_subsets = 1);
	ProjectionDataDeviceAlias(
	    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData*
	    pp_reference, std::vector<const BinIterator*> pp_binIteratorList,
	    size_t memoryUsagePerLOR, size_t memAvailable);
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
}  // namespace yrt
