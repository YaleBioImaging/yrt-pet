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

class ProjectionListDevice : public ProjectionList
{
public:
	// This constructor will initialize its own LORsDevice, use the given list
	// of bin iterators, and compute its own batch setups
	ProjectionListDevice(const Scanner& pr_scanner,
	                     const ProjectionData* pp_reference,
	                     std::vector<const BinIterator*> pp_binIteratorList,
	                     const std::vector<Constraint*>& constraints,
	                     const std::set<ProjectionPropertyType>& properties,
	                     size_t memAvailable);
	// This constructor will initialize its own LORsDevice, use the reference
	//  to create a bin iterator for every subset, and compute its own batch
	//  setups
	ProjectionListDevice(const Scanner& pr_scanner,
	                     const ProjectionData* pp_reference,
	                     const std::vector<Constraint*>& constraints,
	                     const std::set<ProjectionPropertyType>& properties,
	                     size_t memAvailable, int num_OSEM_subsets = 1);
	// This constructor will share the given LORsDevice (instead of initializing
	//  its own), use the reference to create a bin iterator for every subset,
	//  and compute its own batch setups
	ProjectionListDevice(std::shared_ptr<LORsDevice> pp_LORs,
	                     const ProjectionData* pp_reference,
	                     size_t memAvailable, int num_OSEM_subsets = 1);
	// This constructor will share the given LORsDevice (instead of initializing
	//  its own), use the given list of bin iterators, and compute its own batch
	//  setups
	ProjectionListDevice(std::shared_ptr<LORsDevice> pp_LORs,
	                     const ProjectionData* pp_reference,
	                     std::vector<const BinIterator*> pp_binIteratorList,
	                     size_t memAvailable);
	// This constructor will use the given ProjectionListDevice to share the
	//  same LORsDevice, same bin iterators, and same batch setups. The only
	//  difference is the projection values, which would not be shared
	explicit ProjectionListDevice(const ProjectionListDevice* orig);

	// Load the events' detector ids from a specific subset & batch id and
	//  prepare the projection values buffer
	void prepareBatchLORs(int subsetId, int batchId,
	                      GPULaunchConfig launchConfig);
	void precomputeBatchLORs(int subsetId, int batchId);
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
	// Bytes used per event
	size_t getMemoryUsagePerEvent() const;

	virtual float* getProjValuesDevicePointer() = 0;
	virtual const float* getProjValuesDevicePointer() const = 0;
	const ProjectionPropertyManager*
	    getProjectionPropertyManagerDevicePointer() const;
	const PropertyUnit* getProjectionPropertiesDevicePointer() const;

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
	void addProjValues(const ProjectionListDevice* projValues,
	                   GPULaunchConfig launchConfig);
	void convertToACFsDevice(GPULaunchConfig launchConfig);
	void multiplyProjValues(const ProjectionListDevice* projValues,
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
	// For the constructors
	void initLORsDevice(const std::vector<Constraint*>& constraints,
	                    std::set<ProjectionPropertyType> properties);
	void createBinIterators(int num_OSEM_subsets);
	void createBatchSetups(size_t memAvailable);

	// Loading helper with all the logic
	template <bool GatherRandoms = false>
	void loadProjValuesFromHostInternal(const ProjectionData* src,
	                                    const Histogram* histo,
	                                    GPULaunchConfig launchConfig);

	std::shared_ptr<LORsDevice> mp_LORs;
	const Scanner& mr_scanner;
	std::vector<GPUBatchSetup> m_batchSetups;  // One batch setup per subset

	// In case we need to compute our own BinIterators
	std::vector<std::unique_ptr<BinIterator>> m_binIterators;
};

class ProjectionListDeviceOwned : public ProjectionListDevice
{
public:
	ProjectionListDeviceOwned(
	    const Scanner& pr_scanner, const ProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    const std::vector<Constraint*>& constraints,
	    const std::set<ProjectionPropertyType>& properties,
	    size_t memAvailable);
	ProjectionListDeviceOwned(
	    const Scanner& pr_scanner, const ProjectionData* pp_reference,
	    const std::vector<Constraint*>& constraints,
	    const std::set<ProjectionPropertyType>& properties, size_t memAvailable,
	    int num_OSEM_subsets = 1);
	ProjectionListDeviceOwned(std::shared_ptr<LORsDevice> pp_LORs,
	                          const ProjectionData* pp_reference,
	                          size_t memAvailable, int num_OSEM_subsets = 1);
	ProjectionListDeviceOwned(
	    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    size_t memAvailable);
	explicit ProjectionListDeviceOwned(const ProjectionListDevice* orig);

	~ProjectionListDeviceOwned() override = default;

	bool allocateForProjValuesIfNeeded(GPULaunchConfig launchConfig);

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

class ProjectionListDeviceAlias : public ProjectionListDevice
{
public:
	ProjectionListDeviceAlias(
	    const Scanner& pr_scanner, const ProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    const std::vector<Constraint*>& constraints,
	    const std::set<ProjectionPropertyType>& properties,
	    size_t memAvailable);
	ProjectionListDeviceAlias(
	    const Scanner& pr_scanner, const ProjectionData* pp_reference,
	    const std::vector<Constraint*>& constraints,
	    const std::set<ProjectionPropertyType>& properties, size_t memAvailable,
	    int num_OSEM_subsets = 1);
	ProjectionListDeviceAlias(std::shared_ptr<LORsDevice> pp_LORs,
	                          const ProjectionData* pp_reference,
	                          size_t memAvailable, int num_OSEM_subsets = 1);
	ProjectionListDeviceAlias(
	    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    size_t memAvailable);
	explicit ProjectionListDeviceAlias(const ProjectionListDevice* orig);

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
