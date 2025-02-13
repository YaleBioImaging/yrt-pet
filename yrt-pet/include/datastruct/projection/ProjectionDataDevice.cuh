/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "datastruct/projection/LORsDevice.cuh"
#include "datastruct/projection/ProjectionData.hpp"
#include "datastruct/projection/ProjectionList.hpp"
#include "datastruct/scanner/ScannerDevice.cuh"
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
	// The Scanner LUT is already in the device, but still need to generate the
	// BinIterators
	ProjectionDataDevice(std::shared_ptr<ScannerDevice> pp_scannerDevice,
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
	void loadEventLORs(size_t subsetId, size_t batchId,
	                   const cudaStream_t* stream = nullptr);
	void loadProjValuesFromReference(const cudaStream_t* stream = nullptr);
	void loadProjValuesFromHost(const ProjectionData* src,
	                            const cudaStream_t* stream);
	void loadProjValuesFromHostHistogram(const Histogram* histo,
	                                     const cudaStream_t* stream);
	void transferProjValuesToHost(ProjectionData* projDataDest,
	                              const cudaStream_t* stream = nullptr) const;

	std::shared_ptr<ScannerDevice> getScannerDevice() const;

	// Gets the size of the last-loaded batch
	size_t getCurrentBatchSize() const;
	// Gets the index of the last-loaded batch
	size_t getCurrentBatchId() const;
	// Get the index of the last-loaded subset
	size_t getCurrentSubsetId() const;

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
	void clearProjectionsDevice(float value, const cudaStream_t* stream);
	void clearProjectionsDevice(const cudaStream_t* stream = nullptr);
	void divideMeasurements(const ProjectionData* measurements,
	                        const BinIterator* binIter) override;
	void divideMeasurementsDevice(const ProjectionData* measurements,
	                              const cudaStream_t* stream);
	void invertProjValuesDevice(const cudaStream_t* stream);
	void addProjValues(const ProjectionDataDevice* projValues,
	                   const cudaStream_t* stream);
	void convertToACFsDevice(const cudaStream_t* stream);
	void multiplyProjValues(const ProjectionDataDevice* projValues,
	                        const cudaStream_t* stream);
	void multiplyProjValues(float scalar, const cudaStream_t* stream);
	const GPUBatchSetup& getBatchSetup(size_t subsetId) const;
	size_t getNumBatches(size_t subsetId) const;
	bool areLORsGathered() const;

	// Use 90% of what is available
	static constexpr float DefaultMemoryShare = 0.9f;

protected:
	virtual void loadProjValuesFromHostInternal(const ProjectionData* src,
	                                            const Histogram* histo,
	                                            const cudaStream_t* stream);

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
	ProjectionDataDeviceOwned(std::shared_ptr<ScannerDevice> pp_scannerDevice,
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
	ProjectionDataDeviceOwned(const ProjectionDataDevice* orig);

	~ProjectionDataDeviceOwned() override = default;

	bool allocateForProjValues(const cudaStream_t* stream = nullptr);

	float* getProjValuesDevicePointer() override;
	const float* getProjValuesDevicePointer() const override;

protected:
	void loadProjValuesFromHostInternal(const ProjectionData* src,
	                                    const Histogram* histo,
	                                    const cudaStream_t* stream) override;

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
	ProjectionDataDeviceAlias(std::shared_ptr<ScannerDevice> pp_scannerDevice,
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
	ProjectionDataDeviceAlias(const ProjectionDataDevice* orig);

	float* getProjValuesDevicePointer() override;
	const float* getProjValuesDevicePointer() const override;
	size_t getProjValuesDevicePointerInULL() const;

	void setProjValuesDevicePointer(float* ppd_devicePointer);
	void setProjValuesDevicePointer(size_t ppd_pointerInULL);
	bool isDevicePointerSet() const;

private:
	float* mpd_devicePointer;
};
