/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/projection/BinLoader.hpp"
#include "yrt-pet/datastruct/scanner/ScannerDevice.cuh"
#include "yrt-pet/utils/GPUTypes.cuh"
#include "yrt-pet/utils/PageLockedBuffer.cuh"

#include <memory>

namespace yrt
{
class ProjectionData;
class Scanner;
class ImageParams;

// This object is meant to be used in conjunction with ProjectionListDevice
//  to hold projection properties (LOR, TOF, and everything else except for the
//  projection value). Internally, this object uses a BinLoader to hold
//  these properties in the host and a PropStructDevice to hold them in the
//  device.
class LORsDevice
{
public:
	LORsDevice(const std::vector<Constraint*>& constraints,
	           const std::set<ProjectionPropertyType>& projProperties);

	void precomputeBatchLORs(const BinIterator& binIter,
	                         const GPUBatchSetup& batchSetup, int subsetId,
	                         int batchId, const ProjectionData& reference);
	void loadPrecomputedLORsToDevice(GPULaunchConfig launchConfig);

	// Gets the size of the last precomputed batch
	size_t getPrecomputedBatchSize() const;
	// Gets the index of the last precomputed batch
	int getPrecomputedBatchId() const;
	// Get the index of the last precomputed subset
	int getPrecomputedSubsetId() const;
	// Gets the size of the last-loaded batch
	size_t getLoadedBatchSize() const;
	// Gets the index of the last-loaded batch
	int getLoadedBatchId() const;
	// Gets the index of the last-loaded subset
	int getLoadedSubsetId() const;

	const ProjectionPropertyManager*
	    getProjectionPropertyManagerDevicePointer() const;
	const PropertyUnit* getProjectionPropertiesDevicePointer() const;
	PropertyUnit* getProjectionPropertiesDevicePointer();
	bool areLORsGathered() const;

	// Memory (in bytes) that would be used by one row in the bin loader
	size_t getElementSize() const;

private:
	void initializeDeviceArray(
	    const std::set<ProjectionPropertyType>& projProperties);
	void initBinLoader(
	    const std::vector<Constraint*>& constraints = {},
	    const std::set<ProjectionPropertyType>& projProperties = {});
	void allocateForPrecomputedLORsIfNeeded(GPULaunchConfig launchConfig);
	void allocateBinFilterIfNeeded(size_t newBatchSize);

	// Contains temporary projection-space data in the host waiting to be loaded
	//  on the GPU
	std::unique_ptr<BinLoader> mp_binLoader;
	// Projection-space data stored on the device
	std::unique_ptr<PropStructDevice<ProjectionPropertyType>>
	    mp_projectionProperties;

	size_t m_precomputedBatchSize;
	int m_precomputedBatchId;
	int m_precomputedSubsetId;
	bool m_areLORsPrecomputed;
	size_t m_loadedBatchSize;
	int m_loadedBatchId;
	int m_loadedSubsetId;
};
}  // namespace yrt
