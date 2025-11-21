/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/BinFilter.hpp"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/scanner/ScannerDevice.cuh"
#include "yrt-pet/utils/GPUTypes.cuh"
#include "yrt-pet/utils/PageLockedBuffer.cuh"

#include <memory>

namespace yrt
{
class ProjectionData;
class Scanner;
class ImageParams;

class LORsDevice
{
public:
	LORsDevice();

	void precomputeBatchLORs(const BinIterator& binIter,
	                         const GPUBatchSetup& batchSetup, int subsetId,
	                         int batchId, const ProjectionData& reference,
	                         const BinFilter& binFilter);
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

	const char* getProjectionPropertiesDevicePointer() const;
	char* getProjectionPropertiesDevicePointer();
	bool areLORsGathered() const;

private:
	void initializeDeviceArrays();
	void allocateForPrecomputedLORsIfNeeded(GPULaunchConfig launchConfig);

	std::unique_ptr<DeviceArray<char>> mp_projectionProperties;
	PageLockedBuffer<char> m_tempProjectionProperties;
	// Size (in bytes) of ProjectionProperties element of precomputed batch
	size_t m_elementSize;
	bool m_hasTOF;
	size_t m_precomputedBatchSize;
	int m_precomputedBatchId;
	int m_precomputedSubsetId;
	bool m_areLORsPrecomputed;
	size_t m_loadedBatchSize;
	int m_loadedBatchId;
	int m_loadedSubsetId;
};
}  // namespace yrt
