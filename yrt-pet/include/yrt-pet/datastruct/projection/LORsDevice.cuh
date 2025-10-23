/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

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

	const float4* getLorDet1PosDevicePointer() const;
	const float4* getLorDet1OrientDevicePointer() const;
	const float4* getLorDet2PosDevicePointer() const;
	const float4* getLorDet2OrientDevicePointer() const;
	const float* getLorTOFValueDevicePointer() const;
	float4* getLorDet1PosDevicePointer();
	float4* getLorDet1OrientDevicePointer();
	float4* getLorDet2PosDevicePointer();
	float4* getLorDet2OrientDevicePointer();
	const frame_t* getDynamicFrameDevicePointer() const;
	float* getLorTOFValueDevicePointer();
	bool areLORsGathered() const;

	static constexpr size_t MemoryUsagePerLOR = sizeof(float4) * 4;

	static constexpr size_t MemoryUsagePerLORWithTOF =
	    MemoryUsagePerLOR + sizeof(float);

private:
	void initializeDeviceArrays();
	void allocateForPrecomputedLORsIfNeeded(GPULaunchConfig launchConfig);

	std::unique_ptr<DeviceArray<float4>> mp_lorDet1Pos;
	std::unique_ptr<DeviceArray<float4>> mp_lorDet2Pos;
	std::unique_ptr<DeviceArray<float4>> mp_lorDet1Orient;
	std::unique_ptr<DeviceArray<float4>> mp_lorDet2Orient;
	std::unique_ptr<DeviceArray<frame_t>> mp_dynamicFrame;
	std::unique_ptr<DeviceArray<float>> mp_lorTOFValue;
	PageLockedBuffer<float4> m_tempLorDet1Pos;
	PageLockedBuffer<float4> m_tempLorDet2Pos;
	PageLockedBuffer<float4> m_tempLorDet1Orient;
	PageLockedBuffer<float4> m_tempLorDet2Orient;
	PageLockedBuffer<frame_t> m_tempDynamicFrame;
	PageLockedBuffer<float> m_tempLorTOFValue;
	bool m_hasDynamicFraming;
	bool m_hasTOF;
	size_t m_precomputedBatchSize;
	int m_precomputedBatchId;
	int m_precomputedSubsetId;
	bool m_areLORsPrecomputed;
	size_t m_loadedBatchSize;
	int m_loadedBatchId;
	int m_loadedSubsetId;
};
}
