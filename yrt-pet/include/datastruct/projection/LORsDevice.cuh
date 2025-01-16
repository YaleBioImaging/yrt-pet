/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/BinIterator.hpp"
#include "datastruct/scanner/ScannerDevice.cuh"
#include "utils/GPUTypes.cuh"
#include "utils/PageLockedBuffer.cuh"

#include <memory>

class ProjectionData;
class Scanner;
class ImageParams;

class LORsDevice
{
public:
	explicit LORsDevice(std::shared_ptr<ScannerDevice> pp_scannerDevice);
	explicit LORsDevice(const Scanner& pr_scanner);

	// Load the events' detector ids from a specific subset&batch id
	void loadEventLORs(const BinIterator& binIter,
	                   const GPUBatchSetup& batchSetup, size_t subsetId,
	                   size_t batchId, const ProjectionData& reference,
	                   const ImageParams& imgParams,
	                   const cudaStream_t* stream = nullptr);

	std::shared_ptr<ScannerDevice> getScannerDevice() const;
	const Scanner& getScanner() const;

	// Gets the size of the last-loaded batch
	size_t getLoadedBatchSize() const;
	// Gets the index of the last-loaded batch
	size_t getLoadedBatchId() const;
	// Gets the index of the last-loaded subset
	size_t getLoadedSubsetId() const;

	const float4* getLorDet1PosDevicePointer() const;
	const float4* getLorDet1OrientDevicePointer() const;
	const float4* getLorDet2PosDevicePointer() const;
	const float4* getLorDet2OrientDevicePointer() const;
	const float* getLorTOFValueDevicePointer() const;
	float4* getLorDet1PosDevicePointer();
	float4* getLorDet1OrientDevicePointer();
	float4* getLorDet2PosDevicePointer();
	float4* getLorDet2OrientDevicePointer();
	float* getLorTOFValueDevicePointer();
	bool areLORsGathered() const;

	static constexpr size_t MemoryUsagePerLOR = sizeof(float4) * 4;

	static constexpr size_t MemoryUsagePerLORWithTOF =
	    MemoryUsagePerLOR + sizeof(float);

private:
	void initializeDeviceArrays();
	void allocateForLORs(bool hasTOF, const cudaStream_t* stream = nullptr);

	// Note: This is currently unused. Could only be useful in the future if
	//  YRT-PET has several ways to load LORs and treat them in the kernels. It
	//  wouldn't allow for the possibility to keep the "getArbitraryLOR(...)"
	//  function, but would be useful for lots of other cases.
	std::shared_ptr<ScannerDevice> mp_scannerDevice;

	std::unique_ptr<DeviceArray<float4>> mp_lorDet1Pos;
	std::unique_ptr<DeviceArray<float4>> mp_lorDet2Pos;
	std::unique_ptr<DeviceArray<float4>> mp_lorDet1Orient;
	std::unique_ptr<DeviceArray<float4>> mp_lorDet2Orient;
	PageLockedBuffer<float4> m_tempLorDet1Pos;
	PageLockedBuffer<float4> m_tempLorDet2Pos;
	PageLockedBuffer<float4> m_tempLorDet1Orient;
	PageLockedBuffer<float4> m_tempLorDet2Orient;
	std::unique_ptr<DeviceArray<float>> mp_lorTOFValue;
	bool m_areLORsGathered;
	size_t m_loadedBatchSize;
	size_t m_loadedBatchId;
	size_t m_loadedSubsetId;
};
