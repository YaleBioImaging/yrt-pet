/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/LORsDevice.cuh"

#include "datastruct/projection/ProjectionData.hpp"
#include "operators/OperatorProjectorDevice.cuh"
#include "utils/PageLockedBuffer.cuh"
#include "utils/ReconstructionUtils.hpp"

#include "omp.h"


LORsDevice::LORsDevice(std::shared_ptr<ScannerDevice> pp_scannerDevice)
    : mp_scannerDevice(std::move(pp_scannerDevice)),
      m_areLORsGathered(false),
      m_loadedBatchSize(0ull),
      m_loadedBatchId(0ull),
      m_loadedSubsetId(0ull)
{
	initializeDeviceArrays();
}

LORsDevice::LORsDevice(const Scanner& pr_scanner)
    : m_areLORsGathered(false),
      m_loadedBatchSize(0ull),
      m_loadedBatchId(0ull),
      m_loadedSubsetId(0ull)
{
	mp_scannerDevice = std::make_shared<ScannerDevice>(pr_scanner);
	initializeDeviceArrays();
}

void LORsDevice::loadEventLORs(const BinIterator& binIter,
                               const GPUBatchSetup& batchSetup, size_t subsetId,
                               size_t batchId, const ProjectionData& reference,
                               const ImageParams& imgParams,
                               const cudaStream_t* stream)
{
	m_areLORsGathered = false;
	const bool hasTOF = reference.hasTOF();

	const size_t batchSize = batchSetup.getBatchSize(batchId);

	m_tempLorDet1Pos.reAllocateIfNeeded(batchSize);
	m_tempLorDet2Pos.reAllocateIfNeeded(batchSize);
	m_tempLorDet1Orient.reAllocateIfNeeded(batchSize);
	m_tempLorDet2Orient.reAllocateIfNeeded(batchSize);
	float4* tempBufferLorDet1Pos_ptr = m_tempLorDet1Pos.getPointer();
	float4* tempBufferLorDet2Pos_ptr = m_tempLorDet2Pos.getPointer();
	float4* tempBufferLorDet1Orient_ptr = m_tempLorDet1Orient.getPointer();
	float4* tempBufferLorDet2Orient_ptr = m_tempLorDet2Orient.getPointer();

	PageLockedBuffer<float> tempBufferLorTOFValue;
	if (hasTOF)
	{
		tempBufferLorTOFValue.allocate(batchSize);
	}
	float* tempBufferLorTOFValue_ptr = tempBufferLorTOFValue.getPointer();

	const size_t offset = batchId * batchSetup.getBatchSize(0);
	auto* binIter_ptr = &binIter;
	const Vector3D offsetVec{imgParams.off_x, imgParams.off_y, imgParams.off_z};
	const ProjectionData* reference_ptr = &reference;

	bin_t binId;
	size_t binIdx;
#pragma omp parallel for default(none) private(binIdx, binId)                  \
    firstprivate(offset, batchSize, binIter_ptr, offsetVec,                    \
                     tempBufferLorDet1Pos_ptr, tempBufferLorDet2Pos_ptr,       \
                     tempBufferLorDet1Orient_ptr, tempBufferLorDet2Orient_ptr, \
                     tempBufferLorTOFValue_ptr, reference_ptr, hasTOF)
	for (binIdx = 0; binIdx < batchSize; binIdx++)
	{
		binId = binIter_ptr->get(binIdx + offset);
		auto [lor, tofValue, det1Orient, det2Orient] =
		    reference_ptr->getProjectionProperties(binId);

		// TODO NOW: Remove this from here and do the subtraction in the kernel
		//  instead
		lor.point1 = lor.point1 - offsetVec;
		lor.point2 = lor.point2 - offsetVec;

		tempBufferLorDet1Pos_ptr[binIdx].x = lor.point1.x;
		tempBufferLorDet1Pos_ptr[binIdx].y = lor.point1.y;
		tempBufferLorDet1Pos_ptr[binIdx].z = lor.point1.z;
		tempBufferLorDet2Pos_ptr[binIdx].x = lor.point2.x;
		tempBufferLorDet2Pos_ptr[binIdx].y = lor.point2.y;
		tempBufferLorDet2Pos_ptr[binIdx].z = lor.point2.z;
		tempBufferLorDet1Orient_ptr[binIdx].x = det1Orient.x;
		tempBufferLorDet1Orient_ptr[binIdx].y = det1Orient.y;
		tempBufferLorDet1Orient_ptr[binIdx].z = det1Orient.z;
		tempBufferLorDet2Orient_ptr[binIdx].x = det2Orient.x;
		tempBufferLorDet2Orient_ptr[binIdx].y = det2Orient.y;
		tempBufferLorDet2Orient_ptr[binIdx].z = det2Orient.z;
		if (hasTOF)
		{
			tempBufferLorTOFValue_ptr[binIdx] = tofValue;
		}
	}

	m_loadedBatchSize = batchSize;
	m_loadedBatchId = batchId;
	m_loadedSubsetId = subsetId;

	allocateForLORs(hasTOF, stream);

	mp_lorDet1Pos->copyFromHost(tempBufferLorDet1Pos_ptr, batchSize, stream,
	                            false);
	mp_lorDet2Pos->copyFromHost(tempBufferLorDet2Pos_ptr, batchSize, stream,
	                            false);
	mp_lorDet1Orient->copyFromHost(tempBufferLorDet1Orient_ptr, batchSize,
	                               stream, false);
	mp_lorDet2Orient->copyFromHost(tempBufferLorDet2Orient_ptr, batchSize,
	                               stream, false);
	if (hasTOF)
	{
		mp_lorTOFValue->copyFromHost(tempBufferLorTOFValue_ptr, batchSize,
		                             stream, false);
	}

	if (stream != nullptr)
	{
		cudaStreamSynchronize(*stream);
	}

	m_areLORsGathered = true;
}

void LORsDevice::initializeDeviceArrays()
{
	mp_lorDet1Pos = std::make_unique<DeviceArray<float4>>();
	mp_lorDet2Pos = std::make_unique<DeviceArray<float4>>();
	mp_lorDet1Orient = std::make_unique<DeviceArray<float4>>();
	mp_lorDet2Orient = std::make_unique<DeviceArray<float4>>();
	mp_lorTOFValue = std::make_unique<DeviceArray<float>>();
}

void LORsDevice::allocateForLORs(bool hasTOF, const cudaStream_t* stream)
{
	ASSERT_MSG(m_loadedBatchSize > 0, "No batch loaded");
	bool hasAllocated = false;

	hasAllocated |= mp_lorDet1Pos->allocate(m_loadedBatchSize, stream, false);
	hasAllocated |= mp_lorDet2Pos->allocate(m_loadedBatchSize, stream, false);
	hasAllocated |=
	    mp_lorDet1Orient->allocate(m_loadedBatchSize, stream, false);
	hasAllocated |=
	    mp_lorDet2Orient->allocate(m_loadedBatchSize, stream, false);
	if (hasTOF)
	{
		hasAllocated |=
		    mp_lorTOFValue->allocate(m_loadedBatchSize, stream, false);
	}

	if (hasAllocated && stream != nullptr)
	{
		cudaStreamSynchronize(*stream);
	}
}

std::shared_ptr<ScannerDevice> LORsDevice::getScannerDevice() const
{
	return mp_scannerDevice;
}

const Scanner& LORsDevice::getScanner() const
{
	return mp_scannerDevice->getScanner();
}

const float4* LORsDevice::getLorDet1PosDevicePointer() const
{
	return mp_lorDet1Pos->getDevicePointer();
}

const float4* LORsDevice::getLorDet1OrientDevicePointer() const
{
	return mp_lorDet1Orient->getDevicePointer();
}

const float4* LORsDevice::getLorDet2PosDevicePointer() const
{
	return mp_lorDet2Pos->getDevicePointer();
}

const float4* LORsDevice::getLorDet2OrientDevicePointer() const
{
	return mp_lorDet2Orient->getDevicePointer();
}

float4* LORsDevice::getLorDet1PosDevicePointer()
{
	return mp_lorDet1Pos->getDevicePointer();
}

float4* LORsDevice::getLorDet1OrientDevicePointer()
{
	return mp_lorDet1Orient->getDevicePointer();
}

float4* LORsDevice::getLorDet2PosDevicePointer()
{
	return mp_lorDet2Pos->getDevicePointer();
}

float4* LORsDevice::getLorDet2OrientDevicePointer()
{
	return mp_lorDet2Orient->getDevicePointer();
}

const float* LORsDevice::getLorTOFValueDevicePointer() const
{
	return mp_lorTOFValue->getDevicePointer();
}

float* LORsDevice::getLorTOFValueDevicePointer()
{
	return mp_lorTOFValue->getDevicePointer();
}

bool LORsDevice::areLORsGathered() const
{
	return m_areLORsGathered;
}

size_t LORsDevice::getLoadedBatchSize() const
{
	return m_loadedBatchSize;
}

size_t LORsDevice::getLoadedBatchId() const
{
	return m_loadedBatchId;
}

size_t LORsDevice::getLoadedSubsetId() const
{
	return m_loadedSubsetId;
}
