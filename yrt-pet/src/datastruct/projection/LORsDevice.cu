/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/LORsDevice.cuh"

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/operators/OperatorProjectorDevice.cuh"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/PageLockedBuffer.cuh"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

namespace yrt
{
LORsDevice::LORsDevice()
    : m_hasTOF(false),
      m_precomputedBatchSize(0ull),
      m_precomputedBatchId(-1),
      m_precomputedSubsetId(-1),
      m_areLORsPrecomputed(false),
      m_loadedBatchSize(0ull),
      m_loadedBatchId(-1),
      m_loadedSubsetId(-1)
{
	initializeDeviceArrays();
}

void LORsDevice::precomputeBatchLORs(const BinIterator& binIter,
                                     const GPUBatchSetup& batchSetup,
                                     int subsetId, int batchId,
                                     const ProjectionData& reference)
{
	if (m_precomputedSubsetId != subsetId || m_precomputedBatchId != batchId ||
	    m_areLORsPrecomputed == false)
	{
		m_areLORsPrecomputed = false;
		m_hasTOF = reference.hasTOF();

		const size_t batchSize = batchSetup.getBatchSize(batchId);

		m_tempLorDet1Pos.reAllocateIfNeeded(batchSize);
		m_tempLorDet2Pos.reAllocateIfNeeded(batchSize);
		m_tempLorDet1Orient.reAllocateIfNeeded(batchSize);
		m_tempLorDet2Orient.reAllocateIfNeeded(batchSize);
		float4* tempBufferLorDet1Pos_ptr = m_tempLorDet1Pos.getPointer();
		float4* tempBufferLorDet2Pos_ptr = m_tempLorDet2Pos.getPointer();
		float4* tempBufferLorDet1Orient_ptr = m_tempLorDet1Orient.getPointer();
		float4* tempBufferLorDet2Orient_ptr = m_tempLorDet2Orient.getPointer();

		float* tempBufferLorTOFValue_ptr = nullptr;
		if (m_hasTOF)
		{
			m_tempLorTOFValue.reAllocateIfNeeded(batchSize);
			tempBufferLorTOFValue_ptr = m_tempLorTOFValue.getPointer();
		}

		const size_t offset = batchId * batchSetup.getBatchSize(0);
		auto* binIter_ptr = &binIter;
		const ProjectionData* reference_ptr = &reference;

		bin_t binId;
		size_t binIdx;
#pragma omp parallel for default(none) private(binIdx, binId)                \
    firstprivate(offset, batchSize, binIter_ptr, tempBufferLorDet1Pos_ptr,   \
                     tempBufferLorDet2Pos_ptr, tempBufferLorDet1Orient_ptr,  \
                     tempBufferLorDet2Orient_ptr, tempBufferLorTOFValue_ptr, \
                     reference_ptr, m_hasTOF)
		for (binIdx = 0; binIdx < batchSize; binIdx++)
		{
			binId = binIter_ptr->get(binIdx + offset);
			auto [bin, lor, tofValue, det1Orient, det2Orient] =
			    reference_ptr->getProjectionProperties(binId);

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
			    if (m_hasTOF)
			    {
				    tempBufferLorTOFValue_ptr[binIdx] = tofValue;
			    }
		    });

		m_precomputedBatchSize = batchSize;
		m_precomputedBatchId = batchId;
		m_precomputedSubsetId = subsetId;
	}

	m_areLORsPrecomputed = true;
}

void LORsDevice::loadPrecomputedLORsToDevice(GPULaunchConfig launchConfig)
{
	const cudaStream_t* stream = launchConfig.stream;

	if (m_loadedSubsetId != m_precomputedSubsetId ||
	    m_loadedBatchId != m_precomputedBatchId)
	{
		allocateForPrecomputedLORsIfNeeded({stream, false});

		mp_lorDet1Pos->copyFromHost(m_tempLorDet1Pos.getPointer(),
		                            m_precomputedBatchSize, {stream, false});
		mp_lorDet2Pos->copyFromHost(m_tempLorDet2Pos.getPointer(),
		                            m_precomputedBatchSize, {stream, false});
		mp_lorDet1Orient->copyFromHost(m_tempLorDet1Orient.getPointer(),
		                               m_precomputedBatchSize, {stream, false});
		mp_lorDet2Orient->copyFromHost(m_tempLorDet2Orient.getPointer(),
		                               m_precomputedBatchSize, {stream, false});
		if (m_hasTOF)
		{
			mp_lorTOFValue->copyFromHost(m_tempLorTOFValue.getPointer(),
			                             m_precomputedBatchSize,
			                             {stream, false});
		}

		// In case the LOR loading is done for other reasons than projections
		if (launchConfig.synchronize == true)
		{
			if (stream != nullptr)
			{
				cudaStreamSynchronize(*stream);
			}
			else
			{
				cudaDeviceSynchronize();
			}
		}

		m_loadedBatchSize = m_precomputedBatchSize;
		m_loadedBatchId = m_precomputedBatchId;
		m_loadedSubsetId = m_precomputedSubsetId;
	}
}

size_t LORsDevice::getPrecomputedBatchSize() const
{
	return m_precomputedBatchSize;
}

int LORsDevice::getPrecomputedBatchId() const
{
	return m_precomputedBatchId;
}

int LORsDevice::getPrecomputedSubsetId() const
{
	return m_precomputedSubsetId;
}

void LORsDevice::initializeDeviceArrays()
{
	mp_lorDet1Pos = std::make_unique<DeviceArray<float4>>();
	mp_lorDet2Pos = std::make_unique<DeviceArray<float4>>();
	mp_lorDet1Orient = std::make_unique<DeviceArray<float4>>();
	mp_lorDet2Orient = std::make_unique<DeviceArray<float4>>();
	mp_lorTOFValue = std::make_unique<DeviceArray<float>>();
}

void LORsDevice::allocateForPrecomputedLORsIfNeeded(
    GPULaunchConfig launchConfig)
{
	ASSERT_MSG(m_precomputedBatchSize > 0, "No batch of LORs precomputed");
	bool hasAllocated = false;

	hasAllocated |= mp_lorDet1Pos->allocate(m_precomputedBatchSize,
	                                        {launchConfig.stream, false});
	hasAllocated |= mp_lorDet2Pos->allocate(m_precomputedBatchSize,
	                                        {launchConfig.stream, false});
	hasAllocated |= mp_lorDet1Orient->allocate(m_precomputedBatchSize,
	                                           {launchConfig.stream, false});
	hasAllocated |= mp_lorDet2Orient->allocate(m_precomputedBatchSize,
	                                           {launchConfig.stream, false});
	if (m_hasTOF)
	{
		hasAllocated |= mp_lorTOFValue->allocate(m_precomputedBatchSize,
		                                         {launchConfig.stream, false});
	}

	if (hasAllocated && launchConfig.synchronize)
	{
		if (launchConfig.stream != nullptr)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
		else
		{
			cudaDeviceSynchronize();
		}
	}
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
	return m_areLORsPrecomputed;
}

size_t LORsDevice::getLoadedBatchSize() const
{
	return m_loadedBatchSize;
}

int LORsDevice::getLoadedBatchId() const
{
	return m_loadedBatchId;
}

int LORsDevice::getLoadedSubsetId() const
{
	return m_loadedSubsetId;
}

}  // namespace yrt
