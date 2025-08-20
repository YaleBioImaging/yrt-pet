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

#include "omp.h"
#include <cstddef>
#include <cstring>
#include <functional>

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

template <bool flagSensOrRecon>
void LORsDevice::precomputeBatchLORs(
    const BinIterator& binIter, const GPUBatchSetup& batchSetup, int subsetId,
    int batchId, const ProjectionData& reference,
    const BinIteratorConstrained& binIterConstrained)
{
	if (m_precomputedSubsetId != subsetId || m_precomputedBatchId != batchId ||
	    m_areLORsPrecomputed == false)
	{
		m_areLORsPrecomputed = false;
		m_hasTOF = reference.hasTOF();

		const size_t batchSize = batchSetup.getBatchSize(batchId);
		const int numThreads = globals::getNumThreads();
		const size_t blockSize = std::ceil(batchSize / (float)numThreads);

		auto projPropManager = binIterConstrained.getPropertyManagerRecon();
		m_elementSize = projPropManager.getElementSize();
		auto projectionProperties = projPropManager.createDataArray(batchSize);
		auto projectionPropertiesPtr = projectionProperties.get();

		m_tempProjectionProperties.reAllocateIfNeeded(
		    batchSize * projPropManager.getElementSize());
		char* tempBufferProjectionProperties_ptr =
		    m_tempProjectionProperties.getPointer();
		auto info = binIterConstrained.getConstraintManager().createDataArray(
		    numThreads);
		auto infoPtr = info.get();

		const size_t offset = batchId * batchSetup.getBatchSize(0);
		auto* binIter_ptr = &binIter;
		const ProjectionData* reference_ptr = &reference;

		util::parallel_do_indexed(
		    numThreads,
		    [offset, blockSize, batchSize, &binIterConstrained, projPropManager,
		     binIter_ptr, &infoPtr, &tempBufferProjectionProperties_ptr,
		     reference_ptr](int tid)
		    {
			    for (size_t binIdx = tid * blockSize;
			         binIdx < std::min({(tid + 1) * blockSize, batchSize});
			         binIdx++)
			    {
				    bin_t bin = binIter_ptr->get(binIdx + offset);
				    if constexpr (flagSensOrRecon)
				    {
					    binIterConstrained.collectInfoSens(
					        bin, binIdx, *reference_ptr,
					        tempBufferProjectionProperties_ptr, infoPtr);
				    }
				    else
				    {
					    binIterConstrained.collectInfoRecon(
					        bin, binIdx, *reference_ptr,
					        tempBufferProjectionProperties_ptr, infoPtr);
				    }
				    if (binIterConstrained.isValid(infoPtr))
				    {
					    reference_ptr->getProjectionProperties(
					        tempBufferProjectionProperties_ptr, projPropManager,
					        bin, tid);
				    }
				    else
				    {
					    float* ptr = projPropManager.getDataPtr<float>(
					        tempBufferProjectionProperties_ptr, binIdx,
					        ProjectionPropertyType::LOR);
					    memset(ptr, 0, 6 * sizeof(float));
				    }
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

		mp_projectionProperties->copyFromHost(
		    m_tempProjectionProperties.getPointer(),
		    m_precomputedBatchSize * m_elementSize, {stream, false});

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
	mp_projectionProperties = std::make_unique<DeviceArray<char>>();
}

void LORsDevice::allocateForPrecomputedLORsIfNeeded(
    GPULaunchConfig launchConfig)
{
	ASSERT_MSG(m_precomputedBatchSize > 0, "No batch of LORs precomputed");
	const bool hasAllocated = mp_projectionProperties->allocate(
		m_precomputedBatchSize * m_elementSize,
		{launchConfig.stream, false});

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

const char* LORsDevice::getProjectionPropertiesDevicePointer() const
{
	return mp_projectionProperties->getDevicePointer();
}

char* LORsDevice::getProjectionPropertiesDevicePointer()
{
	return mp_projectionProperties->getDevicePointer();
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
