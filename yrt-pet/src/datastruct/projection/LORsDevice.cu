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

		auto projPropManager = binIterConstrained.getPropertyManager();
		m_elementSize = projPropManager.getElementSize();

		m_tempProjectionProperties.reAllocateIfNeeded(batchSize *
		                                              m_elementSize);
		char* tempBufferProjectionProperties_ptr =
		    m_tempProjectionProperties.getPointer();
		auto consManager = binIterConstrained.getConstraintManager();
		auto info = consManager.createDataArray(numThreads);
		auto infoPtr = info.get();

		const size_t offset = batchId * batchSetup.getBatchSize(0);
		auto* binIter_ptr = &binIter;
		const ProjectionData* reference_ptr = &reference;

		util::parallelForChunked(
			batchSize, numThreads,
		    [offset, batchSize, &binIterConstrained, consManager,
		     projPropManager, binIter_ptr, &infoPtr,
		     &tempBufferProjectionProperties_ptr,
		     reference_ptr](size_t binIdx, int tid)
		    {
			    bin_t bin = binIter_ptr->get(binIdx + offset);
			    binIterConstrained.collectInfo(
			        bin, binIdx, tid, *reference_ptr,
			        tempBufferProjectionProperties_ptr, infoPtr);
			    if (binIterConstrained.isValid(consManager, infoPtr))
			    {
				    reference_ptr->getProjectionProperties(
				        tempBufferProjectionProperties_ptr, projPropManager,
				        bin, binIdx);
			    }
			    else
			    {
				    // Assume first 6 floats of ProjectionProperties are LOR
				    // end-points
				    float* ptr = projPropManager.getDataPtr<float>(
				        tempBufferProjectionProperties_ptr, binIdx,
				        ProjectionPropertyType::LOR);
				    memset(ptr, 0, sizeof(Line3D));
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
	    m_precomputedBatchSize * m_elementSize, {launchConfig.stream, false});

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
