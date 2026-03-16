/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/LORsDevice.cuh"

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/operators/OperatorProjectorDevice.cuh"


namespace yrt
{
LORsDevice::LORsDevice(const std::vector<Constraint*>& constraints,
                       const std::set<ProjectionPropertyType>& projProperties)
    : m_precomputedBatchSize(0ull),
      m_precomputedBatchId(-1),
      m_precomputedSubsetId(-1),
      m_areLORsPrecomputed(false),
      m_loadedBatchSize(0ull),
      m_loadedBatchId(-1),
      m_loadedSubsetId(-1)
{
	initializeDeviceArray(projProperties);
	initBinLoader(constraints, projProperties);
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

		const size_t batchSize = batchSetup.getBatchSize(batchId);

		allocateBinFilterIfNeeded(batchSize);

		const size_t offset = batchId * batchSetup.getBatchSize(0);
		const BinIteratorBatched binIterForBatch(&binIter, offset, batchSize);

		// The batch loading:
		mp_binLoader->collectFromBins(reference, binIterForBatch, true);

		// Save for future
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

		mp_projectionProperties->getDeviceArray().copyFromHost(
		    mp_binLoader->getProjectionPropertiesRawPointer(),
		    m_precomputedBatchSize * getElementSize(), {stream, false});

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

const ProjectionPropertyManager*
    LORsDevice::getProjectionPropertyManagerDevicePointer() const
{
	return mp_projectionProperties->getManagerDevicePointer();
}

const PropertyUnit* LORsDevice::getProjectionPropertiesDevicePointer() const
{
	return mp_projectionProperties->getDevicePointer();
}

PropertyUnit* LORsDevice::getProjectionPropertiesDevicePointer()
{
	return mp_projectionProperties->getDevicePointer();
}

bool LORsDevice::areLORsGathered() const
{
	return m_areLORsPrecomputed;
}

size_t LORsDevice::getElementSize() const
{
	return mp_binLoader->getPropertyManager().getElementSize();
}

void LORsDevice::initializeDeviceArray(
    const std::set<ProjectionPropertyType>& projProperties)
{
	mp_projectionProperties =
	    std::make_unique<PropStructDevice<ProjectionPropertyType>>(
	        projProperties);
}

void LORsDevice::initBinLoader(
    const std::vector<Constraint*>& constraints,
    const std::set<ProjectionPropertyType>& projProperties)
{
	mp_binLoader = std::make_unique<BinLoader>(constraints, projProperties);
}

void LORsDevice::allocateForPrecomputedLORsIfNeeded(
    GPULaunchConfig launchConfig)
{
	ASSERT_MSG(m_precomputedBatchSize > 0, "No batch of LORs precomputed");
	const bool hasAllocated = mp_projectionProperties->allocate(
	    m_precomputedBatchSize, {launchConfig.stream, false});

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

void LORsDevice::allocateBinFilterIfNeeded(size_t newBatchSize)
{
	if (mp_binLoader->getProjectionPropertiesSize() < newBatchSize)
	{
		mp_binLoader->allocate(newBatchSize);
	}
}

}  // namespace yrt
