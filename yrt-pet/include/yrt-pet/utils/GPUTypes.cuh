/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "Assert.hpp"

#include <driver_types.h>
#include <vector_types.h>

namespace yrt
{

class GPUBatchSetup
{
public:
	GPUBatchSetup(size_t totalSize, size_t maxBatchSize)
	{
		ASSERT(maxBatchSize > 0);
		// Note: totalSize and maxBatchSize must have the same units
		numBatches = totalSize / maxBatchSize + 1;
		lastBatchSize = totalSize % maxBatchSize;
		batchSize = maxBatchSize;

		// Special case for when there's only one batch
		if (numBatches == 1)
		{
			batchSize = lastBatchSize;
		}
		// Special case for when batches fit perfectly
		if (lastBatchSize == 0)
		{
			numBatches--;
			lastBatchSize = batchSize;
		}
	}
	[[nodiscard]] size_t getBatchSize(int batchId) const
	{
		ASSERT(batchId < numBatches);
		if (batchId == numBatches - 1)
		{
			return lastBatchSize;
		}
		return batchSize;
	}
	[[nodiscard]] int getNumBatches() const { return numBatches; }

private:
	size_t batchSize;
	size_t lastBatchSize;
	int numBatches;
};

// Structure encoding how a kernel should be called
struct GPULaunchConfig
{
	// Which stream to run
	const cudaStream_t* stream = nullptr;
	// Whether to synchronize after the operation or not
	bool synchronize = true;
};

struct GPULaunchParams
{
	unsigned int gridSize;
	unsigned int blockSize;
};

struct GPULaunchParams3D
{
	dim3 gridSize;
	dim3 blockSize;
};

}  // namespace yrt
