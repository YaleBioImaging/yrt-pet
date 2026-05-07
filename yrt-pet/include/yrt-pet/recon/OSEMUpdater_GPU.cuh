/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageDevice.cuh"

#include <memory>
#include <utility>
#include <vector>

namespace yrt
{
class BinIterator;
class OSEM_GPU;

class OSEMUpdater_GPU
{
public:
	explicit OSEMUpdater_GPU(OSEM_GPU* pp_osem);
	~OSEMUpdater_GPU();

	// Iterates over all batches to compute the sensitivity image
	void computeSensitivityImage(ImageDevice& destImage) const;

	// Iterates over all batches to do the updates
	void computeEMUpdateImage(const ImageDevice& inputImage,
	                          ImageDevice& destImage) const;
	void preloadMultiGPULORCacheIfRequested() const;

private:
	struct MultiGPUImageBuffers
	{
		ImageParams params;
		std::vector<int> deviceIds;
		std::vector<std::unique_ptr<ImageDeviceOwned>> primaryPartialImages;
		std::vector<std::unique_ptr<ImageDeviceOwned>> workerInputImages;
		std::vector<std::unique_ptr<ImageDeviceOwned>> workerPartialImages;
		bool initialized = false;
	};
	struct MultiGPUReconWorkerContext;
	struct MultiGPUReconCache;

	void computeSensitivityImageMultiGPU(ImageDevice& destImage) const;
	void computeEMUpdateImageMultiGPU(const ImageDevice& inputImage,
	                                  ImageDevice& destImage) const;
	void ensureMultiGPUImageBuffers(MultiGPUImageBuffers& buffers,
	                                const ImageParams& params,
	                                bool allocateInputImages) const;
	void releaseMultiGPUImageBuffers(MultiGPUImageBuffers& buffers) const;
	bool isMultiGPULORCacheEnabled() const;
	void ensureMultiGPUReconCache(
	    int subsetId, const BinIterator& subsetIterator,
	    const std::vector<std::pair<size_t, size_t>>& ranges) const;
	void releaseMultiGPUReconCache() const;

	OSEM_GPU* mp_osem;
	mutable MultiGPUImageBuffers m_sensitivityBuffers;
	mutable MultiGPUImageBuffers m_emBuffers;
	mutable std::unique_ptr<MultiGPUReconCache> mp_reconCache;
};
}  // namespace yrt
