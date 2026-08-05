/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/RawParameters.hpp"
#include "yrt-pet/scatter/ScatterDeviceKernels.cuh"
#include "yrt-pet/scatter/ScatterSpace.hpp"
#include "yrt-pet/scatter/SingleScatterSimulator.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/GPUStream.cuh"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplay.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <thread>
#include <vector>

namespace yrt::scatter
{

void SingleScatterSimulator::runSSSDevice(ScatterSpace& outScatterSpace,
                                          bool onlyDirectPlanes,
                                          size_t maxVRAM_bytes,
                                          const cudaStream_t* stream0,
                                          const cudaStream_t* stream1) const
{
	ASSERT_MSG(outScatterSpace.isMemoryValid(),
	           "Destination scatter-space array is unallocated");

	const size_t numTOFBins = outScatterSpace.getNumTOFBins();
	const size_t numPlanes = outScatterSpace.getNumPlanes();
	const size_t numAngles = outScatterSpace.getNumAngles();
	const size_t numThreads = globals::getNumThreads();

	// Number of bin in scatter space that will need to be computed
	ssize_t numScsBins;
	if (onlyDirectPlanes)
	{
		numScsBins = numTOFBins * numPlanes * numAngles * numAngles;
	}
	else
	{
		numScsBins = outScatterSpace.getSizeTotal();
	}

	ssize_t availMem_bytes = maxVRAM_bytes;
	if (availMem_bytes == 0)
	{
		constexpr float memShare = 0.9f;  // Use 90% of the available memory
		// Probe GPU memory and compute available memory
		size_t freeMem = 0, totalMem = 0;
		cudaMemGetInfo(&freeMem, &totalMem);
		// Overwrite
		availMem_bytes =
		    static_cast<ssize_t>(static_cast<float>(freeMem) * memShare);
	}

	// Compute the amount of memory needed in image space
	const ImageParams& mu_p = mr_mu.getParams();
	const ImageParams& lambda_p = mr_lambda.getParams();
	RawImageParams muParams = getRawImageParams(mu_p);
	RawImageParams lambdaParams = getRawImageParams(lambda_p);
	const ssize_t muSize_bytes = mu_p.nx * mu_p.ny * mu_p.nz * sizeof(float);
	const ssize_t lambdaSize_bytes =
	    lambda_p.nx * lambda_p.ny * lambda_p.nz * sizeof(float);
	const ssize_t imgSize_bytes = muSize_bytes + lambdaSize_bytes;
	const ssize_t sampleSize_bytes =
	    3 * static_cast<ssize_t>(m_numSamples) * sizeof(float);
	constexpr ssize_t bytesPerScsBin = sizeof(Line3D) +  // LOR
	                                   sizeof(float)     // TOF
	                                   + sizeof(float);  // Result

	// Compute the number of bins we will be able to fit in the memory
	// We use 2 device buffer sets, one being filled and one being transferred
	//  to CPU. (Ping-pong buffers)
	ssize_t maxBatchLORs = (availMem_bytes - imgSize_bytes - sampleSize_bytes) /
	                       (2 * bytesPerScsBin);

	if (maxBatchLORs < 0)
	{
		std::cerr << "Insufficient VRAM. Need: "
		          << imgSize_bytes + sampleSize_bytes << " bytes. Only "
		          << availMem_bytes << " bytes available." << std::endl;
		throw std::runtime_error("Insufficient VRAM");
	}

	ssize_t batchSize_bins =
	    std::min(numScsBins, maxBatchLORs > 0 ? maxBatchLORs : 1);

	// We compute the number of batches while considering the one extra batch
	//  being transferred back to CPU
	const ssize_t numBatches =
	    (numScsBins + batchSize_bins - 1) / batchSize_bins;

	auto getBatchOffset = [&](ssize_t b) { return b * batchSize_bins; };
	auto getBatchCount = [&](ssize_t b)
	{
		const ssize_t batchCount =
		    std::min(batchSize_bins, numScsBins - getBatchOffset(b));
		ASSERT(batchCount > 0);
		return batchCount;
	};

	const bool doubleBuffer = (numBatches > 1);

	// Upload to the GPU the parts that do not change from batch to batch
	float* d_muData = nullptr;
	float* d_lambdaData = nullptr;
	cudaMalloc(&d_muData, muSize_bytes);
	cudaMalloc(&d_lambdaData, lambdaSize_bytes);
	cudaMemcpy(d_muData, mr_mu.getRawPointer(), muSize_bytes,
	           cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambdaData, mr_lambda.getRawPointer(), lambdaSize_bytes,
	           cudaMemcpyHostToDevice);
	const RawImageConst d_mu{muParams, d_muData};
	const RawImageConst d_lambda{lambdaParams, d_lambdaData};

	// Upload the scatter samples to the GPU
	float *d_xSamp = nullptr, *d_ySamp = nullptr, *d_zSamp = nullptr;
	cudaMalloc(&d_xSamp, m_numSamples * sizeof(float));
	cudaMalloc(&d_ySamp, m_numSamples * sizeof(float));
	cudaMalloc(&d_zSamp, m_numSamples * sizeof(float));
	cudaMemcpy(d_xSamp, m_xSamples.data(), m_numSamples * sizeof(float),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(d_ySamp, m_ySamples.data(), m_numSamples * sizeof(float),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(d_zSamp, m_zSamples.data(), m_numSamples * sizeof(float),
	           cudaMemcpyHostToDevice);

	// Allocate device buffers for the SCS bin data (LOR, TOF, and result)
	const int numBufs = doubleBuffer ? 2 : 1;
	Line3D* d_lor[2] = {nullptr, nullptr};
	float *d_tof[2] = {nullptr, nullptr}, *d_res[2] = {nullptr, nullptr};
	for (int i = 0; i < numBufs; ++i)
	{
		cudaMalloc(&d_lor[i], batchSize_bins * sizeof(Line3D));
		cudaMalloc(&d_tof[i], batchSize_bins * sizeof(float));
		cudaMalloc(&d_res[i], batchSize_bins * sizeof(float));
	}

	// Host-side ping-pong buffers
	std::vector<Line3D> lorHost[2];
	std::vector<float> tofHost[2], resHost[2];
	std::vector<ssize_t> idxHost[2];  // Stores the output SCS bin
	for (int i = 0; i < numBufs; ++i)
	{
		lorHost[i].resize(batchSize_bins);
		tofHost[i].resize(batchSize_bins);
		resHost[i].resize(batchSize_bins);
		idxHost[i].resize(batchSize_bins);
	}

	// Create a CUDA stream for both buffers if we have a double buffer
	const cudaStream_t* streams[2] = {nullptr, nullptr};
	std::unique_ptr<GPUStream> stream0_ptr = nullptr;
	std::unique_ptr<GPUStream> stream1_ptr = nullptr;
	streams[0] = stream0;
	if (streams[0] == nullptr)
	{
		stream0_ptr = std::make_unique<GPUStream>();
		streams[0] = &stream0_ptr->getStream();
	}
	streams[1] = stream1;
	if (doubleBuffer && streams[1] == nullptr)
	{
		stream1_ptr = std::make_unique<GPUStream>();
		streams[1] = &stream1_ptr->getStream();
	}

	// Batch filling function
	auto fillBatch = [&](ssize_t batchIdx, std::vector<Line3D>& lorV,
	                     std::vector<float>& tofV,
	                     std::vector<ssize_t>& idxV) -> ssize_t
	{
		const ssize_t offset = getBatchOffset(batchIdx);
		const size_t count = getBatchCount(batchIdx);

		if (onlyDirectPlanes)
		{
			const ssize_t planeAngles = numAngles * numAngles;
			const ssize_t tofPlane = numPlanes * planeAngles;
			util::parallelForChunked(
			    count, numThreads,
			    [&](size_t localIdx, ssize_t)
			    {
				    // Unravel index
				    const size_t globalIdx = offset + localIdx;
				    const size_t tofBin = globalIdx / tofPlane;
				    const size_t rest = globalIdx % tofPlane;
				    const size_t planeIdx = rest / planeAngles;
				    const size_t rest2 = rest % planeAngles;
				    const size_t angleIdx1 = rest2 / numAngles;
				    const size_t angleIdx2 = rest2 % numAngles;

				    const ScatterSpace::ScatterSpaceIndex scsIdx{
				        tofBin, planeIdx, angleIdx1, planeIdx, angleIdx2};
				    const auto [tof_ps, lor] =
				        outScatterSpace.getTOFAndLORFromIndex(scsIdx);
				    lorV[localIdx] = lor;
				    tofV[localIdx] = tof_ps;
				    idxV[localIdx] = outScatterSpace.getFlatIdx(scsIdx);
			    });
		}
		else
		{
			util::parallelForChunked(
			    count, numThreads,
			    [&](size_t localIdx, unsigned int)
			    {
				    const ssize_t globalIdx = offset + localIdx;
				    const ScatterSpace::ScatterSpaceIndex scsIdx =
				        outScatterSpace.unravelIndex(globalIdx);
				    const auto [tof_ps, lor] =
				        outScatterSpace.getTOFAndLORFromIndex(scsIdx);
				    lorV[localIdx] = lor;
				    tofV[localIdx] = tof_ps;
				    idxV[localIdx] = globalIdx;
			    });
		}
		return count;
	};

	// Main batch loop with double-buffering

	fillBatch(0, lorHost[0], tofHost[0], idxHost[0]);  // Pre-fill batch 0

	ssize_t totalProcessed = 0;
	for (ssize_t b = 0; b < numBatches; ++b)
	{
		const int curBuf = static_cast<int>(b % 2);
		const int nextBuf = static_cast<int>((b + 1) % 2);
		const cudaStream_t& curStream = *streams[curBuf];
		const ssize_t curSize = getBatchCount(b);

		// Start filling next batch in parallel with GPU work
		std::thread fillThread;
		if (doubleBuffer && b + 1 < numBatches)
		{
			fillThread = std::thread(
			    [&, nextBuf, nb = b + 1]()
			    {
				    fillBatch(nb, lorHost[nextBuf], tofHost[nextBuf],
				              idxHost[nextBuf]);
			    });
		}

		// Upload to device (async)
		cudaMemcpyAsync(d_lor[curBuf], lorHost[curBuf].data(),
		                curSize * sizeof(Line3D), cudaMemcpyHostToDevice,
		                curStream);
		cudaMemcpyAsync(d_tof[curBuf], tofHost[curBuf].data(),
		                curSize * sizeof(float), cudaMemcpyHostToDevice,
		                curStream);
		cudaStreamSynchronize(curStream);

		// Launch kernel
		constexpr int blockSize = 256;
		const int gridSize =
		    static_cast<int>((curSize + blockSize - 1) / blockSize);
		computeSingleScatterInLORKernel<<<gridSize, blockSize, 0, curStream>>>(
		    d_lor[curBuf], d_tof[curBuf], d_res[curBuf],
		    static_cast<int>(curSize), m_numSamples, d_xSamp, d_ySamp, d_zSamp,
		    m_detectionThreshold, m_energyLLD, m_energyResolution, d_mu,
		    d_lambda);

		// Read results back (async)
		cudaMemcpyAsync(resHost[curBuf].data(), d_res[curBuf],
		                curSize * sizeof(float), cudaMemcpyDeviceToHost,
		                curStream);
		cudaStreamSynchronize(curStream);

		// Write results (parallel on CPU)
		util::parallelForChunked(curSize, numThreads,
		                         [&](size_t i, unsigned int)
		                         {
			                         outScatterSpace.setValueFlat(
			                             idxHost[curBuf][i],
			                             fmaxf(0.0f, resHost[curBuf][i]));
		                         });

		if (fillThread.joinable())
			fillThread.join();

		totalProcessed += curSize;
		std::cout << "Processed " << totalProcessed << "/" << numScsBins
		          << " bins" << std::endl;
	}

	// Cleanup
	for (int i = 0; i < numBufs; ++i)
	{
		cudaFree(d_lor[i]);
		cudaFree(d_tof[i]);
		cudaFree(d_res[i]);
	}
	cudaFree(d_xSamp);
	cudaFree(d_ySamp);
	cudaFree(d_zSamp);
	cudaFree(d_muData);
	cudaFree(d_lambdaData);
}

}  // namespace yrt::scatter
