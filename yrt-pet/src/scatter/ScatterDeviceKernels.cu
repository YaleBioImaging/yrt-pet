/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/ScatterDeviceKernels.cuh"
#include "yrt-pet/utils/GPUStream.cuh"

#include <cuda_runtime.h>

namespace yrt::scatter
{

__global__ void computeSingleScatterInLORKernel(
    const Line3D* lorData, const float* tofValues, float* results, int numLORs,
    int numSamples, const float* xSamples, const float* ySamples,
    const float* zSamples, float energyLLD, float sigmaEnergy,
    float crystalDepth, float axialFOV, float collimatorRadius,
    CrystalMaterial crystalMaterial, Cylinder cyl1, Cylinder cyl2,
    Plane endPlate1, Plane endPlate2, RawImageConst mu, RawImageConst lambda)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numLORs)
		return;

	Line3D lor = lorData[idx];

	results[idx] = computeSingleScatterInLOR(
	    lor, tofValues[idx], numSamples, xSamples, ySamples, zSamples,
	    energyLLD, sigmaEnergy, crystalDepth, axialFOV, collimatorRadius,
	    crystalMaterial, cyl1, cyl2, endPlate1, endPlate2, mu, lambda);
}

void launchComputeSingleScatterInLOR(
    const Line3D* lorData, const float* tofValues, float* results, int numLORs,
    const float* xSamples, const float* ySamples, const float* zSamples,
    int numSamples, float energyLLD, float sigmaEnergy, float crystalDepth,
    float axialFOV, float collimatorRadius, CrystalMaterial crystalMaterial,
    const Cylinder& cyl1, const Cylinder& cyl2, const Plane& endPlate1,
    const Plane& endPlate2, const RawImageConst& d_mu,
    const RawImageConst& d_lambda, cudaStream_t* stream)
{
	Line3D* d_lorData = nullptr;
	float *d_results = nullptr, *d_tof = nullptr;
	float *d_xSamp = nullptr, *d_ySamp = nullptr, *d_zSamp = nullptr;

	cudaMalloc(&d_lorData, numLORs * sizeof(Line3D));
	cudaMalloc(&d_tof, numLORs * sizeof(float));
	cudaMalloc(&d_results, numLORs * sizeof(float));
	cudaMalloc(&d_xSamp, numSamples * sizeof(float));
	cudaMalloc(&d_ySamp, numSamples * sizeof(float));
	cudaMalloc(&d_zSamp, numSamples * sizeof(float));

	cudaMemcpyAsync(d_lorData, lorData, numLORs * sizeof(Line3D),
	                cudaMemcpyHostToDevice, stream ? *stream : nullptr);
	cudaMemcpyAsync(d_tof, tofValues, numLORs * sizeof(float),
	                cudaMemcpyHostToDevice, stream ? *stream : nullptr);
	cudaMemcpyAsync(d_xSamp, xSamples, numSamples * sizeof(float),
	                cudaMemcpyHostToDevice, stream ? *stream : nullptr);
	cudaMemcpyAsync(d_ySamp, ySamples, numSamples * sizeof(float),
	                cudaMemcpyHostToDevice, stream ? *stream : nullptr);
	cudaMemcpyAsync(d_zSamp, zSamples, numSamples * sizeof(float),
	                cudaMemcpyHostToDevice, stream ? *stream : nullptr);

	constexpr int blockSize = 256;
	int gridSize = (numLORs + blockSize - 1) / blockSize;

	if (stream != nullptr)
	{
		computeSingleScatterInLORKernel<<<gridSize, blockSize, 0, *stream>>>(
		    d_lorData, d_tof, d_results, numLORs, numSamples, d_xSamp, d_ySamp,
		    d_zSamp, energyLLD, sigmaEnergy, crystalDepth, axialFOV,
		    collimatorRadius, crystalMaterial, cyl1, cyl2, endPlate1, endPlate2,
		    d_mu, d_lambda);
	}
	else
	{
		computeSingleScatterInLORKernel<<<gridSize, blockSize>>>(
		    d_lorData, d_tof, d_results, numLORs, numSamples, d_xSamp, d_ySamp,
		    d_zSamp, energyLLD, sigmaEnergy, crystalDepth, axialFOV,
		    collimatorRadius, crystalMaterial, cyl1, cyl2, endPlate1, endPlate2,
		    d_mu, d_lambda);
	}

	cudaMemcpyAsync(results, d_results, numLORs * sizeof(float),
	                cudaMemcpyDeviceToHost, stream ? *stream : nullptr);
	if (stream != nullptr)
	{
		cudaStreamSynchronize(*stream);
	}

	cudaFree(d_lorData);
	cudaFree(d_tof);
	cudaFree(d_results);
	cudaFree(d_xSamp);
	cudaFree(d_ySamp);
	cudaFree(d_zSamp);
}

}  // namespace yrt::scatter
