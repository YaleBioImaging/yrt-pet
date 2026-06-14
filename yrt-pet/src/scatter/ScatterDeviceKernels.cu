/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/ScatterDeviceKernels.cuh"

#include <cuda_runtime.h>

namespace yrt::scatter
{

__global__ void computeSingleScatterInLORKernel(
    const float3* lorPoints1, const float3* lorPoints2, const float* tofValues,
    float* results, int numLORs, int numSamples, const float* xSamples,
    const float* ySamples, const float* zSamples, float energyLLD,
    float sigmaEnergy, float crystalDepth, float axialFOV,
    float collimatorRadius, CrystalMaterial crystalMaterial, Cylinder cyl1,
    Cylinder cyl2, Plane endPlate1, Plane endPlate2, const float* mu_data,
    const float* lambda_data, RawImageParams mu_params,
    RawImageParams lambda_params, float3 imageOffset)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numLORs)
		return;

	Line3D lor;
	lor.point1.x = lorPoints1[idx].x;
	lor.point1.y = lorPoints1[idx].y;
	lor.point1.z = lorPoints1[idx].z;
	lor.point2.x = lorPoints2[idx].x;
	lor.point2.y = lorPoints2[idx].y;
	lor.point2.z = lorPoints2[idx].z;

	results[idx] = computeSingleScatterInLOR(
	    lor, tofValues[idx], numSamples, xSamples, ySamples, zSamples,
	    energyLLD, sigmaEnergy, crystalDepth, axialFOV, collimatorRadius,
	    crystalMaterial, cyl1, cyl2, endPlate1, endPlate2, mu_data, lambda_data,
	    mu_params, lambda_params, imageOffset);
}

void launchComputeSingleScatterInLOR(
    const float3* lorPoints1, const float3* lorPoints2, const float* tofValues,
    float* results, int numLORs, const float* xSamples, const float* ySamples,
    const float* zSamples, int numSamples, float energyLLD, float sigmaEnergy,
    float crystalDepth, float axialFOV, float collimatorRadius,
    CrystalMaterial crystalMaterial, const Cylinder& cyl1, const Cylinder& cyl2,
    const Plane& endPlate1, const Plane& endPlate2, const float* d_muData,
    const float* d_lambdaData, const RawImageParams& muParams,
    const RawImageParams& lambdaParams, float3 imageOffset, cudaStream_t stream)
{
	float3 *d_lorP1 = nullptr, *d_lorP2 = nullptr;
	float *d_results = nullptr, *d_tof = nullptr;
	float *d_xSamp = nullptr, *d_ySamp = nullptr, *d_zSamp = nullptr;

	cudaMalloc(&d_lorP1, numLORs * sizeof(float3));
	cudaMalloc(&d_lorP2, numLORs * sizeof(float3));
	cudaMalloc(&d_tof, numLORs * sizeof(float));
	cudaMalloc(&d_results, numLORs * sizeof(float));
	cudaMalloc(&d_xSamp, numSamples * sizeof(float));
	cudaMalloc(&d_ySamp, numSamples * sizeof(float));
	cudaMalloc(&d_zSamp, numSamples * sizeof(float));

	cudaMemcpyAsync(d_lorP1, lorPoints1, numLORs * sizeof(float3),
	                cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_lorP2, lorPoints2, numLORs * sizeof(float3),
	                cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_tof, tofValues, numLORs * sizeof(float),
	                cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_xSamp, xSamples, numSamples * sizeof(float),
	                cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_ySamp, ySamples, numSamples * sizeof(float),
	                cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_zSamp, zSamples, numSamples * sizeof(float),
	                cudaMemcpyHostToDevice, stream);

	constexpr int blockSize = 256;
	int gridSize = (numLORs + blockSize - 1) / blockSize;

	computeSingleScatterInLORKernel<<<gridSize, blockSize, 0, stream>>>(
	    d_lorP1, d_lorP2, d_tof, d_results, numLORs, numSamples, d_xSamp,
	    d_ySamp, d_zSamp, energyLLD, sigmaEnergy, crystalDepth, axialFOV,
	    collimatorRadius, crystalMaterial, cyl1, cyl2, endPlate1, endPlate2,
	    d_muData, d_lambdaData, muParams, lambdaParams, imageOffset);

	cudaMemcpyAsync(results, d_results, numLORs * sizeof(float),
	                cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	cudaFree(d_lorP1);
	cudaFree(d_lorP2);
	cudaFree(d_tof);
	cudaFree(d_results);
	cudaFree(d_xSamp);
	cudaFree(d_ySamp);
	cudaFree(d_zSamp);
}

}  // namespace yrt::scatter
