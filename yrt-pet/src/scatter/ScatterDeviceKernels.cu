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
    const float* zSamples, float detectionThreshold, float energyLLD,
    float energyResolution, RawImageConst mu, RawImageConst lambda)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numLORs)
	{
		Line3D lor = lorData[idx];
		float result = 0.0f;

		if (lor.isValid())
		{
			result = computeSingleScatterInLOR(
			    lor, tofValues[idx], numSamples, xSamples, ySamples, zSamples,
			    detectionThreshold, energyLLD, energyResolution, mu, lambda);
		}
		results[idx] = result;
	}
}

}  // namespace yrt::scatter
