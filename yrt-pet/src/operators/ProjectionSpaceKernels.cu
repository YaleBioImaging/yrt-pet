/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../../include/yrt-pet/operators/ProjectionSpaceKernels.cuh"
#include "yrt-pet/geometry/Constants.hpp"

#include <complex>

namespace yrt
{
__global__ void divideMeasurements_kernel(const float* pd_dataIn,
                                          float* pd_dataOut,
                                          const size_t maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		if (pd_dataOut[eventId] > EPS_FLT)
		{
			pd_dataOut[eventId] = pd_dataIn[eventId] / pd_dataOut[eventId];
		}
	}
}

__global__ void addProjValues_kernel(const float* pd_dataIn, float* pd_dataOut,
                                     const size_t maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		pd_dataOut[eventId] += pd_dataIn[eventId];
	}
}

__global__ void invertProjValues_kernel(const float* pd_dataIn,
                                        float* pd_dataOut,
                                        const size_t maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		if (pd_dataIn[eventId] != 0.0f)
		{
			pd_dataOut[eventId] = 1.0f / pd_dataIn[eventId];
		}
		else
		{
			pd_dataOut[eventId] = 0.0f;
		}
	}
}

__global__ void convertToACFs_kernel(const float* pd_dataIn, float* pd_dataOut,
                                     const float unitFactor,
                                     const size_t maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		pd_dataOut[eventId] = exp(-pd_dataIn[eventId] * unitFactor);
	}
}

__global__ void multiplyProjValues_kernel(const float* pd_dataIn,
                                          float* pd_dataOut,
                                          const size_t maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		pd_dataOut[eventId] *= pd_dataIn[eventId];
	}
}

__global__ void multiplyProjValues_kernel(float scalar, float* pd_dataOut,
                                          const size_t maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		pd_dataOut[eventId] *= scalar;
	}
}

__global__ void clearProjections_kernel(float* pd_dataIn, float value,
                                        const size_t maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		pd_dataIn[eventId] = value;
	}
}
}  // namespace yrt
