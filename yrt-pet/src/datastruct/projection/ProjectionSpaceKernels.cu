/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/ProjectionSpaceKernels.cuh"
#include "geometry/Constants.hpp"

#include <complex>

__global__ void divideMeasurements_kernel(const float* d_dataIn,
                                          float* d_dataOut,
                                          const int maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		if (d_dataOut[eventId] > EPS_FLT)
		{
			d_dataOut[eventId] = d_dataIn[eventId] / d_dataOut[eventId];
		}
	}
}

__global__ void addProjValues_kernel(const float* d_dataIn, float* d_dataOut,
                                     const int maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		d_dataOut[eventId] += d_dataIn[eventId];
	}
}

__global__ void invertProjValues_kernel(const float* d_dataIn, float* d_dataOut,
                                        const int maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		if (d_dataIn[eventId] != 0.0f)
		{
			d_dataOut[eventId] = 1.0f / d_dataIn[eventId];
		}
		else
		{
			d_dataOut[eventId] = 0.0f;
		}
	}
}

__global__ void convertToACFs_kernel(const float* d_dataIn, float* d_dataOut,
                                     const float unitFactor,
                                     const int maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		d_dataOut[eventId] = exp(-d_dataIn[eventId] * unitFactor);
	}
}

__global__ void multiplyProjValues_kernel(const float* d_dataIn,
                                          float* d_dataOut,
                                          const int maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		d_dataOut[eventId] *= d_dataIn[eventId];
	}
}

__global__ void multiplyProjValues_kernel(float scalar, float* d_dataOut,
                                          const int maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		d_dataOut[eventId] *= scalar;
	}
}

__global__ void clearProjections_kernel(float* d_dataIn, float value,
                                        const int maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		d_dataIn[eventId] = value;
	}
}
