/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/ProjectorUtils.hpp"
#include "operators/OperatorProjectorSiddon_GPUKernels.cuh"
#include "operators/ProjectionPsfManagerDevice.cuh"
#include "operators/ProjectionPsfUtils.cuh"

#include <cuda_runtime.h>


template <bool IsForward, bool HasTOF>
__global__ void OperatorProjectorSiddonCU_kernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < batchSize)
	{
		/* TODO NOW*/
	}
}

template __global__ void OperatorProjectorSiddonCU_kernel<true, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorSiddonCU_kernel<false, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorSiddonCU_kernel<true, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorSiddonCU_kernel<false, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);

