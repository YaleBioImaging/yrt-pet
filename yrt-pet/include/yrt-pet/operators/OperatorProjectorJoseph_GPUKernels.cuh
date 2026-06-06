/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/TimeOfFlight.hpp"
#include "yrt-pet/recon/CUParameters.hpp"
#include "yrt-pet/utils/GPUKernelUtils.cuh"

#ifdef BUILD_CUDA

namespace yrt
{
template <bool IsForward, bool HasTOF>
__global__ void OperatorProjectorJosephCU_kernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
}

#endif
