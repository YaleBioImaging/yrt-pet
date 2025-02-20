/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/ProjectionPsfManagerDevice.cuh"
#include "operators/TimeOfFlight.hpp"
#include "recon/CUParameters.hpp"
#include "utils/GPUKernelUtils.cuh"

#ifdef __CUDACC__

template <bool IsForward, bool HasTOF, bool IsIncremental, bool IsMultiRay>
__global__ void OperatorProjectorSiddonCU_kernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, int p_numRays, size_t batchSize);

#endif
