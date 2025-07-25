/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/ProjectionPsfManagerDevice.cuh"
#include "yrt-pet/operators/TimeOfFlight.hpp"
#include "yrt-pet/recon/CUParameters.hpp"
#include "yrt-pet/utils/GPUKernelUtils.cuh"

#ifdef BUILD_CUDA

namespace yrt
{

// Note: This is useless for now
__global__ void gatherLORs_kernel(const uint2* pd_lorDetsId,
                                  const float4* pd_detsPos,
                                  const float4* pd_detsOrient,
                                  float4* pd_lorDet1Pos, float4* pd_lorDet2Pos,
                                  float4* pd_lorDet1Orient,
                                  float4* pd_lorDet2Orient,
                                  CUImageParams imgParams, size_t batchSize);

template <bool IsForward, bool HasTOF, bool HasProjPSF>
__global__ void OperatorProjectorDDCU_kernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);

}  // namespace yrt

#endif
