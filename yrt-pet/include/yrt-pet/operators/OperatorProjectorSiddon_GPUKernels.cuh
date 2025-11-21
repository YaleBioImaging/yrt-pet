/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/ProjectionPsfManagerDevice.cuh"
#include "yrt-pet/operators/OperatorProjectorUpdaterDevice.cuh"
#include "yrt-pet/operators/TimeOfFlight.hpp"
#include "yrt-pet/recon/CUParameters.hpp"
#include "yrt-pet/utils/GPUKernelUtils.cuh"

#ifdef BUILD_CUDA

namespace yrt
{
template <bool IsForward, bool HasTOF, bool IsIncremental, bool IsMultiRay>
__global__ void OperatorProjectorSiddonCU_kernel(
    float* pd_projValues, float* pd_image, OperatorProjectorUpdaterDevice* pd_updater,
    const char* pd_projProperties,
    const ProjectionPropertyManager* pd_projPropManager,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, int p_numRays, size_t batchSize);
}

#endif
