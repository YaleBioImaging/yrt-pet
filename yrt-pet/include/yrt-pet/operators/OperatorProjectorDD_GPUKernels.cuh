/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/ProjectionPsfManagerDevice.cuh"
#include "yrt-pet/operators/TimeOfFlight.hpp"
#include "yrt-pet/recon/CUParameters.hpp"
#include "yrt-pet/utils/GPUKernelUtils.cuh"

#ifdef BUILD_CUDA

namespace yrt
{

// Note: This is useless for now
template <bool IsForward, bool HasTOF, bool HasProjPSF>
__global__ void OperatorProjectorDDCU_kernel(
    float* pd_projValues, float* pd_image, const char* pd_projectionProperties,
    const ProjectionPropertyManager* pd_projPropManager,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);

}  // namespace yrt

#endif
