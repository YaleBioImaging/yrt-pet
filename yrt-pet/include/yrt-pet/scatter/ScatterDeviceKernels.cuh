/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#ifdef __CUDACC__

#include "yrt-pet/geometry/Line3D.cuh"
#include "yrt-pet/recon/RawParameters.hpp"
#include "yrt-pet/scatter/SingleScatterSimulatorUtils.cuh"

#include <cuda_runtime.h>

namespace yrt::scatter
{

__global__ void computeSingleScatterInLORKernel(
    const Line3D* lorData, const float* tofValues, float* results, int numLORs,
    int numSamples, const float* xSamples, const float* ySamples,
    const float* zSamples, float detectionThreshdold, float energyLLD,
    float energyResolution, RawImageConst mu, RawImageConst lambda);

}  // namespace yrt::scatter

#endif
