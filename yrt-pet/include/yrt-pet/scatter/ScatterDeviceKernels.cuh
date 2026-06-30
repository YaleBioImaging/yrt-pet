/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#ifdef __CUDACC__

#include "yrt-pet/geometry/Cylinder3D.cuh"
#include "yrt-pet/geometry/Line3D.cuh"
#include "yrt-pet/geometry/Plane3D.cuh"
#include "yrt-pet/recon/RawParameters.hpp"
#include "yrt-pet/scatter/Crystal.hpp"
#include "yrt-pet/scatter/SingleScatterSimulatorUtils.cuh"

#include <cuda_runtime.h>

namespace yrt::scatter
{

__global__ void computeSingleScatterInLORKernel(
    const Line3D* lorData, const float* tofValues, float* results, int numLORs,
    int numSamples, const float* xSamples, const float* ySamples,
    const float* zSamples, float energyLLD, float sigmaEnergy,
    float crystalDepth, float axialFOV, float collimatorRadius,
    CrystalMaterial crystalMaterial, Cylinder3D cyl1, Cylinder3D cyl2,
    Plane3D endPlate1, Plane3D endPlate2, RawImageConst mu,
    RawImageConst lambda);

}  // namespace yrt::scatter

#endif
