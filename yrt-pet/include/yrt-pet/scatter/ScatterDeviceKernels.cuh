/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#ifdef __CUDACC__

#include "yrt-pet/geometry/Cylinder.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/geometry/Plane.hpp"
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
    CrystalMaterial crystalMaterial, Cylinder cyl1, Cylinder cyl2,
    Plane endPlate1, Plane endPlate2, RawImageConst mu, RawImageConst lambda);

}  // namespace yrt::scatter

#endif
