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

namespace yrt::scatter
{

void launchComputeSingleScatterInLOR(
    const float3* lorPoints1, const float3* lorPoints2, const float* tofValues,
    float* results, int numLORs, const float* xSamples,
    const float* ySamples, const float* zSamples, int numSamples,
    float energyLLD, float sigmaEnergy, float crystalDepth, float axialFOV,
    float collimatorRadius, CrystalMaterial crystalMaterial,
    const Cylinder& cyl1, const Cylinder& cyl2, const Plane& endPlate1,
    const Plane& endPlate2, const float* d_muData, const float* d_lambdaData, const RawImageParams& muParams,
    const RawImageParams& lambdaParams, float3 imageOffset,
    cudaStream_t stream = 0);

}  // namespace yrt::scatter

#endif
