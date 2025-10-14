/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

namespace yrt
{
// kernels definitions
__global__ void divideMeasurements_kernel(const float* d_dataIn,
                                          float* d_dataOut,
                                          size_t maxNumberOfEvents);
__global__ void addProjValues_kernel(const float* d_dataIn, float* d_dataOut,
                                     size_t maxNumberOfEvents);
__global__ void invertProjValues_kernel(const float* d_dataIn, float* d_dataOut,
                                        size_t maxNumberOfEvents);
__global__ void convertToACFs_kernel(const float* d_dataIn, float* d_dataOut,
                                     float unitFactor, size_t maxNumberOfEvents);
__global__ void multiplyProjValues_kernel(const float* d_dataIn,
                                          float* d_dataOut,
                                          size_t maxNumberOfEvents);
__global__ void multiplyProjValues_kernel(float scalar, float* d_dataOut,
                                          size_t maxNumberOfEvents);
__global__ void clearProjections_kernel(float* d_dataIn, float value,
                                        size_t maxNumberOfEvents);
}  // namespace yrt
