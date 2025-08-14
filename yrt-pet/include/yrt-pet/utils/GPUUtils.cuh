/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#if BUILD_CUDA

#include <iostream>

#include <cuda_runtime.h>

namespace yrt
{

// returns false if there is an error
__host__ bool cudaCheckError();

void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true);

void gpuErrchk(cudaError_t code);

namespace globals
{

__host__ size_t getDeviceInfo(bool verbose = false);

}

}  // namespace yrt

#endif

#if BUILD_CUDA
#define HOST_DEVICE_CALLABLE __host__ __device__
#else
#define HOST_DEVICE_CALLABLE
#endif
