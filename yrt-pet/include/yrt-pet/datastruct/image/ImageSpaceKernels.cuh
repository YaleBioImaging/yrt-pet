/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Types.hpp"

namespace yrt
{

__global__ void updateEM_kernel(const float* d_imgIn, float* d_imgOut,
                                const float* d_sensImg, int nx, int ny, int nz,
                                float EM_threshold);
__global__ void updateEMDynamic_kernel(const float* d_imgIn, float* d_imgOut,
								const float* d_sensImg, int nx, int ny, int nz,
								int nt, const float* c_r, float EM_threshold);

__global__ void applyThreshold_kernel(float* pd_imgIn, const float* pd_imgMask,
                                      float threshold, float val_le_scale,
                                      float val_le_off, float val_gt_scale,
                                      float val_gt_off, int nx, int ny, int nz);

__global__ void applyThresholdBroadcast_kernel(float* pd_imgIn, const float* pd_imgMask,
									  float threshold, float val_le_scale,
									  float val_le_off, float val_gt_scale,
									  float val_gt_off, int nx, int ny, int nz,
									  int nt);

__global__ void setValue_kernel(float* d_imgIn, float value, int nx, int ny,
                                int nz);

__global__ void addFirstImageToSecond_kernel(const float* d_imgIn,
                                             float* d_imgOut, int nx, int ny,
                                             int nz);

template <bool WEIGHED_AVG>
__global__ void
    timeAverageMoveImage_kernel(const float* pd_imgIn, float* pd_imgOut, int nx,
                                int ny, int nz, float length_x, float length_y,
                                float length_z, float off_x, float off_y,
                                float off_z, const transform_t* pd_invTransforms,
                                float* frameWeights, int numTransforms);

template <int Axis>
__global__ void convolve3DSeparable_kernel(const float* input, float* output,
                                           const float* kernel, int kernelSize,
                                           int nx, int ny, int nz);
}  // namespace yrt
