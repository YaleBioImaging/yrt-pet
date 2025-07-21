/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cstdio>

namespace yrt
{
namespace kernel
{

void build_K_neighbors(float* x, float* k, int* k_i, int* k_j, size_t nz,
                       size_t ny, size_t nx, int W, float sigma2,
                       int numThreads);
void build_K_full(float* x, float* k, int* k_i, int* k_j, size_t nz, size_t ny,
                  size_t nx, int num_k, float sigma2, int numThreads);
void build_K_knn_neighbors(float* x, float* k, int* k_i, int* k_j, size_t nz,
                           size_t ny, size_t nx, int W, int P, int num_k,
                           float sigma2, int numThreads);
}  // namespace kernel
}  // namespace yrt
