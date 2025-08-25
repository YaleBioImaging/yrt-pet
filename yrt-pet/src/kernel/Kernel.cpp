/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/kernel/Kernel.hpp"
#include "yrt-pet/utils/Array.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Tools.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <queue>
#include <utility>

namespace yrt
{
void kernel::build_K_neighbors(float* x, float* k, int* k_i, int* k_j,
                               size_t nz, size_t ny, size_t nx, int W,
                               float sigma2, int numThreads)
{

	size_t numPixels = nx * ny * nz;
	size_t num_neighbors = (2 * W + 1) * (2 * W + 1) * (2 * W + 1);
	float sc = -1.0f / (2.0f * sigma2);

	util::parallelForChunked(
	    numPixels, numThreads,
	    [nx, ny, nz, num_neighbors, x, W, k, k_i, k_j, sc](size_t i,
	                                                       size_t /*tid*/)
	    {
		    int iz = i / (ny * nx);
		    int iy = (i % (ny * nx)) / nx;
		    int ix = i % nx;
		    float v0 = x[IDX3(ix, iy, iz, nx, ny)];
		    int j = IDX2(0, i, num_neighbors);
		    for (int kz = -W; kz <= W; kz++)
		    {
			    int jz = std::max(0, std::min((int)nz - 1, iz + kz));
			    for (int ky = -W; ky <= W; ky++)
			    {
				    int jy = std::max(0, std::min((int)ny - 1, iy + ky));
				    for (int kx = -W; kx <= W; kx++)
				    {
					    int jx = std::max(0, std::min((int)nx - 1, ix + kx));
					    float v1 = x[IDX3(jx, jy, jz, nx, ny)];
					    k[j] = expf(sc * (v0 - v1) * (v0 - v1));
					    k_i[j] = i;
					    k_j[j] = jz * ny * nx + jy * nx + jx;
					    j++;
				    }
			    }
		    }
	    });
}

void kernel::build_K_knn_neighbors(float* x, float* k, int* k_i, int* k_j,
                                   size_t nz, size_t ny, size_t nx, int W,
                                   int P, int num_k, float sigma2,
                                   int numThreads)
{
	size_t numPixels = nx * ny * nz;
	float sc = -1.0f / sigma2;
	std::unique_ptr<size_t[]> idxBuffer =
	    std::make_unique<size_t[]>(numThreads * num_k);
	std::unique_ptr<float[]> valBuffer =
	    std::make_unique<float[]>(numThreads * num_k);
	size_t* idxBufferPtr = idxBuffer.get();
	float* valBufferPtr = valBuffer.get();


	auto cmp = [](std::pair<size_t, float> left, std::pair<size_t, float> right)
	{ return left.second < right.second; };

	util::parallelForChunked(
	    numPixels, numThreads,
	    [idxBufferPtr, valBufferPtr, num_k, nx, ny, nz, cmp, W, P, sc, x, k,
	     k_i, k_j](size_t i, size_t tid)
	    {
		    size_t* idxBufferT = idxBufferPtr + tid * num_k;
		    float* valBufferT = valBufferPtr + tid * num_k;

		    int iz = i / (ny * nx);
		    int iy = (i % (ny * nx)) / nx;
		    int ix = i % nx;

		    // Find k nearest neighbors
		    std::priority_queue<std::pair<size_t, float>,
		                        std::vector<std::pair<size_t, float>>,
		                        decltype(cmp)>
		        n_list(cmp);

		    for (int jz = std::max(0, iz - W);
		         jz <= std::min((int)nz - 1, iz + W); jz++)
		    {
			    for (int jy = std::max(0, iy - W);
			         jy <= std::min((int)ny - 1, iy + W); jy++)
			    {
				    for (int jx = std::max(0, ix - W);
				         jx <= std::min((int)nx - 1, ix + W); jx++)
				    {
					    size_t j = jz * ny * nx + jy * nx + jx;
					    float d = 0.f;
					    for (int pz = -P; pz <= P; pz++)
					    {
						    int v0_z = util::reflect(nz, iz + pz);
						    int v1_z = util::reflect(nz, jz + pz);
						    for (int py = -P; py <= P; py++)
						    {
							    int v0_y = util::reflect(ny, iy + py);
							    int v1_y = util::reflect(ny, jy + py);
							    for (int px = -P; px <= P; px++)
							    {
								    int v0_x = util::reflect(nx, ix + px);
								    int v1_x = util::reflect(nx, jx + px);
								    float v0 =
								        x[IDX3(v0_x, v0_y, v0_z, nx, ny)];
								    float v1 =
								        x[IDX3(v1_x, v1_y, v1_z, nx, ny)];
								    d += (v0 - v1) * (v0 - v1);
							    }
						    }
					    }
					    if ((int)n_list.size() < num_k)
					    {
						    n_list.push(std::pair<size_t, float>(j, d));
					    }
					    else if (d < n_list.top().second)
					    {
						    n_list.pop();
						    n_list.push(std::pair<size_t, float>(j, d));
					    }
				    }
			    }
		    }

		    // Exponentiate, calculate norm
		    const size_t size = n_list.size();
		    float norm = 0.f;
		    for (size_t ki = 0; ki < size; ki++)
		    {
			    const auto p = n_list.top();
			    const float d_out = expf(sc * p.second);
			    valBufferT[ki] = d_out;
			    idxBufferT[ki] = p.first;
			    norm += d_out;
			    n_list.pop();
		    }

		    ASSERT((int)size <= num_k);

		    // Populate K row
		    for (size_t ki = 0; ki < size; ki++)
		    {
			    const size_t idx_flat = IDX2(ki, i, num_k);
			    k[idx_flat] = valBufferT[ki] / norm;
			    k_i[idx_flat] = i;
			    k_j[idx_flat] = idxBufferT[ki];
		    }
	    });
}

void kernel::build_K_full(float* x, float* k, int* k_i, int* k_j, size_t nz,
                          size_t ny, size_t nx, int num_k, float sigma2,
                          int numThreads)
{
	size_t numPixels = nx * ny * nz;
	float sc = -1.0f / (2.0f * sigma2);

	auto cmp = [](std::pair<size_t, float> left, std::pair<size_t, float> right)
	{ return left.second < right.second; };

	util::parallelForChunked(
	    numPixels, numThreads,
	    [numPixels, num_k, cmp, sc, x, k, k_i, k_j](size_t i, size_t /*tid*/)
	    {
		    float v0 = x[i];

		    // Find k nearest neighbors
		    std::priority_queue<std::pair<size_t, float>,
		                        std::vector<std::pair<size_t, float>>,
		                        decltype(cmp)>
		        n_list(cmp);
		    for (size_t j = 0; j < numPixels; j++)
		    {
			    float v1 = x[j];
			    float d = std::abs(v0 - v1);
			    if ((int)n_list.size() < num_k)
			    {
				    n_list.push(std::pair<size_t, float>(j, d));
			    }
			    else if (d < n_list.top().second)
			    {
				    n_list.pop();
				    n_list.push(std::pair<size_t, float>(j, d));
			    }
		    }

		    ASSERT((int)n_list.size() == num_k);

		    // Populate K row
		    int nb_idx = 0;
		    while (!n_list.empty())
		    {
			    std::pair<size_t, float> nb_info = n_list.top();
			    size_t j = nb_info.first;
			    float d2 = nb_info.second * nb_info.second;
			    size_t idx_flat = IDX2(nb_idx, i, num_k);
			    k[idx_flat] = expf(sc * d2);
			    k_i[idx_flat] = i;
			    k_j[idx_flat] = j;
			    nb_idx++;
			    n_list.pop();
		    }
	    });
}
}  // namespace yrt
