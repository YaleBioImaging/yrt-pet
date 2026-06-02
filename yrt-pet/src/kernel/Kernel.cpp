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
                               ssize_t nz, ssize_t ny, ssize_t nx, int W,
                               float sigma2, int numThreads)
{
	const ssize_t numPixels = nx * ny * nz;
	const ssize_t numNeighbors = (2 * W + 1) * (2 * W + 1) * (2 * W + 1);
	float sc = -1.0f / (2.0f * sigma2);

	util::parallelForChunked(
	    numPixels, numThreads,
	    [nx, ny, nz, numNeighbors, x, W, k, k_i, k_j, sc](size_t i,
	                                                      size_t /*tid*/)
	    {
		    const ssize_t iz = i / (ny * nx);
		    const ssize_t iy = (i % (ny * nx)) / nx;
		    const ssize_t ix = i % nx;
		    const float v0 = x[IDX3(ix, iy, iz, nx, ny)];
		    int j = IDX2(0, i, numNeighbors);
		    for (int kz = -W; kz <= W; kz++)
		    {
			    const ssize_t jz = std::max(0l, std::min(nz - 1l, iz + kz));
			    for (int ky = -W; ky <= W; ky++)
			    {
				    const ssize_t jy = std::max(0l, std::min(ny - 1l, iy + ky));
				    for (int kx = -W; kx <= W; kx++)
				    {
					    const ssize_t jx =
					        std::max(0l, std::min(nx - 1l, ix + kx));
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
                                   ssize_t nz, ssize_t ny, ssize_t nx, int W,
                                   int P, int num_k, float sigma2,
                                   int numThreads)
{
	const ssize_t numPixels = nx * ny * nz;
	float sc = -1.0f / sigma2;
	std::unique_ptr<ssize_t[]> idxBuffer =
	    std::make_unique<ssize_t[]>(numThreads * num_k);
	std::unique_ptr<float[]> valBuffer =
	    std::make_unique<float[]>(numThreads * num_k);
	ssize_t* idxBufferPtr = idxBuffer.get();
	float* valBufferPtr = valBuffer.get();


	auto cmp = [](const std::pair<ssize_t, float>& left,
	              const std::pair<ssize_t, float>& right)
	{ return left.second < right.second; };

	util::parallelForChunked(
	    numPixels, numThreads,
	    [idxBufferPtr, valBufferPtr, num_k, nx, ny, nz, cmp, W, P, sc, x, k,
	     k_i, k_j](size_t i, size_t tid)
	    {
		    ssize_t* idxBufferT = idxBufferPtr + tid * num_k;
		    float* valBufferT = valBufferPtr + tid * num_k;

		    const ssize_t iz = i / (ny * nx);
		    const ssize_t iy = (i % (ny * nx)) / nx;
		    const ssize_t ix = i % nx;

		    // Find k nearest neighbors
		    std::priority_queue<std::pair<ssize_t, float>,
		                        std::vector<std::pair<ssize_t, float>>,
		                        decltype(cmp)>
		        n_list(cmp);

		    for (ssize_t jz = std::max(0l, iz - W);
		         jz <= std::min(nz - 1l, iz + W); jz++)
		    {
			    for (ssize_t jy = std::max(0l, iy - W);
			         jy <= std::min(ny - 1l, iy + W); jy++)
			    {
				    for (ssize_t jx = std::max(0l, ix - W);
				         jx <= std::min(nx - 1l, ix + W); jx++)
				    {
					    ssize_t j = jz * ny * nx + jy * nx + jx;
					    float d = 0.f;
					    for (ssize_t pz = -P; pz <= P; pz++)
					    {
						    const ssize_t v0_z = util::reflect(nz, iz + pz);
						    const ssize_t v1_z = util::reflect(nz, jz + pz);
						    for (int py = -P; py <= P; py++)
						    {
							    const ssize_t v0_y = util::reflect(ny, iy + py);
							    const ssize_t v1_y = util::reflect(ny, jy + py);
							    for (int px = -P; px <= P; px++)
							    {
								    const ssize_t v0_x =
								        util::reflect(nx, ix + px);
								    const ssize_t v1_x =
								        util::reflect(nx, jx + px);
								    const float v0 =
								        x[IDX3(v0_x, v0_y, v0_z, nx, ny)];
								    const float v1 =
								        x[IDX3(v1_x, v1_y, v1_z, nx, ny)];
								    d += (v0 - v1) * (v0 - v1);
							    }
						    }
					    }
					    if (static_cast<ssize_t>(n_list.size()) < num_k)
					    {
						    n_list.push(std::pair<ssize_t, float>(j, d));
					    }
					    else if (d < n_list.top().second)
					    {
						    n_list.pop();
						    n_list.push(std::pair<ssize_t, float>(j, d));
					    }
				    }
			    }
		    }

		    // Exponentiate, calculate norm
		    const ssize_t size = static_cast<ssize_t>(n_list.size());
		    float norm = 0.f;
		    for (ssize_t ki = 0l; ki < size; ki++)
		    {
			    const auto p = n_list.top();
			    const float d_out = expf(sc * p.second);
			    valBufferT[ki] = d_out;
			    idxBufferT[ki] = p.first;
			    norm += d_out;
			    n_list.pop();
		    }

		    ASSERT(size <= num_k);

		    // Populate K row
		    for (ssize_t ki = 0l; ki < size; ki++)
		    {
			    const ssize_t idx_flat = IDX2(ki, i, num_k);
			    k[idx_flat] = valBufferT[ki] / norm;
			    k_i[idx_flat] = i;
			    k_j[idx_flat] = idxBufferT[ki];
		    }
	    });
}

void kernel::build_K_full(float* x, float* k, int* k_i, int* k_j, ssize_t nz,
                          ssize_t ny, ssize_t nx, int num_k, float sigma2,
                          int numThreads)
{
	const ssize_t numPixels = nx * ny * nz;
	float sc = -1.0f / (2.0f * sigma2);

	auto cmp = [](const std::pair<ssize_t, float>& left,
	              const std::pair<ssize_t, float>& right)
	{ return left.second < right.second; };

	util::parallelForChunked(
	    numPixels, numThreads,
	    [numPixels, num_k, cmp, sc, x, k, k_i, k_j](size_t i, size_t /*tid*/)
	    {
		    const float v0 = x[i];

		    // Find k nearest neighbors
		    std::priority_queue<std::pair<ssize_t, float>,
		                        std::vector<std::pair<ssize_t, float>>,
		                        decltype(cmp)>
		        n_list(cmp);
		    for (ssize_t j = 0; j < numPixels; j++)
		    {
			    const float v1 = x[j];
			    const float d = std::abs(v0 - v1);
			    if (static_cast<ssize_t>(n_list.size()) < num_k)
			    {
				    n_list.push(std::pair<ssize_t, float>(j, d));
			    }
			    else if (d < n_list.top().second)
			    {
				    n_list.pop();
				    n_list.push(std::pair<ssize_t, float>(j, d));
			    }
		    }

		    ASSERT(static_cast<ssize_t>(n_list.size()) == num_k);

		    // Populate K row
		    int nb_idx = 0;
		    while (!n_list.empty())
		    {
			    const std::pair<ssize_t, float> nb_info = n_list.top();
			    const ssize_t j = nb_info.first;
			    const float d2 = nb_info.second * nb_info.second;
			    const ssize_t idx_flat = IDX2(nb_idx, i, num_k);
			    k[idx_flat] = expf(sc * d2);
			    k_i[idx_flat] = i;
			    k_j[idx_flat] = j;
			    nb_idx++;
			    n_list.pop();
		    }
	    });
}
}  // namespace yrt
