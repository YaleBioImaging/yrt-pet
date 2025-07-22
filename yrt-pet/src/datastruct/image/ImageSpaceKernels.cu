/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/image/ImageSpaceKernels.cuh"

namespace yrt
{
__global__ void updateEM_kernel(const float* pd_imgIn, float* pd_imgOut,
                                const float* pd_sensImg, const int nx,
                                const int ny, const int nz,
                                const float EM_threshold)
{
	const long id_z = blockIdx.z * blockDim.z + threadIdx.z;
	const long id_y = blockIdx.y * blockDim.y + threadIdx.y;
	const long id_x = blockIdx.x * blockDim.x + threadIdx.x;

	if (id_z < nz && id_y < ny && id_x < nx)
	{
		const long pixelId = id_z * nx * ny + id_y * nx + id_x;
		if (pd_sensImg[pixelId] > EM_threshold)
		{
			pd_imgOut[pixelId] *= pd_imgIn[pixelId] / pd_sensImg[pixelId];
		}
	}
}

__global__ void applyThreshold_kernel(
    float* pd_imgIn, const float* pd_imgMask, const float threshold,
    const float val_le_scale, const float val_le_off, const float val_gt_scale,
    const float val_gt_off, const int nx, const int ny, const int nz)
{
	const long id_z = blockIdx.z * blockDim.z + threadIdx.z;
	const long id_y = blockIdx.y * blockDim.y + threadIdx.y;
	const long id_x = blockIdx.x * blockDim.x + threadIdx.x;

	if (id_z < nz && id_y < ny && id_x < nx)
	{
		const long pixelId = id_z * nx * ny + id_y * nx + id_x;
		if (pd_imgMask[pixelId] <= threshold)
		{
			pd_imgIn[pixelId] = pd_imgIn[pixelId] * val_le_scale + val_le_off;
		}
		else
		{
			pd_imgIn[pixelId] = pd_imgIn[pixelId] * val_gt_scale + val_gt_off;
		}
	}
}

__global__ void setValue_kernel(float* pd_imgIn, const float value,
                                const int nx, const int ny, const int nz)
{
	const long id_z = blockIdx.z * blockDim.z + threadIdx.z;
	const long id_y = blockIdx.y * blockDim.y + threadIdx.y;
	const long id_x = blockIdx.x * blockDim.x + threadIdx.x;

	if (id_z < nz && id_y < ny && id_x < nx)
	{
		const long pixelId = id_z * nx * ny + id_y * nx + id_x;
		pd_imgIn[pixelId] = value;
	}
}

__global__ void addFirstImageToSecond_kernel(const float* pd_imgIn,
                                             float* pd_imgOut, int nx, int ny,
                                             int nz)
{
	const long id_z = blockIdx.z * blockDim.z + threadIdx.z;
	const long id_y = blockIdx.y * blockDim.y + threadIdx.y;
	const long id_x = blockIdx.x * blockDim.x + threadIdx.x;

	if (id_z < nz && id_y < ny && id_x < nx)
	{
		const long pixelId = id_z * nx * ny + id_y * nx + id_x;
		pd_imgOut[pixelId] = pd_imgOut[pixelId] + pd_imgIn[pixelId];
	}
}

__device__ constexpr int circular(int M, int x)
{
	if (x < 0)
	{
		return x + M;
	}
	if (x >= M)
	{
		return x - M;
	}
	return x;
}

__device__ constexpr int idx3(int x, int y, int z, int nx, int ny)
{
	return x + nx * (y + ny * z);
}

template <int Axis>
__global__ void convolve3DSeparable_kernel(const float* input, float* output,
                                           const float* kernel, int kernelSize,
                                           int nx, int ny, int nz)
{

	// Get the thread indices in 3D
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	static_assert(Axis < 3 && Axis >= 0);

	if (x < nx && y < ny && z < nz)
	{
		const int halfKernelSize = kernelSize / 2;

		// Perform the convolution
		float sum = 0.0f;

		for (int kk = -halfKernelSize; kk <= halfKernelSize; kk++)
		{
			int r;
			int imgIdx;

			if constexpr (Axis == 0)
			{
				r = circular(nx, x - kk);
				imgIdx = idx3(r, y, z, nx, ny);
			}
			else if constexpr (Axis == 1)
			{
				r = circular(ny, y - kk);
				imgIdx = idx3(x, r, z, nx, ny);
			}
			else if constexpr (Axis == 2)
			{
				r = circular(nz, z - kk);
				imgIdx = idx3(x, y, r, nx, ny);
			}

			sum += kernel[kk + halfKernelSize] * input[imgIdx];
		}

		// Write the result to the output array
		output[idx3(x, y, z, nx, ny)] = sum;
	}
}
template __global__ void convolve3DSeparable_kernel<0>(const float* input,
                                                       float* output,
                                                       const float* kernel,
                                                       int kernelSize, int nx,
                                                       int ny, int nz);
template __global__ void convolve3DSeparable_kernel<1>(const float* input,
                                                       float* output,
                                                       const float* kernel,
                                                       int kernelSize, int nx,
                                                       int ny, int nz);
template __global__ void convolve3DSeparable_kernel<2>(const float* input,
                                                       float* output,
                                                       const float* kernel,
                                                       int kernelSize, int nx,
                                                       int ny, int nz);
}  // namespace yrt
