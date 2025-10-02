/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/image/ImageSpaceKernels.cuh"
#include "yrt-pet/datastruct/image/ImageUtils.cuh"
#include "yrt-pet/geometry/TransformUtils.hpp"

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

template <bool WEIGHED_AVG>
__global__ void timeAverageMoveImage_kernel(
    const float* pd_imgIn, float* pd_imgOut, int nx, int ny, int nz,
    float length_x, float length_y, float length_z, float off_x, float off_y,
    float off_z, const transform_t* pd_invTransforms, float* frameWeights,
    int numTransforms)
{
	const long id_x = blockIdx.x * blockDim.x + threadIdx.x;
	const long id_y = blockIdx.y * blockDim.y + threadIdx.y;
	const long id_z = blockIdx.z * blockDim.z + threadIdx.z;

	if (id_z < nz && id_y < ny && id_x < nx)
	{
		const long flatId = id_z * nx * ny + id_y * nx + id_x;

		const float vx = length_x / nx;
		const float vy = length_y / ny;
		const float vz = length_z / nz;

		const float inv_vx = 1 / vx;
		const float inv_vy = 1 / vy;
		const float inv_vz = 1 / vz;

		const float origin_x = off_x - length_x * 0.5f + vx * 0.5f;
		const float origin_y = off_y - length_y * 0.5f + vy * 0.5f;
		const float origin_z = off_z - length_z * 0.5f + vz * 0.5f;

		const float pos_x = util::indexToPosition(id_x, vx, length_x, off_x);
		const float pos_y = util::indexToPosition(id_y, vy, length_y, off_y);
		const float pos_z = util::indexToPosition(id_z, vz, length_z, off_z);

		// If weights pointer is null, use 1.0 everywhere
		float frameWeight = 1.0f;
		float voxelWeights[8];
		int voxelIndices[8];

		// Output voxel value
		float outVoxelValue = pd_imgOut[flatId];

		for (int transform_i = 0; transform_i < numTransforms; transform_i++)
		{
			if constexpr (WEIGHED_AVG)
			{
				frameWeight = frameWeights[transform_i];
			}

			const transform_t inv = pd_invTransforms[transform_i];

			// Apply inverse transform (matrix-vector multiply)
			const float newX =
			    fmaf(inv.r00, pos_x,
			         fmaf(inv.r01, pos_y, fmaf(inv.r02, pos_z, inv.tx)));
			const float newY =
			    fmaf(inv.r10, pos_x,
			         fmaf(inv.r11, pos_y, fmaf(inv.r12, pos_z, inv.ty)));
			const float newZ =
			    fmaf(inv.r20, pos_x,
			         fmaf(inv.r21, pos_y, fmaf(inv.r22, pos_z, inv.tz)));

			// Interpolate into the image
			util::trilinearInterpolateCore(
			    newX, newY, newZ, nx, ny, nz, origin_x, origin_y, origin_z,
			    inv_vx, inv_vy, inv_vz, voxelIndices, voxelWeights);

			for (size_t i = 0; i < 8; i++)
			{
				outVoxelValue +=
				    pd_imgIn[voxelIndices[i]] * voxelWeights[i] * frameWeight;
			}
		}

		pd_imgOut[flatId] = outVoxelValue;
	}
}
template __global__ void timeAverageMoveImage_kernel<true>(
    const float* d_imgIn, float* d_imgOut, int nx, int ny, int nz,
    float length_x, float length_y, float length_z, float off_x, float off_y,
    float off_z, const transform_t* transforms, float* frameWeights,
    int numTransforms);
template __global__ void timeAverageMoveImage_kernel<false>(
    const float* d_imgIn, float* d_imgOut, int nx, int ny, int nz,
    float length_x, float length_y, float length_z, float off_x, float off_y,
    float off_z, const transform_t* transforms, float* frameWeights,
    int numTransforms);

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

__global__ void convolve3D_kernel_F(const float* input, float* output,
								  const float* kernel,
								  int kernelSizeX, int kernelSizeY, int kernelSizeZ,
								  int nx, int ny, int nz)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= nx || y >= ny || z >= nz) return;

	float sum = 0.0f;

	const int halfX = kernelSizeX / 2;
	const int halfY = kernelSizeY / 2;
	const int halfZ = kernelSizeZ / 2;

	for (int kz = -halfZ; kz <= halfZ; kz++) {
		for (int ky = -halfY; ky <= halfY; ky++) {
			for (int kx = -halfX; kx <= halfX; kx++) {
				int r_x = circular(nx, x + kx);
				int r_y = circular(ny, y + ky);
				int r_z = circular(nz, z + kz);

				int imgIdx = idx3(r_x, r_y, r_z, nx, ny);

				int kernelIdx = (kz + halfZ) * (kernelSizeY * kernelSizeX)
							  + (ky + halfY) * kernelSizeX
							  + (kx + halfX);

				sum += kernel[kernelIdx] * input[imgIdx];
			}
		}
	}
	output[idx3(x, y, z, nx, ny)] = sum;
}

__global__ void convolve3D_kernel_T(const float* input, float* output,
								  const float* kernel,
								  int kernelSizeX, int kernelSizeY, int kernelSizeZ,
								  int nx, int ny, int nz)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= nx || y >= ny || z >= nz) return;

	float temp1 = input[idx3(x, y, z, nx, ny)];

	const int halfX = kernelSizeX / 2;
	const int halfY = kernelSizeY / 2;
	const int halfZ = kernelSizeZ / 2;

	for (int kz = -halfZ; kz <= halfZ; kz++) {
		for (int ky = -halfY; ky <= halfY; ky++) {
			for (int kx = -halfX; kx <= halfX; kx++) {
				int r_x = circular(nx, x + kx);
				int r_y = circular(ny, y + ky);
				int r_z = circular(nz, z + kz);

				int outIdx = idx3(r_x, r_y, r_z, nx, ny);

				int kernelIdx = (kz + halfZ) * (kernelSizeY * kernelSizeX)
							  + (ky + halfY) * kernelSizeX
							  + (kx + halfX);

				atomicAdd(&output[outIdx], temp1 * kernel[kernelIdx]);
			}
		}
	}
}
//?? combine f and t together using template
}  // namespace yrt
