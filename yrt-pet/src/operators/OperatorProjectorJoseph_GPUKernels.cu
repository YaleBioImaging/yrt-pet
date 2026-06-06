/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/geometry/ProjectorUtils.hpp"
#include "yrt-pet/operators/OperatorProjectorJoseph_GPUKernels.cuh"

#include <cuda_runtime.h>

namespace yrt
{
namespace
{
struct JosephLine
{
	float p1x;
	float p1y;
	float p1z;
	float p2x;
	float p2y;
	float p2z;
};

struct JosephAxisCache
{
	float halfLength;
	float voxel;
	float start;
	float invDelta;
	float halfAlphaStep;
	float rayLength;
};

__device__ inline int joseph_image_offset(int vx, int vy, int vz,
                                          const CUImageParams& imgParams)
{
	const int nx = imgParams.voxelNumber[0];
	const int ny = imgParams.voxelNumber[1];
	return vx + nx * (vy + ny * vz);
}

__device__ inline float joseph_half_length(const CUImageParams& imgParams,
                                           int axis)
{
	return 0.5f * imgParams.imgLength[axis];
}

__device__ inline float joseph_axis_coord(const JosephLine& line, int axis,
                                          float alpha)
{
	if (axis == 0)
	{
		return line.p1x + alpha * (line.p2x - line.p1x);
	}
	if (axis == 1)
	{
		return line.p1y + alpha * (line.p2y - line.p1y);
	}
	return line.p1z + alpha * (line.p2z - line.p1z);
}

__device__ inline float joseph_axis_delta(const JosephLine& line, int axis)
{
	if (axis == 0)
	{
		return line.p2x - line.p1x;
	}
	if (axis == 1)
	{
		return line.p2y - line.p1y;
	}
	return line.p2z - line.p1z;
}

__device__ inline float joseph_axis_start(const JosephLine& line, int axis)
{
	if (axis == 0)
	{
		return line.p1x;
	}
	if (axis == 1)
	{
		return line.p1y;
	}
	return line.p1z;
}

__device__ inline int joseph_axis_size(const CUImageParams& imgParams,
                                       int axis)
{
	return imgParams.voxelNumber[axis];
}

__device__ inline float joseph_grid_coord(float coord, float halfLength,
                                          float invVoxel)
{
	return (coord + halfLength) * invVoxel - 0.5f;
}

template <bool HasTOF>
__device__ inline bool joseph_alpha_range(
    const JosephLine& line, const CUImageParams& imgParams, float rayLength,
    float tofValue, const TimeOfFlightHelper* tofHelper, float& alphaMin,
    float& alphaMax)
{
	const float dx = line.p2x - line.p1x;
	const float dy = line.p2y - line.p1y;
	const float dz = line.p2z - line.p1z;
	if (dx == 0.0f && dy == 0.0f && dz == 0.0f)
	{
		alphaMin = 1.0f;
		alphaMax = 0.0f;
		return false;
	}

	float fovMin = 0.0f;
	float fovMax = 1.0f;
	const float a = dx * dx + dy * dy;
	const float b = 2.0f * (dx * line.p1x + dy * line.p1y);
	const float c = line.p1x * line.p1x + line.p1y * line.p1y -
	                imgParams.fovRadius * imgParams.fovRadius;
	const float delta = b * b - 4.0f * a * c;
	if (a != 0.0f)
	{
		if (delta <= 0.0f)
		{
			alphaMin = 1.0f;
			alphaMax = 0.0f;
			return false;
		}
		const float sqrtDelta = sqrtf(delta);
		fovMin = (-b - sqrtDelta) / (2.0f * a);
		fovMax = (-b + sqrtDelta) / (2.0f * a);
	}

	const float invX = dx == 0.0f ? 0.0f : 1.0f / dx;
	const float invY = dy == 0.0f ? 0.0f : 1.0f / dy;
	const float invZ = dz == 0.0f ? 0.0f : 1.0f / dz;
	float axMin;
	float axMax;
	float ayMin;
	float ayMax;
	float azMin;
	float azMax;
	util::get_alpha(-joseph_half_length(imgParams, 0),
	                joseph_half_length(imgParams, 0), line.p1x, line.p2x,
	                invX, axMin, axMax);
	util::get_alpha(-joseph_half_length(imgParams, 1),
	                joseph_half_length(imgParams, 1), line.p1y, line.p2y,
	                invY, ayMin, ayMax);
	util::get_alpha(-joseph_half_length(imgParams, 2),
	                joseph_half_length(imgParams, 2), line.p1z, line.p2z,
	                invZ, azMin, azMax);

	alphaMin = max(0.0f, fovMin, axMin, ayMin, azMin);
	alphaMax = min(1.0f, fovMax, axMax, ayMax, azMax);

	if constexpr (HasTOF)
	{
		float tofMin;
		float tofMax;
		tofHelper->getAlphaRange(tofMin, tofMax, rayLength, tofValue);
		alphaMin = fmaxf(alphaMin, tofMin);
		alphaMax = fminf(alphaMax, tofMax);
	}

	return alphaMin < alphaMax;
}

__device__ inline int joseph_major_axis(const JosephLine& line,
                                        const CUImageParams& imgParams)
{
	const float sx =
	    fabsf(line.p2x - line.p1x) / imgParams.voxelSize[0];
	const float sy =
	    fabsf(line.p2y - line.p1y) / imgParams.voxelSize[1];
	const float sz =
	    fabsf(line.p2z - line.p1z) / imgParams.voxelSize[2];
	if (sx >= sy && sx >= sz)
	{
		return 0;
	}
	return sy >= sz ? 1 : 2;
}

__device__ inline bool joseph_sample_bounds(
    const JosephLine& line, const CUImageParams& imgParams, int axis,
    float alphaMin, float alphaMax, int& first, int& last)
{
	const float invVoxel = 1.0f / imgParams.voxelSize[axis];
	const float halfLength = joseph_half_length(imgParams, axis);
	const float grid0 = joseph_grid_coord(
	    joseph_axis_coord(line, axis, alphaMin), halfLength, invVoxel);
	const float grid1 = joseph_grid_coord(
	    joseph_axis_coord(line, axis, alphaMax), halfLength, invVoxel);
	first = static_cast<int>(ceilf(fminf(grid0, grid1)));
	last = static_cast<int>(floorf(fmaxf(grid0, grid1)));
	first = first < 0 ? 0 : first;
	const int maxIndex = joseph_axis_size(imgParams, axis) - 1;
	last = last > maxIndex ? maxIndex : last;
	return first <= last;
}

__device__ inline JosephAxisCache
    joseph_make_axis_cache(const JosephLine& line,
                           const CUImageParams& imgParams, int axis,
                           float rayLength)
{
	const float dAxis = joseph_axis_delta(line, axis);

	JosephAxisCache cache;
	cache.halfLength = joseph_half_length(imgParams, axis);
	cache.voxel = imgParams.voxelSize[axis];
	cache.start = joseph_axis_start(line, axis);
	cache.invDelta = dAxis == 0.0f ? 0.0f : 1.0f / dAxis;
	cache.halfAlphaStep =
	    dAxis == 0.0f ? 0.0f : 0.5f * cache.voxel / fabsf(dAxis);
	cache.rayLength = rayLength;
	return cache;
}

__device__ inline float joseph_sample_alpha(const JosephAxisCache& cache,
                                            int majorIndex)
{
	const float centerCoord =
	    -cache.halfLength + (static_cast<float>(majorIndex) + 0.5f) *
	                            cache.voxel;
	return (centerCoord - cache.start) * cache.invDelta;
}

template <bool HasTOF>
__device__ inline float joseph_sample_weight(
    const JosephAxisCache& cache, float centerAlpha, float alphaMin,
    float alphaMax, float tofValue, const TimeOfFlightHelper* tofHelper)
{
	const float segmentStart =
	    fmaxf(alphaMin, centerAlpha - cache.halfAlphaStep);
	const float segmentEnd =
	    fminf(alphaMax, centerAlpha + cache.halfAlphaStep);
	if (segmentStart >= segmentEnd)
	{
		return 0.0f;
	}

	float weight = cache.rayLength * (segmentEnd - segmentStart);
	if constexpr (HasTOF)
	{
		weight *= tofHelper->getWeight(cache.rayLength, tofValue,
		                               segmentStart * cache.rayLength,
		                               segmentEnd * cache.rayLength);
	}
	return weight;
}

__device__ inline float joseph_image_value(const float* image, int vx, int vy,
                                           int vz,
                                           const CUImageParams& imgParams)
{
	if (vx < 0 || vy < 0 || vz < 0 ||
	    vx >= imgParams.voxelNumber[0] || vy >= imgParams.voxelNumber[1] ||
	    vz >= imgParams.voxelNumber[2])
	{
		return 0.0f;
	}
	return image[joseph_image_offset(vx, vy, vz, imgParams)];
}

__device__ inline float joseph_bilinear_forward(
    const float* image, int axis, int majorIndex, float alpha,
    const JosephLine& line, const CUImageParams& imgParams)
{
	const float x = line.p1x + alpha * (line.p2x - line.p1x);
	const float y = line.p1y + alpha * (line.p2y - line.p1y);
	const float z = line.p1z + alpha * (line.p2z - line.p1z);
	if (axis == 0)
	{
		const float gy = joseph_grid_coord(
		    y, joseph_half_length(imgParams, 1),
		    1.0f / imgParams.voxelSize[1]);
		const float gz = joseph_grid_coord(
		    z, joseph_half_length(imgParams, 2),
		    1.0f / imgParams.voxelSize[2]);
		const int y0 = static_cast<int>(floorf(gy));
		const int z0 = static_cast<int>(floorf(gz));
		const float fy = gy - static_cast<float>(y0);
		const float fz = gz - static_cast<float>(z0);
		return (1.0f - fy) * (1.0f - fz) *
		           joseph_image_value(image, majorIndex, y0, z0,
		                              imgParams) +
		       fy * (1.0f - fz) *
		           joseph_image_value(image, majorIndex, y0 + 1, z0,
		                              imgParams) +
		       (1.0f - fy) * fz *
		           joseph_image_value(image, majorIndex, y0, z0 + 1,
		                              imgParams) +
		       fy * fz *
		           joseph_image_value(image, majorIndex, y0 + 1, z0 + 1,
		                              imgParams);
	}
	if (axis == 1)
	{
		const float gx = joseph_grid_coord(
		    x, joseph_half_length(imgParams, 0),
		    1.0f / imgParams.voxelSize[0]);
		const float gz = joseph_grid_coord(
		    z, joseph_half_length(imgParams, 2),
		    1.0f / imgParams.voxelSize[2]);
		const int x0 = static_cast<int>(floorf(gx));
		const int z0 = static_cast<int>(floorf(gz));
		const float fx = gx - static_cast<float>(x0);
		const float fz = gz - static_cast<float>(z0);
		return (1.0f - fx) * (1.0f - fz) *
		           joseph_image_value(image, x0, majorIndex, z0,
		                              imgParams) +
		       fx * (1.0f - fz) *
		           joseph_image_value(image, x0 + 1, majorIndex, z0,
		                              imgParams) +
		       (1.0f - fx) * fz *
		           joseph_image_value(image, x0, majorIndex, z0 + 1,
		                              imgParams) +
		       fx * fz *
		           joseph_image_value(image, x0 + 1, majorIndex, z0 + 1,
		                              imgParams);
	}

	const float gx = joseph_grid_coord(
	    x, joseph_half_length(imgParams, 0), 1.0f / imgParams.voxelSize[0]);
	const float gy = joseph_grid_coord(
	    y, joseph_half_length(imgParams, 1), 1.0f / imgParams.voxelSize[1]);
	const int x0 = static_cast<int>(floorf(gx));
	const int y0 = static_cast<int>(floorf(gy));
	const float fx = gx - static_cast<float>(x0);
	const float fy = gy - static_cast<float>(y0);
	return (1.0f - fx) * (1.0f - fy) *
	           joseph_image_value(image, x0, y0, majorIndex, imgParams) +
	       fx * (1.0f - fy) *
	           joseph_image_value(image, x0 + 1, y0, majorIndex, imgParams) +
	       (1.0f - fx) * fy *
	           joseph_image_value(image, x0, y0 + 1, majorIndex, imgParams) +
	       fx * fy *
	           joseph_image_value(image, x0 + 1, y0 + 1, majorIndex,
	                              imgParams);
}

__device__ inline void joseph_add_voxel(float* image, int vx, int vy, int vz,
                                        float update,
                                        const CUImageParams& imgParams)
{
	if (update == 0.0f || vx < 0 || vy < 0 || vz < 0 ||
	    vx >= imgParams.voxelNumber[0] || vy >= imgParams.voxelNumber[1] ||
	    vz >= imgParams.voxelNumber[2])
	{
		return;
	}
	atomicAdd(&image[joseph_image_offset(vx, vy, vz, imgParams)], update);
}

__device__ inline void joseph_bilinear_backproject(
    float* image, int axis, int majorIndex, float alpha, float update,
    const JosephLine& line, const CUImageParams& imgParams)
{
	const float x = line.p1x + alpha * (line.p2x - line.p1x);
	const float y = line.p1y + alpha * (line.p2y - line.p1y);
	const float z = line.p1z + alpha * (line.p2z - line.p1z);
	if (axis == 0)
	{
		const float gy = joseph_grid_coord(
		    y, joseph_half_length(imgParams, 1),
		    1.0f / imgParams.voxelSize[1]);
		const float gz = joseph_grid_coord(
		    z, joseph_half_length(imgParams, 2),
		    1.0f / imgParams.voxelSize[2]);
		const int y0 = static_cast<int>(floorf(gy));
		const int z0 = static_cast<int>(floorf(gz));
		const float fy = gy - static_cast<float>(y0);
		const float fz = gz - static_cast<float>(z0);
		joseph_add_voxel(image, majorIndex, y0, z0,
		                 update * (1.0f - fy) * (1.0f - fz), imgParams);
		joseph_add_voxel(image, majorIndex, y0 + 1, z0,
		                 update * fy * (1.0f - fz), imgParams);
		joseph_add_voxel(image, majorIndex, y0, z0 + 1,
		                 update * (1.0f - fy) * fz, imgParams);
		joseph_add_voxel(image, majorIndex, y0 + 1, z0 + 1,
		                 update * fy * fz, imgParams);
		return;
	}
	if (axis == 1)
	{
		const float gx = joseph_grid_coord(
		    x, joseph_half_length(imgParams, 0),
		    1.0f / imgParams.voxelSize[0]);
		const float gz = joseph_grid_coord(
		    z, joseph_half_length(imgParams, 2),
		    1.0f / imgParams.voxelSize[2]);
		const int x0 = static_cast<int>(floorf(gx));
		const int z0 = static_cast<int>(floorf(gz));
		const float fx = gx - static_cast<float>(x0);
		const float fz = gz - static_cast<float>(z0);
		joseph_add_voxel(image, x0, majorIndex, z0,
		                 update * (1.0f - fx) * (1.0f - fz), imgParams);
		joseph_add_voxel(image, x0 + 1, majorIndex, z0,
		                 update * fx * (1.0f - fz), imgParams);
		joseph_add_voxel(image, x0, majorIndex, z0 + 1,
		                 update * (1.0f - fx) * fz, imgParams);
		joseph_add_voxel(image, x0 + 1, majorIndex, z0 + 1,
		                 update * fx * fz, imgParams);
		return;
	}

	const float gx = joseph_grid_coord(
	    x, joseph_half_length(imgParams, 0), 1.0f / imgParams.voxelSize[0]);
	const float gy = joseph_grid_coord(
	    y, joseph_half_length(imgParams, 1), 1.0f / imgParams.voxelSize[1]);
	const int x0 = static_cast<int>(floorf(gx));
	const int y0 = static_cast<int>(floorf(gy));
	const float fx = gx - static_cast<float>(x0);
	const float fy = gy - static_cast<float>(y0);
	joseph_add_voxel(image, x0, y0, majorIndex,
	                 update * (1.0f - fx) * (1.0f - fy), imgParams);
	joseph_add_voxel(image, x0 + 1, y0, majorIndex,
	                 update * fx * (1.0f - fy), imgParams);
	joseph_add_voxel(image, x0, y0 + 1, majorIndex,
	                 update * (1.0f - fx) * fy, imgParams);
	joseph_add_voxel(image, x0 + 1, y0 + 1, majorIndex,
	                 update * fx * fy, imgParams);
}
}  // namespace

template <bool IsForward, bool HasTOF>
__global__ void OperatorProjectorJosephCU_kernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* /*pd_lorDet1Orient*/,
    const float4* /*pd_lorDet2Orient*/, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper,
    CUScannerParams /*scannerParams*/, CUImageParams imgParams,
    size_t batchSize)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId >= batchSize)
	{
		return;
	}

	const float4 imageOffset =
	    make_float4(imgParams.offset[0], imgParams.offset[1],
	                imgParams.offset[2], 0.0f);
	const float4 p1 = pd_lorDet1Pos[eventId] - imageOffset;
	const float4 p2 = pd_lorDet2Pos[eventId] - imageOffset;
	const JosephLine line{p1.x, p1.y, p1.z, p2.x, p2.y, p2.z};

	const float dx = line.p2x - line.p1x;
	const float dy = line.p2y - line.p1y;
	const float dz = line.p2z - line.p1z;
	const float rayLength = norm3df(dx, dy, dz);
	const float tofValue = HasTOF ? pd_lorTOFValue[eventId] : 0.0f;

	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range<HasTOF>(line, imgParams, rayLength, tofValue,
	                                pd_tofHelper, alphaMin, alphaMax))
	{
		if constexpr (IsForward)
		{
			pd_projValues[eventId] = 0.0f;
		}
		return;
	}

	const int axis = joseph_major_axis(line, imgParams);
	int first;
	int last;
	if (!joseph_sample_bounds(line, imgParams, axis, alphaMin, alphaMax,
	                          first, last))
	{
		if constexpr (IsForward)
		{
			pd_projValues[eventId] = 0.0f;
		}
		return;
	}

	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, imgParams, axis, rayLength);
	if constexpr (IsForward)
	{
		float projection = 0.0f;
		for (int majorIndex = first; majorIndex <= last; majorIndex++)
		{
			const float alpha = joseph_sample_alpha(axisCache, majorIndex);
			const float weight = joseph_sample_weight<HasTOF>(
			    axisCache, alpha, alphaMin, alphaMax, tofValue,
			    pd_tofHelper);
			if (weight == 0.0f)
			{
				continue;
			}

			projection += weight * joseph_bilinear_forward(
			                            pd_image, axis, majorIndex, alpha,
			                            line, imgParams);
		}

		pd_projValues[eventId] = projection;
	}
	else
	{
		const float projectionValue = pd_projValues[eventId];
		if (projectionValue == 0.0f)
		{
			return;
		}
		for (int majorIndex = first; majorIndex <= last; majorIndex++)
		{
			const float alpha = joseph_sample_alpha(axisCache, majorIndex);
			const float weight = joseph_sample_weight<HasTOF>(
			    axisCache, alpha, alphaMin, alphaMax, tofValue,
			    pd_tofHelper);
			if (weight == 0.0f)
			{
				continue;
			}

			joseph_bilinear_backproject(pd_image, axis, majorIndex, alpha,
			                            projectionValue * weight, line,
			                            imgParams);
		}
	}
}

template __global__ void OperatorProjectorJosephCU_kernel<true, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorJosephCU_kernel<false, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorJosephCU_kernel<true, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorJosephCU_kernel<false, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);

}  // namespace yrt
