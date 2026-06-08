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

struct JosephTraceCache
{
	float transverse0;
	float transverse1;
	float transverseStep0;
	float transverseStep1;
	float interiorWeight;
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

template <int Axis>
__device__ inline JosephTraceCache
    joseph_make_trace_cache(const JosephLine& line,
                            const CUImageParams& imgParams,
                            const JosephAxisCache& axisCache, int first)
{
	constexpr int axis0 = Axis == 0 ? 1 : 0;
	constexpr int axis1 = Axis == 2 ? 1 : 2;

	const float alphaFirst = joseph_sample_alpha(axisCache, first);
	const float alphaStep = axisCache.voxel * axisCache.invDelta;

	JosephTraceCache cache;
	cache.transverse0 = joseph_grid_coord(
	    joseph_axis_coord(line, axis0, alphaFirst),
	    joseph_half_length(imgParams, axis0),
	    1.0f / imgParams.voxelSize[axis0]);
	cache.transverse1 = joseph_grid_coord(
	    joseph_axis_coord(line, axis1, alphaFirst),
	    joseph_half_length(imgParams, axis1),
	    1.0f / imgParams.voxelSize[axis1]);
	cache.transverseStep0 =
	    joseph_axis_delta(line, axis0) * alphaStep /
	    imgParams.voxelSize[axis0];
	cache.transverseStep1 =
	    joseph_axis_delta(line, axis1) * alphaStep /
	    imgParams.voxelSize[axis1];
	cache.interiorWeight = axisCache.rayLength *
	                       (2.0f * axisCache.halfAlphaStep);
	return cache;
}

template <bool HasTOF>
__device__ inline float joseph_sample_weight_fast(
    const JosephAxisCache& cache, int majorIndex, int first, int last,
    float alphaMin, float alphaMax, float tofValue,
    const TimeOfFlightHelper* tofHelper, float interiorWeight)
{
	if constexpr (!HasTOF)
	{
		if (majorIndex != first && majorIndex != last)
		{
			return interiorWeight;
		}
	}
	else
	{
		(void)first;
		(void)last;
		(void)interiorWeight;
	}

	const float alpha = joseph_sample_alpha(cache, majorIndex);
	return joseph_sample_weight<HasTOF>(cache, alpha, alphaMin, alphaMax,
	                                    tofValue, tofHelper);
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

template <int Axis>
__device__ inline float joseph_bilinear_forward(
    const float* image, int majorIndex, float transverse0, float transverse1,
    const CUImageParams& imgParams)
{
	const int i0 = static_cast<int>(floorf(transverse0));
	const int i1 = static_cast<int>(floorf(transverse1));
	const float f0 = transverse0 - static_cast<float>(i0);
	const float f1 = transverse1 - static_cast<float>(i1);
	const float w00 = (1.0f - f0) * (1.0f - f1);
	const float w10 = f0 * (1.0f - f1);
	const float w01 = (1.0f - f0) * f1;
	const float w11 = f0 * f1;
	const int nx = imgParams.voxelNumber[0];
	const int ny = imgParams.voxelNumber[1];
	const int nz = imgParams.voxelNumber[2];

	if constexpr (Axis == 0)
	{
		if (majorIndex >= 0 && majorIndex < nx && i0 >= 0 &&
		    i0 + 1 < ny && i1 >= 0 && i1 + 1 < nz)
		{
			const int planeStride = nx * ny;
			const int o00 = majorIndex + nx * (i0 + ny * i1);
			const int o10 = o00 + nx;
			const int o01 = o00 + planeStride;
			const int o11 = o01 + nx;
			return w00 * image[o00] + w10 * image[o10] +
			       w01 * image[o01] + w11 * image[o11];
		}
		return w00 *
		           joseph_image_value(image, majorIndex, i0, i1,
		                              imgParams) +
		       w10 *
		           joseph_image_value(image, majorIndex, i0 + 1, i1,
		                              imgParams) +
		       w01 *
		           joseph_image_value(image, majorIndex, i0, i1 + 1,
		                              imgParams) +
		       w11 *
		           joseph_image_value(image, majorIndex, i0 + 1, i1 + 1,
		                              imgParams);
	}
	if constexpr (Axis == 1)
	{
		if (i0 >= 0 && i0 + 1 < nx && majorIndex >= 0 &&
		    majorIndex < ny && i1 >= 0 && i1 + 1 < nz)
		{
			const int planeStride = nx * ny;
			const int o00 = i0 + nx * (majorIndex + ny * i1);
			const int o10 = o00 + 1;
			const int o01 = o00 + planeStride;
			const int o11 = o01 + 1;
			return w00 * image[o00] + w10 * image[o10] +
			       w01 * image[o01] + w11 * image[o11];
		}
		return w00 *
		           joseph_image_value(image, i0, majorIndex, i1,
		                              imgParams) +
		       w10 *
		           joseph_image_value(image, i0 + 1, majorIndex, i1,
		                              imgParams) +
		       w01 *
		           joseph_image_value(image, i0, majorIndex, i1 + 1,
		                              imgParams) +
		       w11 *
		           joseph_image_value(image, i0 + 1, majorIndex, i1 + 1,
		                              imgParams);
	}

	if constexpr (Axis == 2)
	{
		if (i0 >= 0 && i0 + 1 < nx && i1 >= 0 && i1 + 1 < ny &&
		    majorIndex >= 0 && majorIndex < nz)
		{
			const int o00 = i0 + nx * (i1 + ny * majorIndex);
			const int o10 = o00 + 1;
			const int o01 = o00 + nx;
			const int o11 = o01 + 1;
			return w00 * image[o00] + w10 * image[o10] +
			       w01 * image[o01] + w11 * image[o11];
		}
		return w00 *
		           joseph_image_value(image, i0, i1, majorIndex,
		                              imgParams) +
		       w10 *
		           joseph_image_value(image, i0 + 1, i1, majorIndex,
		                              imgParams) +
		       w01 *
		           joseph_image_value(image, i0, i1 + 1, majorIndex,
		                              imgParams) +
		       w11 *
		           joseph_image_value(image, i0 + 1, i1 + 1, majorIndex,
		                              imgParams);
	}
	return 0.0f;
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

__device__ inline void joseph_add_voxel_in_bounds(float* image, int offset,
                                                  float update)
{
	if (update == 0.0f)
	{
		return;
	}
	atomicAdd(&image[offset], update);
}

template <int Axis>
__device__ inline void joseph_bilinear_backproject(
    float* image, int majorIndex, float transverse0, float transverse1,
    float update, const CUImageParams& imgParams)
{
	if (update == 0.0f)
	{
		return;
	}

	const int i0 = static_cast<int>(floorf(transverse0));
	const int i1 = static_cast<int>(floorf(transverse1));
	const float f0 = transverse0 - static_cast<float>(i0);
	const float f1 = transverse1 - static_cast<float>(i1);
	const float w00 = update * (1.0f - f0) * (1.0f - f1);
	const float w10 = update * f0 * (1.0f - f1);
	const float w01 = update * (1.0f - f0) * f1;
	const float w11 = update * f0 * f1;
	const int nx = imgParams.voxelNumber[0];
	const int ny = imgParams.voxelNumber[1];
	const int nz = imgParams.voxelNumber[2];

	if constexpr (Axis == 0)
	{
		if (majorIndex >= 0 && majorIndex < nx && i0 >= 0 &&
		    i0 + 1 < ny && i1 >= 0 && i1 + 1 < nz)
		{
			const int planeStride = nx * ny;
			const int o00 = majorIndex + nx * (i0 + ny * i1);
			const int o10 = o00 + nx;
			const int o01 = o00 + planeStride;
			const int o11 = o01 + nx;
			joseph_add_voxel_in_bounds(image, o00, w00);
			joseph_add_voxel_in_bounds(image, o10, w10);
			joseph_add_voxel_in_bounds(image, o01, w01);
			joseph_add_voxel_in_bounds(image, o11, w11);
			return;
		}
		joseph_add_voxel(image, majorIndex, i0, i1, w00, imgParams);
		joseph_add_voxel(image, majorIndex, i0 + 1, i1, w10, imgParams);
		joseph_add_voxel(image, majorIndex, i0, i1 + 1, w01, imgParams);
		joseph_add_voxel(image, majorIndex, i0 + 1, i1 + 1, w11,
		                 imgParams);
		return;
	}
	if constexpr (Axis == 1)
	{
		if (i0 >= 0 && i0 + 1 < nx && majorIndex >= 0 &&
		    majorIndex < ny && i1 >= 0 && i1 + 1 < nz)
		{
			const int planeStride = nx * ny;
			const int o00 = i0 + nx * (majorIndex + ny * i1);
			const int o10 = o00 + 1;
			const int o01 = o00 + planeStride;
			const int o11 = o01 + 1;
			joseph_add_voxel_in_bounds(image, o00, w00);
			joseph_add_voxel_in_bounds(image, o10, w10);
			joseph_add_voxel_in_bounds(image, o01, w01);
			joseph_add_voxel_in_bounds(image, o11, w11);
			return;
		}
		joseph_add_voxel(image, i0, majorIndex, i1, w00, imgParams);
		joseph_add_voxel(image, i0 + 1, majorIndex, i1, w10, imgParams);
		joseph_add_voxel(image, i0, majorIndex, i1 + 1, w01, imgParams);
		joseph_add_voxel(image, i0 + 1, majorIndex, i1 + 1, w11,
		                 imgParams);
		return;
	}

	if constexpr (Axis == 2)
	{
		if (i0 >= 0 && i0 + 1 < nx && i1 >= 0 && i1 + 1 < ny &&
		    majorIndex >= 0 && majorIndex < nz)
		{
			const int o00 = i0 + nx * (i1 + ny * majorIndex);
			const int o10 = o00 + 1;
			const int o01 = o00 + nx;
			const int o11 = o01 + 1;
			joseph_add_voxel_in_bounds(image, o00, w00);
			joseph_add_voxel_in_bounds(image, o10, w10);
			joseph_add_voxel_in_bounds(image, o01, w01);
			joseph_add_voxel_in_bounds(image, o11, w11);
			return;
		}
		joseph_add_voxel(image, i0, i1, majorIndex, w00, imgParams);
		joseph_add_voxel(image, i0 + 1, i1, majorIndex, w10, imgParams);
		joseph_add_voxel(image, i0, i1 + 1, majorIndex, w01, imgParams);
		joseph_add_voxel(image, i0 + 1, i1 + 1, majorIndex, w11,
		                 imgParams);
	}
}

template <bool IsForward, bool HasTOF, int Axis>
__device__ inline void joseph_project_axis(
    float* pd_projValues, float* pd_image, long eventId,
    const JosephLine& line, const CUImageParams& imgParams, float rayLength,
    float tofValue, const TimeOfFlightHelper* pd_tofHelper, float alphaMin,
    float alphaMax)
{
	int first;
	int last;
	if (!joseph_sample_bounds(line, imgParams, Axis, alphaMin, alphaMax,
	                          first, last))
	{
		if constexpr (IsForward)
		{
			pd_projValues[eventId] = 0.0f;
		}
		return;
	}

	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, imgParams, Axis, rayLength);
	const JosephTraceCache traceCache =
	    joseph_make_trace_cache<Axis>(line, imgParams, axisCache, first);
	if constexpr (IsForward)
	{
		float projection = 0.0f;
		float transverse0 = traceCache.transverse0;
		float transverse1 = traceCache.transverse1;
		for (int majorIndex = first; majorIndex <= last; majorIndex++)
		{
			const float weight = joseph_sample_weight_fast<HasTOF>(
			    axisCache, majorIndex, first, last, alphaMin, alphaMax,
			    tofValue, pd_tofHelper, traceCache.interiorWeight);
			if (weight != 0.0f)
			{
				projection += weight * joseph_bilinear_forward<Axis>(
				                            pd_image, majorIndex,
				                            transverse0, transverse1,
				                            imgParams);
			}
			transverse0 += traceCache.transverseStep0;
			transverse1 += traceCache.transverseStep1;
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
		float transverse0 = traceCache.transverse0;
		float transverse1 = traceCache.transverse1;
		for (int majorIndex = first; majorIndex <= last; majorIndex++)
		{
			const float weight = joseph_sample_weight_fast<HasTOF>(
			    axisCache, majorIndex, first, last, alphaMin, alphaMax,
			    tofValue, pd_tofHelper, traceCache.interiorWeight);
			if (weight != 0.0f)
			{
				joseph_bilinear_backproject<Axis>(
				    pd_image, majorIndex, transverse0, transverse1,
				    projectionValue * weight, imgParams);
			}
			transverse0 += traceCache.transverseStep0;
			transverse1 += traceCache.transverseStep1;
		}
	}
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
	if (axis == 0)
	{
		joseph_project_axis<IsForward, HasTOF, 0>(
		    pd_projValues, pd_image, eventId, line, imgParams, rayLength,
		    tofValue, pd_tofHelper, alphaMin, alphaMax);
		return;
	}
	if (axis == 1)
	{
		joseph_project_axis<IsForward, HasTOF, 1>(
		    pd_projValues, pd_image, eventId, line, imgParams, rayLength,
		    tofValue, pd_tofHelper, alphaMin, alphaMax);
		return;
	}
	joseph_project_axis<IsForward, HasTOF, 2>(
	    pd_projValues, pd_image, eventId, line, imgParams, rayLength,
	    tofValue, pd_tofHelper, alphaMin, alphaMax);
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
