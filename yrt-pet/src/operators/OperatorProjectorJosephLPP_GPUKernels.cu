/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/geometry/ProjectorUtils.hpp"
#include "yrt-pet/operators/OperatorProjectorJosephLPP_GPUKernels.cuh"

#include <cuda_runtime.h>

namespace yrt
{
namespace
{
// This variant keeps YRT's LOR buffers, image layout, FOV clipping, and
// TimeOfFlightHelper, while using libparallelproj-style Joseph stepping:
// one sample per principal-axis voxel plane with a constant cf correction.
struct JosephLPPLine
{
	float p1x;
	float p1y;
	float p1z;
	float p2x;
	float p2y;
	float p2z;
};

__device__ inline int lpp_image_offset(int vx, int vy, int vz,
                                       const CUImageParams& imgParams)
{
	const int nx = imgParams.voxelNumber[0];
	const int ny = imgParams.voxelNumber[1];
	return vx + nx * (vy + ny * vz);
}

__device__ inline float lpp_half_length(const CUImageParams& imgParams,
                                        int axis)
{
	return 0.5f * imgParams.imgLength[axis];
}

__device__ inline float lpp_voxel0_center(const CUImageParams& imgParams,
                                          int axis)
{
	return -lpp_half_length(imgParams, axis) + 0.5f * imgParams.voxelSize[axis];
}

__device__ inline float lpp_axis_start(const JosephLPPLine& line, int axis)
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

__device__ inline float lpp_axis_delta(const JosephLPPLine& line, int axis)
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

__device__ inline float lpp_axis_coord(const JosephLPPLine& line, int axis,
                                       float alpha)
{
	return lpp_axis_start(line, axis) + alpha * lpp_axis_delta(line, axis);
}

__device__ inline int lpp_axis_size(const CUImageParams& imgParams, int axis)
{
	return imgParams.voxelNumber[axis];
}

template <bool HasTOF>
__device__ inline bool lpp_alpha_range(
    const JosephLPPLine& line, const CUImageParams& imgParams, float rayLength,
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
	util::get_alpha(-lpp_half_length(imgParams, 0),
	                lpp_half_length(imgParams, 0), line.p1x, line.p2x, invX,
	                axMin, axMax);
	util::get_alpha(-lpp_half_length(imgParams, 1),
	                lpp_half_length(imgParams, 1), line.p1y, line.p2y, invY,
	                ayMin, ayMax);
	util::get_alpha(-lpp_half_length(imgParams, 2),
	                lpp_half_length(imgParams, 2), line.p1z, line.p2z, invZ,
	                azMin, azMax);

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

__device__ inline int lpp_major_axis(const JosephLPPLine& line)
{
	const float dx = fabsf(line.p2x - line.p1x);
	const float dy = fabsf(line.p2y - line.p1y);
	const float dz = fabsf(line.p2z - line.p1z);
	if (dy >= dx && dy >= dz)
	{
		return 1;
	}
	if (dz >= dx && dz >= dy)
	{
		return 2;
	}
	return 0;
}

__device__ inline bool lpp_plane_range(
    const JosephLPPLine& line, const CUImageParams& imgParams, int axis,
    float alphaMin, float alphaMax, float rayLength, int& first, int& last,
    float& correction)
{
	const float dAxis = lpp_axis_delta(line, axis);
	if (dAxis == 0.0f)
	{
		return false;
	}

	const float voxel = imgParams.voxelSize[axis];
	const float origin = lpp_voxel0_center(imgParams, axis);
	float f0 = (lpp_axis_coord(line, axis, alphaMin) - origin) / voxel;
	float f1 = (lpp_axis_coord(line, axis, alphaMax) - origin) / voxel;
	if (f0 > f1)
	{
		const float tmp = f0;
		f0 = f1;
		f1 = tmp;
	}

	first = static_cast<int>(floorf(f0)) + 1;
	last = static_cast<int>(floorf(f1));
	if (first < 0)
	{
		first = 0;
	}
	const int maxIndex = lpp_axis_size(imgParams, axis) - 1;
	if (last > maxIndex)
	{
		last = maxIndex;
	}

	correction = rayLength * voxel / fabsf(dAxis);
	return first <= last;
}

__device__ inline float lpp_transverse_slope(const JosephLPPLine& line,
                                             const CUImageParams& imgParams,
                                             int axis, int transverseAxis)
{
	return lpp_axis_delta(line, transverseAxis) * imgParams.voxelSize[axis] /
	       (imgParams.voxelSize[transverseAxis] *
	        lpp_axis_delta(line, axis));
}

__device__ inline float lpp_transverse_intercept(
    const JosephLPPLine& line, const CUImageParams& imgParams, int axis,
    int transverseAxis)
{
	const float dAxis = lpp_axis_delta(line, axis);
	const float axisOrigin = lpp_voxel0_center(imgParams, axis);
	const float transverseOrigin =
	    lpp_voxel0_center(imgParams, transverseAxis);
	return (lpp_axis_start(line, transverseAxis) - transverseOrigin +
	        lpp_axis_delta(line, transverseAxis) *
	            (axisOrigin - lpp_axis_start(line, axis)) / dAxis) /
	       imgParams.voxelSize[transverseAxis];
}

__device__ inline float lpp_image_value(const float* image, int vx, int vy,
                                        int vz,
                                        const CUImageParams& imgParams)
{
	if (vx < 0 || vy < 0 || vz < 0 ||
	    vx >= imgParams.voxelNumber[0] || vy >= imgParams.voxelNumber[1] ||
	    vz >= imgParams.voxelNumber[2])
	{
		return 0.0f;
	}
	return image[lpp_image_offset(vx, vy, vz, imgParams)];
}

__device__ inline float lpp_bilinear_forward(
    const float* image, int axis, int majorIndex, float t0, float t1,
    const CUImageParams& imgParams)
{
	const int i0 = static_cast<int>(floorf(t0));
	const int i1 = static_cast<int>(floorf(t1));
	const float f0 = t0 - static_cast<float>(i0);
	const float f1 = t1 - static_cast<float>(i1);
	if (axis == 0)
	{
		return (1.0f - f0) * (1.0f - f1) *
		           lpp_image_value(image, majorIndex, i0, i1, imgParams) +
		       f0 * (1.0f - f1) *
		           lpp_image_value(image, majorIndex, i0 + 1, i1,
		                           imgParams) +
		       (1.0f - f0) * f1 *
		           lpp_image_value(image, majorIndex, i0, i1 + 1,
		                           imgParams) +
		       f0 * f1 *
		           lpp_image_value(image, majorIndex, i0 + 1, i1 + 1,
		                           imgParams);
	}
	if (axis == 1)
	{
		return (1.0f - f0) * (1.0f - f1) *
		           lpp_image_value(image, i0, majorIndex, i1, imgParams) +
		       f0 * (1.0f - f1) *
		           lpp_image_value(image, i0 + 1, majorIndex, i1,
		                           imgParams) +
		       (1.0f - f0) * f1 *
		           lpp_image_value(image, i0, majorIndex, i1 + 1,
		                           imgParams) +
		       f0 * f1 *
		           lpp_image_value(image, i0 + 1, majorIndex, i1 + 1,
		                           imgParams);
	}
	return (1.0f - f0) * (1.0f - f1) *
	           lpp_image_value(image, i0, i1, majorIndex, imgParams) +
	       f0 * (1.0f - f1) *
	           lpp_image_value(image, i0 + 1, i1, majorIndex, imgParams) +
	       (1.0f - f0) * f1 *
	           lpp_image_value(image, i0, i1 + 1, majorIndex, imgParams) +
	       f0 * f1 *
	           lpp_image_value(image, i0 + 1, i1 + 1, majorIndex,
	                           imgParams);
}

__device__ inline void lpp_add_voxel(float* image, int vx, int vy, int vz,
                                     float update,
                                     const CUImageParams& imgParams)
{
	if (update == 0.0f || vx < 0 || vy < 0 || vz < 0 ||
	    vx >= imgParams.voxelNumber[0] || vy >= imgParams.voxelNumber[1] ||
	    vz >= imgParams.voxelNumber[2])
	{
		return;
	}
	atomicAdd(&image[lpp_image_offset(vx, vy, vz, imgParams)], update);
}

__device__ inline void lpp_bilinear_backproject(
    float* image, int axis, int majorIndex, float t0, float t1, float update,
    const CUImageParams& imgParams)
{
	const int i0 = static_cast<int>(floorf(t0));
	const int i1 = static_cast<int>(floorf(t1));
	const float f0 = t0 - static_cast<float>(i0);
	const float f1 = t1 - static_cast<float>(i1);
	if (axis == 0)
	{
		lpp_add_voxel(image, majorIndex, i0, i1,
		              update * (1.0f - f0) * (1.0f - f1), imgParams);
		lpp_add_voxel(image, majorIndex, i0 + 1, i1,
		              update * f0 * (1.0f - f1), imgParams);
		lpp_add_voxel(image, majorIndex, i0, i1 + 1,
		              update * (1.0f - f0) * f1, imgParams);
		lpp_add_voxel(image, majorIndex, i0 + 1, i1 + 1,
		              update * f0 * f1, imgParams);
		return;
	}
	if (axis == 1)
	{
		lpp_add_voxel(image, i0, majorIndex, i1,
		              update * (1.0f - f0) * (1.0f - f1), imgParams);
		lpp_add_voxel(image, i0 + 1, majorIndex, i1,
		              update * f0 * (1.0f - f1), imgParams);
		lpp_add_voxel(image, i0, majorIndex, i1 + 1,
		              update * (1.0f - f0) * f1, imgParams);
		lpp_add_voxel(image, i0 + 1, majorIndex, i1 + 1,
		              update * f0 * f1, imgParams);
		return;
	}
	lpp_add_voxel(image, i0, i1, majorIndex,
	              update * (1.0f - f0) * (1.0f - f1), imgParams);
	lpp_add_voxel(image, i0 + 1, i1, majorIndex,
	              update * f0 * (1.0f - f1), imgParams);
	lpp_add_voxel(image, i0, i1 + 1, majorIndex,
	              update * (1.0f - f0) * f1, imgParams);
	lpp_add_voxel(image, i0 + 1, i1 + 1, majorIndex, update * f0 * f1,
	              imgParams);
}

template <bool HasTOF>
__device__ inline float lpp_tof_weight(
    const JosephLPPLine& line, int axis, int majorIndex,
    const CUImageParams& imgParams, float rayLength, float correction,
    float tofValue, const TimeOfFlightHelper* tofHelper)
{
	if constexpr (!HasTOF)
	{
		return correction;
	}
	else
	{
		const float axisCoord =
		    lpp_voxel0_center(imgParams, axis) +
		    static_cast<float>(majorIndex) * imgParams.voxelSize[axis];
		const float alpha =
		    (axisCoord - lpp_axis_start(line, axis)) /
		    lpp_axis_delta(line, axis);
		const float offset = alpha * rayLength;
		return correction *
		       tofHelper->getWeight(rayLength, tofValue, offset, offset);
	}
}
}  // namespace

template <bool IsForward, bool HasTOF>
__global__ void OperatorProjectorJosephLPPCU_kernel(
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
	const JosephLPPLine line{p1.x, p1.y, p1.z, p2.x, p2.y, p2.z};

	const float dx = line.p2x - line.p1x;
	const float dy = line.p2y - line.p1y;
	const float dz = line.p2z - line.p1z;
	const float rayLength = norm3df(dx, dy, dz);
	const float tofValue = HasTOF ? pd_lorTOFValue[eventId] : 0.0f;

	float alphaMin;
	float alphaMax;
	if (!lpp_alpha_range<HasTOF>(line, imgParams, rayLength, tofValue,
	                             pd_tofHelper, alphaMin, alphaMax))
	{
		if constexpr (IsForward)
		{
			pd_projValues[eventId] = 0.0f;
		}
		return;
	}

	const int axis = lpp_major_axis(line);
	int first;
	int last;
	float correction;
	if (!lpp_plane_range(line, imgParams, axis, alphaMin, alphaMax, rayLength,
	                     first, last, correction))
	{
		if constexpr (IsForward)
		{
			pd_projValues[eventId] = 0.0f;
		}
		return;
	}

	int tAxis0 = 0;
	int tAxis1 = 1;
	if (axis == 0)
	{
		tAxis0 = 1;
		tAxis1 = 2;
	}
	else if (axis == 1)
	{
		tAxis0 = 0;
		tAxis1 = 2;
	}

	const float slope0 =
	    lpp_transverse_slope(line, imgParams, axis, tAxis0);
	const float slope1 =
	    lpp_transverse_slope(line, imgParams, axis, tAxis1);
	float coord0 =
	    static_cast<float>(first) * slope0 +
	    lpp_transverse_intercept(line, imgParams, axis, tAxis0);
	float coord1 =
	    static_cast<float>(first) * slope1 +
	    lpp_transverse_intercept(line, imgParams, axis, tAxis1);

	if constexpr (IsForward)
	{
		float projection = 0.0f;
		for (int majorIndex = first; majorIndex <= last; majorIndex++)
		{
			const float weight = lpp_tof_weight<HasTOF>(
			    line, axis, majorIndex, imgParams, rayLength, correction,
			    tofValue, pd_tofHelper);
			if (weight != 0.0f)
			{
				projection +=
				    weight * lpp_bilinear_forward(pd_image, axis, majorIndex,
				                                  coord0, coord1, imgParams);
			}
			coord0 += slope0;
			coord1 += slope1;
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
			const float weight = lpp_tof_weight<HasTOF>(
			    line, axis, majorIndex, imgParams, rayLength, correction,
			    tofValue, pd_tofHelper);
			if (weight != 0.0f)
			{
				lpp_bilinear_backproject(pd_image, axis, majorIndex, coord0,
				                         coord1, projectionValue * weight,
				                         imgParams);
			}
			coord0 += slope0;
			coord1 += slope1;
		}
	}
}

template __global__ void OperatorProjectorJosephLPPCU_kernel<true, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorJosephLPPCU_kernel<false, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorJosephLPPCU_kernel<true, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorJosephLPPCU_kernel<false, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);

}  // namespace yrt
