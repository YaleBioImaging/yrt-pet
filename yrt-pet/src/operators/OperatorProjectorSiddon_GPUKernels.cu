/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/ProjectorUtils.hpp"
#include "operators/OperatorProjectorSiddon_GPUKernels.cuh"
#include "operators/ProjectionPsfManagerDevice.cuh"
#include "operators/ProjectionPsfUtils.cuh"

#include <cuda_runtime.h>


enum SIDDON_DIR
{
	DIR_X = 0b001,
	DIR_Y = 0b010,
	DIR_Z = 0b100
};

template <bool IsForward, bool HasTOF, bool IsIncremental>
__global__ void OperatorProjectorSiddonCU_kernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < batchSize)
	{
		float value = 0.0f;
		if constexpr (!IsForward)
		{
			value = pd_projValues[eventId];
		}

		const float4 p1 = pd_lorDet1Pos[eventId];
		const float4 p2 = pd_lorDet2Pos[eventId];
		const float4 p1_minus_p2 = p1 - p2;
		float tofValue = 0.0f;
		if constexpr (HasTOF)
		{
			tofValue = pd_lorTOFValue[eventId];
		}

		// 1. Intersection with FOV
		float t0;
		float t1;
		// Intersection with (centered) FOV cylinder
		float A = (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y);
		float B = 2.0f * ((p2.x - p1.x) * p1.x + (p2.y - p1.y) * p1.y);
		float C = p1.x * p1.x + p1.y * p1.y -
		          imgParams.fovRadius * imgParams.fovRadius;
		float Delta = B * B - 4 * A * C;
		if (A != 0.0)
		{
			if (Delta <= 0.0)
			{
				t0 = 1.0;
				t1 = 0.0;
				return;
			}
			t0 = (-B - sqrt(Delta)) / (2 * A);
			t1 = (-B + sqrt(Delta)) / (2 * A);
		}
		else
		{
			t0 = 0.0;
			t1 = 1.0;
		}

		float d_norm = norm3df(p1_minus_p2.x, p1_minus_p2.y, p1_minus_p2.z);
		bool flat_x = (p1.x == p2.x);
		bool flat_y = (p1.y == p2.y);
		bool flat_z = (p1.z == p2.z);
		float inv_p12_x = flat_x ? 0.0f : 1.0f / (p2.x - p1.x);
		float inv_p12_y = flat_y ? 0.0f : 1.0f / (p2.y - p1.y);
		float inv_p12_z = flat_z ? 0.0f : 1.0f / (p2.z - p1.z);
		int dir_x = (inv_p12_x >= 0.0) ? 1 : -1;
		int dir_y = (inv_p12_y >= 0.0) ? 1 : -1;
		int dir_z = (inv_p12_z >= 0.0) ? 1 : -1;

		// 2. Intersection with volume
		const int nx = imgParams.voxelNumber[0];
		const int ny = imgParams.voxelNumber[1];
		const int nz = imgParams.voxelNumber[2];
		const float imgLength_x = imgParams.imgLength[0];
		const float imgLength_y = imgParams.imgLength[1];
		const float imgLength_z = imgParams.imgLength[2];
		const int num_xy = nx * ny;
		const float dx = imgParams.voxelSize[0];
		const float dy = imgParams.voxelSize[1];
		const float dz = imgParams.voxelSize[2];
		float inv_dx = 1.0f / dx;
		float inv_dy = 1.0f / dy;
		float inv_dz = 1.0f / dz;

		float x0 = -imgLength_x * 0.5f;
		float x1 = imgLength_x * 0.5f;
		float y0 = -imgLength_y * 0.5f;
		float y1 = imgLength_y * 0.5f;
		float z0 = -imgLength_z * 0.5f;
		float z1 = imgLength_z * 0.5f;
		float ax_min, ax_max, ay_min, ay_max, az_min, az_max;
		Util::get_alpha(-0.5f * imgLength_x, 0.5f * imgLength_x, p1.x, p2.x,
		                inv_p12_x, ax_min, ax_max);
		Util::get_alpha(-0.5f * imgLength_y, 0.5f * imgLength_y, p1.y, p2.y,
		                inv_p12_y, ay_min, ay_max);
		Util::get_alpha(-0.5f * imgLength_z, 0.5f * imgLength_z, p1.z, p2.z,
		                inv_p12_z, az_min, az_max);
		float amin = std::max({0.0f, t0, ax_min, ay_min, az_min});
		float amax = std::min({1.0f, t1, ax_max, ay_max, az_max});
		if (HasTOF)
		{
			float amin_tof, amax_tof;
			pd_tofHelper->getAlphaRange(amin_tof, amax_tof, d_norm, tofValue);
			amin = std::max(amin, amin_tof);
			amax = std::min(amax, amax_tof);
		}

		float a_cur = amin;
		float a_next = -1.0f;
		float x_cur = (inv_p12_x > 0.0f) ? x0 : x1;
		float y_cur = (inv_p12_y > 0.0f) ? y0 : y1;
		float z_cur = (inv_p12_z > 0.0f) ? z0 : z1;
		if ((inv_p12_x >= 0.0f && p1.x > x1) ||
		    (inv_p12_x < 0.0f && p1.x < x0) ||
		    (inv_p12_y >= 0.0f && p1.y > y1) ||
		    (inv_p12_y < 0.0f && p1.y < y0) ||
		    (inv_p12_z >= 0.0f && p1.z > z1) || (inv_p12_z < 0.0f && p1.z < z0))
		{
			return;
		}
		// Move starting point inside FOV
		float ax_next = flat_x ? std::numeric_limits<float>::max() : ax_min;
		if (!flat_x)
		{
			int kx =
			    (int)ceil(dir_x * (a_cur * (p2.x - p1.x) - x_cur + p1.x) / dx);
			x_cur += kx * dir_x * dx;
			ax_next = (x_cur - p1.x) * inv_p12_x;
		}
		float ay_next = flat_y ? std::numeric_limits<float>::max() : ay_min;
		if (!flat_y)
		{
			int ky =
			    (int)ceil(dir_y * (a_cur * (p2.y - p1.y) - y_cur + p1.y) / dy);
			y_cur += ky * dir_y * dy;
			ay_next = (y_cur - p1.y) * inv_p12_y;
		}
		float az_next = flat_z ? std::numeric_limits<float>::max() : az_min;
		if (!flat_z)
		{
			int kz =
			    (int)ceil(dir_z * (a_cur * (p2.z - p1.z) - z_cur + p1.z) / dz);
			z_cur += kz * dir_z * dz;
			az_next = (z_cur - p1.z) * inv_p12_z;
		}
		// Pixel location (move pixel to pixel instead of calculating position
		// for each intersection)
		bool flag_first = true;
		int vx = -1;
		int vy = -1;
		int vz = -1;
		// The dir variables operate as binary bit-flags to determine in which
		// direction the current pixel should move: format 0bzyx (where z, y and
		// x are bits set to 1 when the pixel should move in the corresponding
		// direction, e.g. 0b101 moves in the z and x directions)
		short dir_prev = -1;
		short dir_next = -1;

		// Prepare data pointer (this assumes that the data is stored as a
		// contiguous array)
		float* raw_img_ptr = pd_image;
		float* cur_img_ptr = nullptr;

		float ax_next_prev = ax_next;
		float ay_next_prev = ay_next;
		float az_next_prev = az_next;

		// 3. Integrate along ray
		bool flag_done = false;
		while (a_cur < amax && !flag_done)
		{
			// Find next intersection (along x, y or z)
			dir_next = 0b000;
			if (ax_next_prev <= ay_next_prev && ax_next_prev <= az_next_prev)
			{
				a_next = ax_next;
				x_cur += dir_x * dx;
				ax_next = (x_cur - p1.x) * inv_p12_x;
				dir_next |= SIDDON_DIR::DIR_X;
			}
			if (ay_next_prev <= ax_next_prev && ay_next_prev <= az_next_prev)
			{
				a_next = ay_next;
				y_cur += dir_y * dy;
				ay_next = (y_cur - p1.y) * inv_p12_y;
				dir_next |= SIDDON_DIR::DIR_Y;
			}
			if (az_next_prev <= ax_next_prev && az_next_prev <= ay_next_prev)
			{
				a_next = az_next;
				z_cur += dir_z * dz;
				az_next = (z_cur - p1.z) * inv_p12_z;
				dir_next |= SIDDON_DIR::DIR_Z;
			}
			// Clip to FOV range
			if (a_next > amax)
			{
				a_next = amax;
			}
			if (a_cur >= a_next)
			{
				ax_next_prev = ax_next;
				ay_next_prev = ay_next;
				az_next_prev = az_next;
				continue;
			}
			// Determine pixel location
			float tof_weight = 1.f;
			float a_mid = 0.5f * (a_cur + a_next);
			if constexpr (HasTOF)
			{
				tof_weight = pd_tofHelper->getWeight(
				    d_norm, tofValue, a_cur * d_norm, a_next * d_norm);
			}
			if (!IsIncremental || flag_first)
			{
				vx = (int)((p1.x + a_mid * (p2.x - p1.x) + imgLength_x / 2) *
				           inv_dx);
				vy = (int)((p1.y + a_mid * (p2.y - p1.y) + imgLength_y / 2) *
				           inv_dy);
				vz = (int)((p1.z + a_mid * (p2.z - p1.z) + imgLength_z / 2) *
				           inv_dz);
				cur_img_ptr = raw_img_ptr + vz * num_xy + vy * nx;
				flag_first = false;
				if (vx < 0 || vx >= nx || vy < 0 || vy >= ny || vz < 0 ||
				    vz >= nz)
				{
					flag_done = true;
				}
			}
			else
			{
				if (dir_prev & SIDDON_DIR::DIR_X)
				{
					vx += dir_x;
					if (vx < 0 || vx >= nx)
					{
						flag_done = true;
					}
				}
				if (dir_prev & SIDDON_DIR::DIR_Y)
				{
					vy += dir_y;
					if (vy < 0 || vy >= ny)
					{
						flag_done = true;
					}
					else
					{
						cur_img_ptr += dir_y * nx;
					}
				}
				if (dir_prev & SIDDON_DIR::DIR_Z)
				{
					vz += dir_z;
					if (vz < 0 || vz >= nz)
					{
						flag_done = true;
					}
					else
					{
						cur_img_ptr += dir_z * num_xy;
					}
				}
			}
			if (flag_done)
			{
				continue;
			}
			dir_prev = dir_next;
			float weight = (a_next - a_cur) * d_norm;
			if constexpr (HasTOF)
			{
				weight *= tof_weight;
			}
			if constexpr (IsForward)
			{
				value += weight * cur_img_ptr[vx];
			}
			else
			{
				float output = value * weight;
				float* ptr = &cur_img_ptr[vx];
#pragma omp atomic
				*ptr += output;
			}
			a_cur = a_next;
			ax_next_prev = ax_next;
			ay_next_prev = ay_next;
			az_next_prev = az_next;
		}

		pd_projValues[eventId] = value;
	}
}

template __global__ void OperatorProjectorSiddonCU_kernel<true, true, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorSiddonCU_kernel<false, true, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorSiddonCU_kernel<true, false, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorSiddonCU_kernel<false, false, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorSiddonCU_kernel<true, true, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorSiddonCU_kernel<false, true, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorSiddonCU_kernel<true, false, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorSiddonCU_kernel<false, false, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize);
