/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/ProjectorUtils.hpp"
#include "operators/OperatorProjectorDD_GPUKernels.cuh"
#include "operators/ProjectionPsfManagerDevice.cuh"
#include "operators/ProjectionPsfUtils.cuh"

#include <cuda_runtime.h>

__global__ void gatherLORs_kernel(const uint2* pd_lorDetsIds,
                                  const float4* pd_detsPos,
                                  const float4* pd_detsOrient,
                                  float4* pd_lorDet1Pos, float4* pd_lorDet2Pos,
                                  float4* pd_lorDet1Orient,
                                  float4* pd_lorDet2Orient,
                                  CUImageParams imgParams, size_t batchSize)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < batchSize)
	{
		const auto offset_val = imgParams.offset;
		const float4 offset =
		    make_float4(offset_val[0], offset_val[1], offset_val[2], 0.0);

		const uint2 lorDetsId = pd_lorDetsIds[eventId];
		const uint lorDet1 = lorDetsId.x;
		const uint lorDet2 = lorDetsId.y;
		float4 p1 = pd_detsPos[lorDet1];
		p1 = p1 - offset;
		float4 p2 = pd_detsPos[lorDet2];
		p2 = p2 - offset;

		pd_lorDet1Pos[eventId] = p1;
		pd_lorDet2Pos[eventId] = p2;
		pd_lorDet1Orient[eventId] = pd_detsOrient[lorDet1];
		pd_lorDet2Orient[eventId] = pd_detsOrient[lorDet2];
	}
}

__device__ inline float get_overlap_safe(const float p0, const float p1,
                                         const float d0, const float d1)
{
	return min(p1, d1) - max(p0, d0);
}

__device__ inline float
    get_overlap_safe(const float p0, const float p1, const float d0,
                     const float d1, const float* psfKernel,
                     const ProjectionPsfProperties& projectionPsfProperties)
{
	return Util::getWeight(psfKernel, projectionPsfProperties, p0 - d1,
	                       p1 - d0);
}

template <bool IsForward, bool HasTOF, bool HasProjPSF>
__global__ void OperatorProjectorDDCU_kernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < batchSize)
	{
		float value = 0.0f;
		if constexpr (!IsForward)
		{
			// Initialize value at proj-space value if backward
			value = pd_projValues[eventId];
		}

		float tofValue;
		if constexpr (HasTOF)
		{
			tofValue = pd_lorTOFValue[eventId];
		}

		float4 d1 = pd_lorDet1Pos[eventId];
		float4 d2 = pd_lorDet2Pos[eventId];

		float4 imageOffset =
		    make_float4(imgParams.offset[0], imgParams.offset[1],
		                imgParams.offset[2], 0.0f);

		d1 -= imageOffset;
		d2 -= imageOffset;

		const float4 n1 = pd_lorDet1Orient[eventId];
		const float4 n2 = pd_lorDet2Orient[eventId];
		const float4 d1_minus_d2 = d1 - d2;
		const float4 d2_minus_d1 = d1_minus_d2 * (-1.0f);

		// ----------------------- Compute TOR
		const float thickness_z = scannerParams.crystalSize_z;
		const float thickness_trans = scannerParams.crystalSize_trans;

		const bool flag_y = fabs(d2_minus_d1.y) > fabs(d2_minus_d1.x);
		const float d_norm =
		    norm3df(d1_minus_d2.x, d1_minus_d2.y, d1_minus_d2.z);

		const float* psfKernel = nullptr;
		float detFootprintExt = 0.f;
		if constexpr (HasProjPSF)
		{
			psfKernel =
			    Util::getKernel(pd_projPsfKernels, projectionPsfProperties,
			                    d1.x, d1.y, d2.x, d2.y);
			detFootprintExt = projectionPsfProperties.halfWidth;
		}

		// ----------------------- Compute Pixel limits
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

		const float inv_d12_x = (d1.x == d2.x) ? 0.0f : 1.0f / (d2.x - d1.x);
		const float inv_d12_y = (d1.y == d2.y) ? 0.0f : 1.0f / (d2.y - d1.y);
		const float inv_d12_z = (d1.z == d2.z) ? 0.0f : 1.0f / (d2.z - d1.z);

		float ax_min, ax_max, ay_min, ay_max, az_min, az_max;
		Util::get_alpha(-0.5f * (imgLength_x - dx), 0.5f * (imgLength_x - dx),
		                d1.x, d2.x, inv_d12_x, ax_min, ax_max);
		Util::get_alpha(-0.5f * (imgLength_y - dy), 0.5f * (imgLength_y - dy),
		                d1.y, d2.y, inv_d12_y, ay_min, ay_max);
		Util::get_alpha(-0.5f * (imgLength_z - dz), 0.5f * (imgLength_z - dz),
		                d1.z, d2.z, inv_d12_z, az_min, az_max);

		float amin = fmaxf(0.0f, ax_min);
		amin = fmaxf(amin, ay_min);
		amin = fmaxf(amin, az_min);

		float amax = fminf(1.0f, ax_max);
		amax = fminf(amax, ay_max);
		amax = fminf(amax, az_max);

		if constexpr (HasTOF)
		{
			float amin_tof, amax_tof;
			pd_tofHelper->getAlphaRange(amin_tof, amax_tof, d_norm, tofValue);
			amin = max(amin, amin_tof);
			amax = min(amax, amax_tof);
		}

		const float x_0 = d1.x + amin * (d2_minus_d1.x);
		const float y_0 = d1.y + amin * (d2_minus_d1.y);
		const float z_0 = d1.z + amin * (d2_minus_d1.z);
		const float x_1 = d1.x + amax * (d2_minus_d1.x);
		const float y_1 = d1.y + amax * (d2_minus_d1.y);
		const float z_1 = d1.z + amax * (d2_minus_d1.z);
		const int x_i_0 =
		    floor(x_0 / dx + 0.5f * static_cast<float>(nx - 1) + 0.5f);
		const int y_i_0 =
		    floor(y_0 / dy + 0.5f * static_cast<float>(ny - 1) + 0.5f);
		const int z_i_0 =
		    floor(z_0 / dz + 0.5f * static_cast<float>(nz - 1) + 0.5f);
		const int x_i_1 =
		    floor(x_1 / dx + 0.5f * static_cast<float>(nx - 1) + 0.5f);
		const int y_i_1 =
		    floor(y_1 / dy + 0.5f * static_cast<float>(ny - 1) + 0.5f);
		const int z_i_1 =
		    floor(z_1 / dz + 0.5f * static_cast<float>(nz - 1) + 0.5f);

		float d1_i, d2_i, n1_i, n2_i;
		if (flag_y)
		{
			d1_i = d1.y;
			d2_i = d2.y;
			n1_i = n1.y;
			n2_i = n2.y;
		}
		else
		{
			d1_i = d1.x;
			d2_i = d2.x;
			n1_i = n1.x;
			n2_i = n2.x;
		}

		// Normal vectors (in-plane and through-plane)
		const float n1_xy_norm2 = n1.x * n1.x + n1.y * n1.y;
		const float n1_xy_norm = std::sqrt(n1_xy_norm2);
		const float n1_p_x = n1.y / n1_xy_norm;
		const float n1_p_y = -n1.x / n1_xy_norm;
		const float n1_z_norm =
		    std::sqrt((n1.x * n1.z) * (n1.x * n1.z) +
		              (n1.y * n1.z) * (n1.y * n1.z) + n1_xy_norm2);
		const float n1_p_i = (n1_i * n1.z) / n1_z_norm;
		const float n1_p_z = -n1_xy_norm2 / n1_z_norm;
		const float n2_xy_norm2 = n2.x * n2.x + n2.y * n2.y;
		const float n2_xy_norm = std::sqrt(n2_xy_norm2);
		const float n2_p_x = n2.y / n2_xy_norm;
		const float n2_p_y = -n2.x / n2_xy_norm;
		const float n2_z_norm =
		    std::sqrt((n2.x * n2.z) * (n2.x * n2.z) +
		              (n2.y * n2.z) * (n2.y * n2.z) + n2_xy_norm2);
		const float n2_p_i = (n2_i * n2.z) / n2_z_norm;
		const float n2_p_z = -n2_xy_norm2 / n2_z_norm;

		// In-plane detector edges
		const float half_thickness_trans = thickness_trans * 0.5f;
		const float d1_xy_lo_x = d1.x - half_thickness_trans * n1_p_x;
		const float d1_xy_lo_y = d1.y - half_thickness_trans * n1_p_y;
		const float d1_xy_hi_x = d1.x + half_thickness_trans * n1_p_x;
		const float d1_xy_hi_y = d1.y + half_thickness_trans * n1_p_y;
		const float d2_xy_lo_x = d2.x - half_thickness_trans * n2_p_x;
		const float d2_xy_lo_y = d2.y - half_thickness_trans * n2_p_y;
		const float d2_xy_hi_x = d2.x + half_thickness_trans * n2_p_x;
		const float d2_xy_hi_y = d2.y + half_thickness_trans * n2_p_y;

		// Through-plane detector edges
		const float half_thickness_z = thickness_z * 0.5f;
		const float d1_z_lo_i = d1_i - half_thickness_z * n1_p_i;
		const float d1_z_lo_z = d1.z - half_thickness_z * n1_p_z;
		const float d1_z_hi_i = d1_i + half_thickness_z * n1_p_i;
		const float d1_z_hi_z = d1.z + half_thickness_z * n1_p_z;
		const float d2_z_lo_i = d2_i - half_thickness_z * n2_p_i;
		const float d2_z_lo_z = d2.z - half_thickness_z * n2_p_z;
		const float d2_z_hi_i = d2_i + half_thickness_z * n2_p_i;
		const float d2_z_hi_z = d2.z + half_thickness_z * n2_p_z;

		int xy_i_0, xy_i_1;
		float lxy, lyx, dxy, dyx;
		int nyx;
		float d1_xy_lo, d1_xy_hi, d2_xy_lo, d2_xy_hi;
		float d1_yx_lo, d1_yx_hi, d2_yx_lo, d2_yx_hi;
		if (flag_y)
		{
			xy_i_0 = max(0, min(y_i_0, y_i_1));
			xy_i_1 = min(ny - 1, max(y_i_0, y_i_1));
			lxy = imgLength_y;
			dxy = dy;
			lyx = imgLength_x;
			dyx = dx;
			nyx = nx;
			d1_xy_lo = d1_xy_lo_y;
			d1_xy_hi = d1_xy_hi_y;
			d2_xy_lo = d2_xy_lo_y;
			d2_xy_hi = d2_xy_hi_y;
			d1_yx_lo = d1_xy_lo_x;
			d1_yx_hi = d1_xy_hi_x;
			d2_yx_lo = d2_xy_lo_x;
			d2_yx_hi = d2_xy_hi_x;
		}
		else
		{
			xy_i_0 = max(0, min(x_i_0, x_i_1));
			xy_i_1 = min(nx - 1, max(x_i_0, x_i_1));
			lxy = imgLength_x;
			dxy = dx;
			lyx = imgLength_y;
			dyx = dy;
			nyx = ny;
			d1_xy_lo = d1_xy_lo_x;
			d1_xy_hi = d1_xy_hi_x;
			d2_xy_lo = d2_xy_lo_x;
			d2_xy_hi = d2_xy_hi_x;
			d1_yx_lo = d1_xy_lo_y;
			d1_yx_hi = d1_xy_hi_y;
			d2_yx_lo = d2_xy_lo_y;
			d2_yx_hi = d2_xy_hi_y;
		}

		float dxy_cos_theta;
		if (d1_i != d2_i)
		{
			dxy_cos_theta = dxy / (fabsf(d1_i - d2_i) / d_norm);
		}
		else
		{
			dxy_cos_theta = dxy;
		}

		for (int xyi = xy_i_0; xyi <= xy_i_1; xyi++)
		{
			const float pix_xy =
			    -0.5f * lxy + (static_cast<float>(xyi) + 0.5f) * dxy;
			const float a_xy_lo = (pix_xy - d1_xy_lo) / (d2_xy_hi - d1_xy_lo);
			const float a_xy_hi = (pix_xy - d1_xy_hi) / (d2_xy_lo - d1_xy_hi);
			const float a_z_lo = (pix_xy - d1_z_lo_i) / (d2_z_lo_i - d1_z_lo_i);
			const float a_z_hi = (pix_xy - d1_z_hi_i) / (d2_z_hi_i - d1_z_hi_i);
			float dd_yx_r_0 = d1_yx_lo + a_xy_lo * (d2_yx_hi - d1_yx_lo);
			float dd_yx_r_1 = d1_yx_hi + a_xy_hi * (d2_yx_lo - d1_yx_hi);
			if (dd_yx_r_0 > dd_yx_r_1)
			{
				// swap
				float tmp = dd_yx_r_1;
				dd_yx_r_1 = dd_yx_r_0;
				dd_yx_r_0 = tmp;
			}
			const float widthFrac_yx = dd_yx_r_1 - dd_yx_r_0;
			// Save bounds without extension for overlap calculation
			const float dd_yx_r_0_ov = dd_yx_r_0;
			const float dd_yx_r_1_ov = dd_yx_r_1;
			dd_yx_r_0 -= detFootprintExt;
			dd_yx_r_1 += detFootprintExt;
			const float offset_dd_yx_i = static_cast<float>(nyx - 1) * 0.5f;
			const int dd_yx_i_0 = max(
			    0, static_cast<int>(rintf(dd_yx_r_0 / dyx + offset_dd_yx_i)));
			const int dd_yx_i_1 =
			    min(nyx - 1,
			        static_cast<int>(rintf(dd_yx_r_1 / dyx + offset_dd_yx_i)));
			for (int yxi = dd_yx_i_0; yxi <= dd_yx_i_1; yxi++)
			{
				const float pix_yx =
				    -0.5f * lyx + (static_cast<float>(yxi) + 0.5f) * dyx;
				float dd_z_r_0 = d1_z_lo_z + a_z_lo * (d2_z_lo_z - d1_z_lo_z);
				float dd_z_r_1 = d1_z_hi_z + a_z_hi * (d2_z_hi_z - d1_z_hi_z);
				if (dd_z_r_0 > dd_z_r_1)
				{
					float tmp = dd_z_r_1;
					dd_z_r_1 = dd_z_r_0;
					dd_z_r_0 = tmp;
				}
				const float widthFrac_z = dd_z_r_1 - dd_z_r_0;
				const float dd_yx_p_0 = pix_yx - dyx * 0.5f;
				const float dd_yx_p_1 = pix_yx + dyx * 0.5f;
				if (dd_yx_r_1 >= dd_yx_p_0 && dd_yx_r_0 < dd_yx_p_1)
				{
					float weight_xy;
					if constexpr (HasProjPSF)
					{
						weight_xy = get_overlap_safe(
						    dd_yx_p_0, dd_yx_p_1, dd_yx_r_0_ov, dd_yx_r_1_ov,
						    psfKernel, projectionPsfProperties);
					}
					else
					{
						weight_xy = get_overlap_safe(
						    dd_yx_p_0, dd_yx_p_1, dd_yx_r_0_ov, dd_yx_r_1_ov);
					}

					const float weight_xy_s = weight_xy / widthFrac_yx;
					const float offset_dd_z_i =
					    static_cast<float>(nz - 1) * 0.5f;
					const int dd_z_i_0 = max(
					    0,
					    static_cast<int>(rintf(dd_z_r_0 / dz + offset_dd_z_i)));
					const int dd_z_i_1 = min(
					    nz - 1,
					    static_cast<int>(rintf(dd_z_r_1 / dz + offset_dd_z_i)));
					for (int zi = dd_z_i_0; zi <= dd_z_i_1; zi++)
					{
						const float pix_z =
						    -0.5f * imgLength_z +
						    (static_cast<float>(zi) + 0.5f) * dz;

						float tof_weight = 1.0f;
						if constexpr (HasTOF)
						{
							const float a_lo =
							    (pix_xy - d1_i - 0.5f * dxy) / (d2_i - d1_i);
							const float a_hi =
							    (pix_xy - d1_i + 0.5f * dxy) / (d2_i - d1_i);
							tof_weight = pd_tofHelper->getWeight(
							    d_norm, tofValue, a_lo * d_norm, a_hi * d_norm);
						}

						const float half_dz = dz * 0.5f;
						const float dd_z_p_0 = pix_z - half_dz;
						const float dd_z_p_1 = pix_z + half_dz;
						if (dd_z_r_1 >= dd_z_p_0 && dd_z_r_0 < dd_z_p_1)
						{
							const float weight_z = get_overlap_safe(
							    dd_z_p_0, dd_z_p_1, dd_z_r_0, dd_z_r_1);
							const float weight_z_s = weight_z / widthFrac_z;
							size_t idx = zi * num_xy;

							if (flag_y)
							{
								idx += nx * xyi + yxi;
							}
							else
							{
								idx += nx * yxi + xyi;
							}

							float weight =
							    weight_xy_s * weight_z_s * dxy_cos_theta;

							if constexpr (HasTOF)
							{
								weight *= tof_weight;
							}

							float* ptr = pd_image + idx;
							if constexpr (IsForward)
							{
								value += (*ptr) * weight;
							}
							else
							{
								atomicAdd(ptr, value * weight);
							}
						}
					}
				}
			}
		}
		pd_projValues[eventId] = value;
	}
}

template __global__ void OperatorProjectorDDCU_kernel<true, true, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorDDCU_kernel<false, true, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorDDCU_kernel<true, false, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorDDCU_kernel<false, false, false>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorDDCU_kernel<true, true, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorDDCU_kernel<false, true, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorDDCU_kernel<true, false, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
template __global__ void OperatorProjectorDDCU_kernel<false, false, true>(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize);
