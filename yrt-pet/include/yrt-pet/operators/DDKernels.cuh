/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/OperatorProjectorDevice.cuh"
#include "yrt-pet/operators/ProjectionPsfUtils.cuh"
#include "yrt-pet/operators/ProjectorUtils.hpp"
#include "yrt-pet/utils/GPUKernelUtils.cuh"

#include <cfloat>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace yrt
{

__device__ inline float getOverlap_safe(const float p0, const float p1,
                                        const float d0, const float d1)
{
	return min(p1, d1) - max(p0, d0);
}

__device__ inline float
    getOverlap_safe(const float p0, const float p1, const float d0,
                    const float d1, const float* psfKernel,
                    const ProjectionPsfProperties& projectionPsfProperties)
{
	return util::getWeight(psfKernel, projectionPsfProperties, p0 - d1,
	                       p1 - d0);
}

template <bool IsForward, bool HasTOF, bool HasProjPSF, bool UseUpdater>
__device__ void
    projectDD(float& value, float* pd_image, UpdaterPointer pd_updater,
              float3 p1, float3 p2, float3 n1, float3 n2, frame_t dynamicFrame,
              const TimeOfFlightHelper* pd_tofHelper, float tofValue,
              ProjectionPsfKernelStruct projPsfKernelStruct,
              CUScannerParams scannerParams, CUImageParams imgParams)
{
	ProjectionPsfProperties projectionPsfProperties =
	    projPsfKernelStruct.properties;
	const float* pd_projPsfKernels = projPsfKernelStruct.kernels;

	float localValue = 0.0f;
	if constexpr (!IsForward)
	{
		localValue = value;
	}

	const float3 p1_minus_p2 = p1 - p2;
	const float3 p2_minus_p1 = p1_minus_p2 * (-1.0f);

	// ----------------------- Compute TOR
	const float thickness_z = scannerParams.crystalSize_z;
	const float thickness_trans = scannerParams.crystalSize_trans;

	const bool flag_y = fabs(p2_minus_p1.y) > fabs(p2_minus_p1.x);
	const float d_norm = norm3df(p1_minus_p2.x, p1_minus_p2.y, p1_minus_p2.z);

	const float* psfKernel = nullptr;
	float detFootprintExt = 0.f;
	if constexpr (HasProjPSF)
	{
		psfKernel = util::getKernel(pd_projPsfKernels, projectionPsfProperties,
		                            p1.x, p1.y, p2.x, p2.y);
		detFootprintExt = projectionPsfProperties.halfWidth;
	}

	// ----------------------- Compute Pixel limits
	const int nx = imgParams.nx;
	const int ny = imgParams.ny;
	const int nz = imgParams.nz;
	const float imgLength_x = imgParams.length_x;
	const float imgLength_y = imgParams.length_y;
	const float imgLength_z = imgParams.length_z;
	const int num_xy = nx * ny;
	const size_t numVoxelsPerFrame = nx * ny * nz;
	const float dx = imgParams.vx;
	const float dy = imgParams.vy;
	const float dz = imgParams.vz;

	const float inv_d12_x = (p1.x == p2.x) ? 0.0f : 1.0f / (p2.x - p1.x);
	const float inv_d12_y = (p1.y == p2.y) ? 0.0f : 1.0f / (p2.y - p1.y);
	const float inv_d12_z = (p1.z == p2.z) ? 0.0f : 1.0f / (p2.z - p1.z);

	float ax_min, ax_max, ay_min, ay_max, az_min, az_max;
	util::get_alpha(-0.5f * (imgLength_x - dx), 0.5f * (imgLength_x - dx), p1.x,
	                p2.x, inv_d12_x, ax_min, ax_max);
	util::get_alpha(-0.5f * (imgLength_y - dy), 0.5f * (imgLength_y - dy), p1.y,
	                p2.y, inv_d12_y, ay_min, ay_max);
	util::get_alpha(-0.5f * (imgLength_z - dz), 0.5f * (imgLength_z - dz), p1.z,
	                p2.z, inv_d12_z, az_min, az_max);

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

	const float x_0 = p1.x + amin * (p2_minus_p1.x);
	const float y_0 = p1.y + amin * (p2_minus_p1.y);
	const float z_0 = p1.z + amin * (p2_minus_p1.z);
	const float x_1 = p1.x + amax * (p2_minus_p1.x);
	const float y_1 = p1.y + amax * (p2_minus_p1.y);
	const float z_1 = p1.z + amax * (p2_minus_p1.z);
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
		d1_i = p1.y;
		d2_i = p2.y;
		n1_i = n1.y;
		n2_i = n2.y;
	}
	else
	{
		d1_i = p1.x;
		d2_i = p2.x;
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
	const float d1_xy_lo_x = p1.x - half_thickness_trans * n1_p_x;
	const float d1_xy_lo_y = p1.y - half_thickness_trans * n1_p_y;
	const float d1_xy_hi_x = p1.x + half_thickness_trans * n1_p_x;
	const float d1_xy_hi_y = p1.y + half_thickness_trans * n1_p_y;
	const float d2_xy_lo_x = p2.x - half_thickness_trans * n2_p_x;
	const float d2_xy_lo_y = p2.y - half_thickness_trans * n2_p_y;
	const float d2_xy_hi_x = p2.x + half_thickness_trans * n2_p_x;
	const float d2_xy_hi_y = p2.y + half_thickness_trans * n2_p_y;

	// Through-plane detector edges
	const float half_thickness_z = thickness_z * 0.5f;
	const float d1_z_lo_i = d1_i - half_thickness_z * n1_p_i;
	const float d1_z_lo_z = p1.z - half_thickness_z * n1_p_z;
	const float d1_z_hi_i = d1_i + half_thickness_z * n1_p_i;
	const float d1_z_hi_z = p1.z + half_thickness_z * n1_p_z;
	const float d2_z_lo_i = d2_i - half_thickness_z * n2_p_i;
	const float d2_z_lo_z = p2.z - half_thickness_z * n2_p_z;
	const float d2_z_hi_i = d2_i + half_thickness_z * n2_p_i;
	const float d2_z_hi_z = p2.z + half_thickness_z * n2_p_z;

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
		const int dd_yx_i_0 =
		    max(0, static_cast<int>(rintf(dd_yx_r_0 / dyx + offset_dd_yx_i)));
		const int dd_yx_i_1 = min(
		    nyx - 1, static_cast<int>(rintf(dd_yx_r_1 / dyx + offset_dd_yx_i)));
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
					weight_xy = getOverlap_safe(
					    dd_yx_p_0, dd_yx_p_1, dd_yx_r_0_ov, dd_yx_r_1_ov,
					    psfKernel, projectionPsfProperties);
				}
				else
				{
					weight_xy = getOverlap_safe(dd_yx_p_0, dd_yx_p_1,
					                            dd_yx_r_0_ov, dd_yx_r_1_ov);
				}

				const float weight_xy_s = weight_xy / widthFrac_yx;
				const float offset_dd_z_i = static_cast<float>(nz - 1) * 0.5f;
				const int dd_z_i_0 = max(
				    0, static_cast<int>(rintf(dd_z_r_0 / dz + offset_dd_z_i)));
				const int dd_z_i_1 =
				    min(nz - 1,
				        static_cast<int>(rintf(dd_z_r_1 / dz + offset_dd_z_i)));
				for (int zi = dd_z_i_0; zi <= dd_z_i_1; zi++)
				{
					const float pix_z = -0.5f * imgLength_z +
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
						const float weight_z = getOverlap_safe(
						    dd_z_p_0, dd_z_p_1, dd_z_r_0, dd_z_r_1);
						const float weight_z_s = weight_z / widthFrac_z;
						size_t imageOffset = zi * num_xy;

						if (flag_y)
						{
							imageOffset += nx * xyi + yxi;
						}
						else
						{
							imageOffset += nx * yxi + xyi;
						}

						float weight = weight_xy_s * weight_z_s * dxy_cos_theta;

						if constexpr (HasTOF)
						{
							weight *= tof_weight;
						}

						if constexpr (IsForward)
						{
							if constexpr (UseUpdater)
							{
								localValue +=
								    (*pd_updater)
								        ->forwardUpdate(
								            weight, pd_image, imageOffset,
								            dynamicFrame, numVoxelsPerFrame);
							}
							else
							{
								localValue +=
								    weight *
								    pd_image[dynamicFrame * numVoxelsPerFrame +
								             imageOffset];
							}
						}
						else
						{
							if constexpr (UseUpdater)
							{
								(*pd_updater)
								    ->backUpdate(localValue, weight, pd_image,
								                 imageOffset, dynamicFrame,
								                 numVoxelsPerFrame);
							}
							else
							{
								const float output = value * weight;
								atomicAdd(pd_image +
								              dynamicFrame * numVoxelsPerFrame +
								              imageOffset,
								          output);
							}
						}
					}
				}
			}
		}
	}
	if constexpr (IsForward)
	{
		value = localValue;
	}
}


template <bool IsForward, bool HasTOF, bool HasProjPSF, bool UseUpdater>
__global__ void projectDD_kernel(
    float* pd_projValues, float* pd_image, UpdaterPointer pd_updater,
    const ProjectionPropertyManager* pd_projPropManager,
    const PropertyUnit* pd_projectionProperties,
    const TimeOfFlightHelper* pd_tofHelper,
    ProjectionPsfKernelStruct projPsfKernelStruct,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < batchSize)
	{
		float& value = pd_projValues[eventId];

		const float* lor = pd_projPropManager->getDataPtr<float>(
		    pd_projectionProperties, eventId, ProjectionPropertyType::LOR);
		static_assert(sizeof(Line3D) == sizeof(float3) * 2);
		const float3 imageOffset =
		    make_float3(imgParams.off_x, imgParams.off_y, imgParams.off_z);

		float3 p1{lor[0], lor[1], lor[2]};
		float3 p2{lor[3], lor[4], lor[5]};

		p1 -= imageOffset;
		p2 -= imageOffset;

		const float* detOrient = pd_projPropManager->getDataPtr<float>(
		    pd_projectionProperties, eventId,
		    ProjectionPropertyType::DET_ORIENT);
		const float3 n1{detOrient[0], detOrient[1], detOrient[2]};
		const float3 n2{detOrient[3], detOrient[4], detOrient[5]};

		float tofValue = 0.0f;
		if constexpr (HasTOF)
		{
			const float* tofValuePtr = pd_projPropManager->getDataPtr<float>(
			    pd_projectionProperties, eventId, ProjectionPropertyType::TOF);
			tofValue = *tofValuePtr;
		}

		frame_t dynamicFrame = 0;
		if (pd_projPropManager->has(ProjectionPropertyType::DYNAMIC_FRAME))
		{
			dynamicFrame = *pd_projPropManager->getDataPtr<frame_t>(
			    pd_projectionProperties, eventId,
			    ProjectionPropertyType::DYNAMIC_FRAME);
		}

		if (dynamicFrame >= 0)
		{
			projectDD<IsForward, HasTOF, HasProjPSF, UseUpdater>(
			    value, pd_image, pd_updater, p1, p2, n1, n2, dynamicFrame,
			    pd_tofHelper, tofValue, projPsfKernelStruct, scannerParams,
			    imgParams);
		}
		else
		{
			// If the frame is disabled, do nothing
			// If doing a forward projection, populate the value at zero
			if constexpr (IsForward)
			{
				value = 0.0f;
			}
		}
	}
}

}  // namespace yrt
