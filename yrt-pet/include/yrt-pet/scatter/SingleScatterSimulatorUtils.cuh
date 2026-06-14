/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/GPUUtils.cuh"

#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/geometry/Cylinder.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/geometry/Plane.hpp"
#include "yrt-pet/operators/ProjectorUtils.hpp"
#include "yrt-pet/recon/RawParameters.hpp"
#include "yrt-pet/scatter/Crystal.hpp"
#include "yrt-pet/utils/Tools.hpp"

#ifndef __CUDACC__
#include "yrt-pet/datastruct/image/Image.hpp"
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>
// Note: siddonForwardProjection replaces ProjectorSiddon for the raw-pointer
// path
#endif

#ifdef __CUDACC__
#include "yrt-pet/geometry/Line3D_impl.inl"
#include "yrt-pet/geometry/Vector3D_impl.inl"
#include "yrt-pet/operators/SiddonKernels.cuh"
#endif


namespace yrt::scatter
{

// Klein-Nishina differential cross section (for Ep=511keV).
HOST_DEVICE_CALLABLE inline float getKleinNishina(float cosa)
{
	float res = (1 + cosa * cosa) / 2;
	res /= (2 - cosa) * (2 - cosa);
	res *= 1 + (1 - cosa) * (1 - cosa) / ((2 - cosa) * (1 + cosa * cosa));
	return res;
}

// Integrated Klein-Nishina up to a proportionality constant.
HOST_DEVICE_CALLABLE inline float getMuScalingFactor(float energy)
{
	float a = energy / 511.0f;
	float res = (1 + a) / (a * a);
	res *= 2.0f * (1 + a) / (1 + 2.0f * a) - log(1 + 2.0f * a) / a;
	res += log(1 + 2 * a) / (2 * a) - (1 + 3 * a) / ((1 + 2 * a) * (1 + 2 * a));
	res /= 20.0f / 9.0f - 1.5f * log(3.0f);
	return res;
}

// The first point of lor must be the detector, the second point must be the
// scatter point.
HOST_DEVICE_CALLABLE inline float
    getIntersectionLengthLORCrystalRaw(const Line3D& lor, const Cylinder& cyl1,
                                       const Cylinder& cyl2)
{
	Vector3D a1, a2, inter1, inter2;
	const Vector3D n1 = lor.point1 - lor.point2;

	cyl1.doesLineIntersectCylinder(lor, a1, a2);
	Vector3D n2 = a1 - lor.point2;
	if (n2.scalProd(n1) > 0)
		inter1.update(a1);
	else
		inter1.update(a2);

	cyl2.doesLineIntersectCylinder(lor, a1, a2);
	n2 = a1 - lor.point2;
	if (n2.scalProd(n1) > 0)
		inter2.update(a1);
	else
		inter2.update(a2);

	return (inter1 - inter2).getNorm();
}

// Return true if the line lor does not cross the end plates.
// First point is detector, second point is scatter point.
HOST_DEVICE_CALLABLE inline bool passCollimatorRaw(const Line3D& lor,
                                                   float collimatorRadius,
                                                   float /*axialFOV*/,
                                                   const Plane& endPlate1,
                                                   const Plane& endPlate2)
{
	if (collimatorRadius < 1e-7f)
		return true;
	Vector3D inter;
	if (lor.point2.z < 0)
		inter = endPlate1.findInterLine(lor);
	else
		inter = endPlate2.findInterLine(lor);
	const float r = sqrtf(inter.x * inter.x + inter.y * inter.y);
	return r < collimatorRadius;
}

// Siddon forward projection using raw data pointer.
// Works on both CPU and GPU (no CUDA-specific dependencies, no TOF, no
// multi-ray, no backprojection, no updater).
HOST_DEVICE_CALLABLE inline float siddonForwardProjection(const float* data,
                                                          RawImageParams params,
                                                          const Line3D& lor)
{
	float projValue = 0.0f;

	const ssize_t nx = params.nx;
	const ssize_t ny = params.ny;
	const ssize_t nz = params.nz;
	const float length_x = params.length_x;
	const float length_y = params.length_y;
	const float length_z = params.length_z;
	const ssize_t num_xy = nx * ny;
	const float dx = params.vx;
	const float dy = params.vy;
	const float dz = params.vz;
	const float inv_dx = 1.0f / dx;
	const float inv_dy = 1.0f / dy;
	const float inv_dz = 1.0f / dz;

	const float p1x = lor.point1.x;
	const float p1y = lor.point1.y;
	const float p1z = lor.point1.z;
	const float p2x = lor.point2.x;
	const float p2y = lor.point2.y;
	const float p2z = lor.point2.z;

	// 1. Intersection with FOV cylinder
	float A = (p2x - p1x) * (p2x - p1x) + (p2y - p1y) * (p2y - p1y);
	float B = 2.0f * ((p2x - p1x) * p1x + (p2y - p1y) * p1y);
	float C = p1x * p1x + p1y * p1y - params.fovRadius * params.fovRadius;
	float Delta = B * B - 4.0f * A * C;

	float t0, t1;
	if (A != 0.0f)
	{
		if (Delta <= 0.0f)
			return 0.0f;
		t0 = (-B - sqrtf(Delta)) / (2.0f * A);
		t1 = (-B + sqrtf(Delta)) / (2.0f * A);
	}
	else
	{
		t0 = 0.0f;
		t1 = 1.0f;
	}

	float d_norm = sqrtf((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y) +
	                     (p1z - p2z) * (p1z - p2z));
	bool flat_x = (p1x == p2x);
	bool flat_y = (p1y == p2y);
	bool flat_z = (p1z == p2z);
	float inv_p12_x = flat_x ? 0.0f : 1.0f / (p2x - p1x);
	float inv_p12_y = flat_y ? 0.0f : 1.0f / (p2y - p1y);
	float inv_p12_z = flat_z ? 0.0f : 1.0f / (p2z - p1z);
	ssize_t dir_x = (inv_p12_x >= 0.0f) ? 1 : -1;
	ssize_t dir_y = (inv_p12_y >= 0.0f) ? 1 : -1;
	ssize_t dir_z = (inv_p12_z >= 0.0f) ? 1 : -1;

	// 2. Intersection with volume
	float ax_min, ax_max, ay_min, ay_max, az_min, az_max;
	util::get_alpha(-0.5f * length_x, 0.5f * length_x, p1x, p2x, inv_p12_x,
	                ax_min, ax_max);
	util::get_alpha(-0.5f * length_y, 0.5f * length_y, p1y, p2y, inv_p12_y,
	                ay_min, ay_max);
	util::get_alpha(-0.5f * length_z, 0.5f * length_z, p1z, p2z, inv_p12_z,
	                az_min, az_max);

	float amin = fmaxf(0.0f, t0);
	amin = fmaxf(amin, ax_min);
	amin = fmaxf(amin, ay_min);
	amin = fmaxf(amin, az_min);
	float amax = fminf(1.0f, t1);
	amax = fminf(amax, ax_max);
	amax = fminf(amax, ay_max);
	amax = fminf(amax, az_max);

	float a_cur = amin;
	float a_next = -1.0f;
	float x0 = -length_x * 0.5f;
	float x1 = length_x * 0.5f;
	float y0 = -length_y * 0.5f;
	float y1 = length_y * 0.5f;
	float z0 = -length_z * 0.5f;
	float z1 = length_z * 0.5f;
	float x_cur = (inv_p12_x > 0.0f) ? x0 : x1;
	float y_cur = (inv_p12_y > 0.0f) ? y0 : y1;
	float z_cur = (inv_p12_z > 0.0f) ? z0 : z1;
	if ((inv_p12_x >= 0.0f && p1x > x1) || (inv_p12_x < 0.0f && p1x < x0) ||
	    (inv_p12_y >= 0.0f && p1y > y1) || (inv_p12_y < 0.0f && p1y < y0) ||
	    (inv_p12_z >= 0.0f && p1z > z1) || (inv_p12_z < 0.0f && p1z < z0))
	{
		return 0.0f;
	}

	constexpr float SIDDON_BIG = 3.402823466e+38f;
	float ax_next = flat_x ? SIDDON_BIG : ax_min;
	if (!flat_x)
	{
		float kx = ceilf(dir_x * (a_cur * (p2x - p1x) - x_cur + p1x) / dx);
		x_cur += kx * dir_x * dx;
		ax_next = (x_cur - p1x) * inv_p12_x;
	}
	float ay_next = flat_y ? SIDDON_BIG : ay_min;
	if (!flat_y)
	{
		float ky = ceilf(dir_y * (a_cur * (p2y - p1y) - y_cur + p1y) / dy);
		y_cur += ky * dir_y * dy;
		ay_next = (y_cur - p1y) * inv_p12_y;
	}
	float az_next = flat_z ? SIDDON_BIG : az_min;
	if (!flat_z)
	{
		float kz = ceilf(dir_z * (a_cur * (p2z - p1z) - z_cur + p1z) / dz);
		z_cur += kz * dir_z * dz;
		az_next = (z_cur - p1z) * inv_p12_z;
	}

	// 3. Integrate along ray
	enum SIDDON_DIR_FLAGS
	{
		DIR_X_FLAG = 0b001,
		DIR_Y_FLAG = 0b010,
		DIR_Z_FLAG = 0b100
	};

	bool flag_first = true;
	ssize_t vx = -1, vy = -1, vz = -1;
	short dir_prev = -1;
	ssize_t offset_img_ptr = 0;

	float ax_next_prev = ax_next;
	float ay_next_prev = ay_next;
	float az_next_prev = az_next;
	bool flag_done = false;

	while (a_cur < amax && !flag_done)
	{
		short dir_next = 0;
		if (ax_next_prev <= ay_next_prev && ax_next_prev <= az_next_prev)
		{
			a_next = ax_next;
			x_cur += dir_x * dx;
			ax_next = (x_cur - p1x) * inv_p12_x;
			dir_next |= DIR_X_FLAG;
		}
		if (ay_next_prev <= ax_next_prev && ay_next_prev <= az_next_prev)
		{
			a_next = ay_next;
			y_cur += dir_y * dy;
			ay_next = (y_cur - p1y) * inv_p12_y;
			dir_next |= DIR_Y_FLAG;
		}
		if (az_next_prev <= ax_next_prev && az_next_prev <= ay_next_prev)
		{
			a_next = az_next;
			z_cur += dir_z * dz;
			az_next = (z_cur - p1z) * inv_p12_z;
			dir_next |= DIR_Z_FLAG;
		}
		if (a_next > amax)
			a_next = amax;
		if (a_cur >= a_next)
		{
			ax_next_prev = ax_next;
			ay_next_prev = ay_next;
			az_next_prev = az_next;
			continue;
		}

		float a_mid = 0.5f * (a_cur + a_next);
		if (flag_first)
		{
			vx = static_cast<ssize_t>(
			    (p1x + a_mid * (p2x - p1x) + length_x * 0.5f) * inv_dx);
			vy = static_cast<ssize_t>(
			    (p1y + a_mid * (p2y - p1y) + length_y * 0.5f) * inv_dy);
			vz = static_cast<ssize_t>(
			    (p1z + a_mid * (p2z - p1z) + length_z * 0.5f) * inv_dz);
			offset_img_ptr = vz * num_xy + vy * nx;
			flag_first = false;
			if (vx < 0 || vx >= nx || vy < 0 || vy >= ny || vz < 0 || vz >= nz)
				flag_done = true;
		}
		else
		{
			if (dir_prev & DIR_X_FLAG)
			{
				vx += dir_x;
				if (vx < 0 || vx >= nx)
					flag_done = true;
			}
			if (dir_prev & DIR_Y_FLAG)
			{
				vy += dir_y;
				if (vy < 0 || vy >= ny)
					flag_done = true;
				else
					offset_img_ptr += dir_y * nx;
			}
			if (dir_prev & DIR_Z_FLAG)
			{
				vz += dir_z;
				if (vz < 0 || vz >= nz)
					flag_done = true;
				else
					offset_img_ptr += dir_z * num_xy;
			}
		}
		if (flag_done)
			continue;

		dir_prev = dir_next;
		float weight = (a_next - a_cur) * d_norm;
		ssize_t imageOffset = vx + offset_img_ptr;
		projValue += weight * data[imageOffset];
		a_cur = a_next;
		ax_next_prev = ax_next;
		ay_next_prev = ay_next;
		az_next_prev = az_next;
	}

	return projValue;
}

// Unified HOST_DEVICE_CALLABLE computeSingleScatterInLOR.
// Uses raw data pointers + RawImageParams (works on both CPU and GPU).
HOST_DEVICE_CALLABLE inline float computeSingleScatterInLOR(
    const Line3D& lor, float /*tof_ps*/, int numSamples, const float* xSamples,
    const float* ySamples, const float* zSamples, float energyLLD,
    float sigmaEnergy, float crystalDepth, float axialFOV,
    float collimatorRadius, CrystalMaterial crystalMaterial,
    const Cylinder& cyl1, const Cylinder& cyl2, const Plane& endPlate1,
    const Plane& endPlate2, const float* mu_data, const float* lambda_data,
    RawImageParams mu_params, RawImageParams lambda_params, float3 imageOffset)
{
	int i;
	double res = 0.0;
	float dist1, dist2, energy, cosa, mu_scaling_factor;
	float vatt, att_s_1_511, att_s_1, att_s_2_511, att_s_2;
	float dsigcompdomega, lamb_s_1, lamb_s_2, sig_s_1, sig_s_2;
	float eps_s_1_511, eps_s_1, eps_s_2_511, eps_s_2, fac1, fac2;
	float tmp, tmp511, delta_1, delta_2, mu_det, mu_det_511;
	Line3D lor_1_s, lor_2_s;
	Vector3D ps, p1, p2, u, v;

	p1.update(lor.point1);
	p2.update(lor.point2);

	Vector3D n1 = {lor.point1.x, lor.point1.y, 0.0f};
	n1.normalize();
	Vector3D n2 = {lor.point2.x, lor.point2.y, 0.0f};
	n2.normalize();

	tmp511 = (energyLLD - 511.0f) / (sqrtf(2.0f) * sigmaEnergy);
	mu_det_511 = getMuDet(511.0, crystalMaterial);

	for (i = 0; i < numSamples; i++)
	{
		ps.update(xSamples[i], ySamples[i], zSamples[i]);

		lor_1_s.update(p1, ps);
		lor_2_s.update(p2, ps);

		if (fabsf(ps.z) > axialFOV / 2 &&
		    (!passCollimatorRaw(lor_1_s, collimatorRadius, axialFOV, endPlate1,
		                        endPlate2) ||
		     !passCollimatorRaw(lor_2_s, collimatorRadius, axialFOV, endPlate1,
		                        endPlate2)))
			continue;

		u.update(ps - p1);
		dist1 = u.getNorm();
		u.x /= dist1;
		u.y /= dist1;
		u.z /= dist1;
		v.update(p2 - ps);
		dist2 = v.getNorm();
		v.x /= dist2;
		v.y /= dist2;
		v.z /= dist2;

		cosa = u.scalProd(v);
		energy = 511.0f / (2.0f - cosa);
		if (energy <= energyLLD)
			continue;
		tmp = (energyLLD - energy) / (sqrtf(2.0f) * sigmaEnergy);
		mu_scaling_factor = getMuScalingFactor(energy);

		// nearest-neighbor voxel value (inline index math)
		{
			const float x = ps.x - mu_params.off_x;
			const float y = ps.y - mu_params.off_y;
			const float z = ps.z - mu_params.off_z;
			const float dx = (x + mu_params.length_x / 2.0f) /
			                 mu_params.length_x *
			                 static_cast<float>(mu_params.nx);
			const float dy = (y + mu_params.length_y / 2.0f) /
			                 mu_params.length_y *
			                 static_cast<float>(mu_params.ny);
			const float dz = (z + mu_params.length_z / 2.0f) /
			                 mu_params.length_z *
			                 static_cast<float>(mu_params.nz);
			const ssize_t ix = static_cast<ssize_t>(dx);
			const ssize_t iy = static_cast<ssize_t>(dy);
			const ssize_t iz = static_cast<ssize_t>(dz);
			if (ix < 0 || ix >= mu_params.nx || iy < 0 || iy >= mu_params.ny ||
			    iz < 0 || iz >= mu_params.nz)
				vatt = 0.0f;
			else
				vatt = mu_data[iz * mu_params.nx * mu_params.ny +
				               iy * mu_params.nx + ix];
		}
		dsigcompdomega = getKleinNishina(cosa);

		// Forward projection for path to detector 1 (mu at 511keV)
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
		{
			float3 p1_f = make_float3(lor_1_s.point1.x, lor_1_s.point1.y,
			                          lor_1_s.point1.z);
			float3 p2_f = make_float3(lor_1_s.point2.x, lor_1_s.point2.y,
			                          lor_1_s.point2.z);
			p1_f -= imageOffset;
			p2_f -= imageOffset;
			RawScannerParams scannerParams{};
			projectSiddonDefault<true, false>(
			    att_s_1_511, const_cast<float*>(mu_data), nullptr, p1_f, p2_f,
			    make_float3(0, 0, 0), make_float3(0, 0, 0), 0, nullptr, 0.0f,
			    scannerParams, mu_params, 1, 0);
			att_s_1_511 /= 10.0f;
		}
#else
		{
			att_s_1_511 =
			    siddonForwardProjection(mu_data, mu_params, lor_1_s) / 10.0f;
		}
#endif

		att_s_1 = att_s_1_511 * mu_scaling_factor;

		// Forward projection for lambda to detector 1
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
		{
			float3 p1_f = make_float3(lor_1_s.point1.x, lor_1_s.point1.y,
			                          lor_1_s.point1.z);
			float3 p2_f = make_float3(lor_1_s.point2.x, lor_1_s.point2.y,
			                          lor_1_s.point2.z);
			p1_f -= imageOffset;
			p2_f -= imageOffset;
			RawScannerParams scannerParams{};
			projectSiddonDefault<true, false>(
			    lamb_s_1, const_cast<float*>(lambda_data), nullptr, p1_f, p2_f,
			    make_float3(0, 0, 0), make_float3(0, 0, 0), 0, nullptr, 0.0f,
			    scannerParams, lambda_params, 1, 0);
		}
#else
		{
			lamb_s_1 =
			    siddonForwardProjection(lambda_data, lambda_params, lor_1_s);
		}
#endif

		delta_1 = getIntersectionLengthLORCrystalRaw(lor_1_s, cyl1, cyl2);

		// Forward projection for path to detector 2 (mu at 511keV)
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
		{
			float3 p1_f = make_float3(lor_2_s.point1.x, lor_2_s.point1.y,
			                          lor_2_s.point1.z);
			float3 p2_f = make_float3(lor_2_s.point2.x, lor_2_s.point2.y,
			                          lor_2_s.point2.z);
			p1_f -= imageOffset;
			p2_f -= imageOffset;
			RawScannerParams scannerParams{};
			projectSiddonDefault<true, false>(
			    att_s_2_511, const_cast<float*>(mu_data), nullptr, p1_f, p2_f,
			    make_float3(0, 0, 0), make_float3(0, 0, 0), 0, nullptr, 0.0f,
			    scannerParams, mu_params, 1, 0);
			att_s_2_511 /= 10.0f;
		}
#else
		{
			if (delta_1 > 10 * crystalDepth)
			{
				std::string errorMessage = "Error computing propagation "
				                           "distance in detector. delta_1=" +
				                           std::to_string(delta_1);
				throw std::runtime_error(errorMessage);
			}

			att_s_2_511 =
			    siddonForwardProjection(mu_data, mu_params, lor_2_s) / 10.0f;
		}
#endif

		att_s_2 = att_s_2_511 * mu_scaling_factor;

		// Forward projection for lambda to detector 2
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
		{
			float3 p1_f = make_float3(lor_2_s.point1.x, lor_2_s.point1.y,
			                          lor_2_s.point1.z);
			float3 p2_f = make_float3(lor_2_s.point2.x, lor_2_s.point2.y,
			                          lor_2_s.point2.z);
			p1_f -= imageOffset;
			p2_f -= imageOffset;
			RawScannerParams scannerParams{};
			projectSiddonDefault<true, false>(
			    lamb_s_2, const_cast<float*>(lambda_data), nullptr, p1_f, p2_f,
			    make_float3(0, 0, 0), make_float3(0, 0, 0), 0, nullptr, 0.0f,
			    scannerParams, lambda_params, 1, 0);
		}
#else
		{
			lamb_s_2 =
			    siddonForwardProjection(lambda_data, lambda_params, lor_2_s);
		}
#endif

		delta_2 = getIntersectionLengthLORCrystalRaw(lor_2_s, cyl1, cyl2);

#if not defined(__CUDACC__) && not defined(__CUDA_ARCH__)
		// Check that the distance between the two cylinders is not too big
		if (delta_2 > 10 * crystalDepth)
		{
			std::string errorMessage =
			    "Error computing propagation distance in detector. delta_2=" +
			    std::to_string(delta_2);
			throw std::runtime_error(errorMessage);
		}
#endif

		sig_s_1 = fabsf(n1.scalProd(u));
		sig_s_2 = fabsf(n2.scalProd(v));

		eps_s_1_511 = eps_s_2_511 = util::erfc(tmp511);
		eps_s_1 = eps_s_2 = util::erfc(tmp);
		mu_det = getMuDet((double)energy, crystalMaterial);
		eps_s_1_511 *= 1 - expf(-delta_1 * mu_det_511);
		eps_s_2_511 *= 1 - expf(-delta_2 * mu_det_511);
		eps_s_1 *= 1 - expf(-delta_1 * mu_det);
		eps_s_2 *= 1 - expf(-delta_2 * mu_det);

		fac1 = lamb_s_1 * expf(-att_s_1_511 - att_s_2);
		fac1 *= eps_s_1_511 * eps_s_2;
		fac2 = lamb_s_2 * expf(-att_s_1 - att_s_2_511);
		fac2 *= eps_s_2_511 * eps_s_1;

		res += vatt * dsigcompdomega * (fac1 + fac2) * sig_s_1 * sig_s_2 /
		       (double)(dist1 * dist1 * dist2 * dist2 * 4 * PI);
	}

	u.update(p2 - p1);
	dist1 = u.getNorm();
	u.x /= dist1;
	u.y /= dist1;
	u.z /= dist1;
	sig_s_1 = fabsf(n1.scalProd(u));
	sig_s_2 = fabsf(n2.scalProd(u));
	eps_s_1_511 = eps_s_2_511 = util::erfc(tmp511);
	Vector3D mid{p1.x + p2.x, p1.y + p2.y, p1.z + p2.z};
	mid.x /= 2;
	mid.y /= 2;
	mid.z /= 2;
	lor_1_s.update(p1, mid);
	delta_1 = getIntersectionLengthLORCrystalRaw(lor_1_s, cyl1, cyl2);
	lor_2_s.update(p2, mid);
	delta_2 = getIntersectionLengthLORCrystalRaw(lor_2_s, cyl1, cyl2);
	eps_s_1_511 *= 1 - expf(-delta_1 * mu_det_511);
	eps_s_2_511 *= 1 - expf(-delta_2 * mu_det_511);
	res /= eps_s_1_511 * eps_s_2_511 * sig_s_1 * sig_s_2 /
	       (double)(dist1 * dist1 * 4 * PI);

	return static_cast<float>(res);
}

#ifndef __CUDACC__

// CPU convenience overload: takes Image objects, extracts raw data, and calls
// the unified HOST_DEVICE_CALLABLE version.
inline float computeSingleScatterInLOR(
    const Line3D& lor, float tof_ps, int numSamples, const float* xSamples,
    const float* ySamples, const float* zSamples, float energyLLD,
    float sigmaEnergy, float crystalDepth, float axialFOV,
    float collimatorRadius, CrystalMaterial crystalMaterial,
    const Cylinder& cyl1, const Cylinder& cyl2, const Plane& endPlate1,
    const Plane& endPlate2, const Image& mu, const Image& lambda)
{
	const float* mu_data = mu.getRawPointer();
	const float* lambda_data = lambda.getRawPointer();
	const ImageParams& mu_p = mu.getParams();
	const ImageParams& lambda_p = lambda.getParams();

	RawImageParams mu_params;
	mu_params.nx = mu_p.nx;
	mu_params.ny = mu_p.ny;
	mu_params.nz = mu_p.nz;
	mu_params.vx = mu_p.vx;
	mu_params.vy = mu_p.vy;
	mu_params.vz = mu_p.vz;
	mu_params.length_x = mu_p.length_x;
	mu_params.length_y = mu_p.length_y;
	mu_params.length_z = mu_p.length_z;
	mu_params.off_x = mu_p.off_x;
	mu_params.off_y = mu_p.off_y;
	mu_params.off_z = mu_p.off_z;
	mu_params.fovRadius = mu_p.fovRadius;

	RawImageParams lambda_params;
	lambda_params.nx = lambda_p.nx;
	lambda_params.ny = lambda_p.ny;
	lambda_params.nz = lambda_p.nz;
	lambda_params.vx = lambda_p.vx;
	lambda_params.vy = lambda_p.vy;
	lambda_params.vz = lambda_p.vz;
	lambda_params.length_x = lambda_p.length_x;
	lambda_params.length_y = lambda_p.length_y;
	lambda_params.length_z = lambda_p.length_z;
	lambda_params.off_x = lambda_p.off_x;
	lambda_params.off_y = lambda_p.off_y;
	lambda_params.off_z = lambda_p.off_z;
	lambda_params.fovRadius = lambda_p.fovRadius;

	const float3 imageOffset = {mu_p.off_x, mu_p.off_y, mu_p.off_z};

	return computeSingleScatterInLOR(
	    lor, tof_ps, numSamples, xSamples, ySamples, zSamples, energyLLD,
	    sigmaEnergy, crystalDepth, axialFOV, collimatorRadius, crystalMaterial,
	    cyl1, cyl2, endPlate1, endPlate2, mu_data, lambda_data, mu_params,
	    lambda_params, imageOffset);
}

#endif

}  // namespace yrt::scatter
