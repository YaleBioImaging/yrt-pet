/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/GPUUtils.cuh"

#include "yrt-pet/geometry/Constants.hpp"
#ifdef __CUDACC__
#include "yrt-pet/geometry/Cylinder3D.cuh"
#include "yrt-pet/geometry/Plane3D.cuh"
#else
#include "yrt-pet/geometry/Cylinder3DBase.hpp"
#include "yrt-pet/geometry/Plane3DBase.hpp"
#endif
#include "yrt-pet/recon/RawParameters.hpp"
#include "yrt-pet/scatter/Crystal.hpp"
#include "yrt-pet/utils/Tools.hpp"

#ifndef __CUDACC__
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#endif
#ifdef __CUDACC__
#include "yrt-pet/geometry/Line3D.cuh"
#endif

#if !defined(__CUDA_ARCH__)
#include "yrt-pet/operators/ProjectorSiddon.hpp"
#endif

#if !defined(__CUDACC__)
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>
#endif

#ifdef __CUDACC__
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
    getIntersectionLengthLORCrystalRaw(const Line3D& lor, const Cylinder3D& cyl1,
                                       const Cylinder3D& cyl2)
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
                                                   const Plane3D& endPlate1,
                                                   const Plane3D& endPlate2)
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

// Unified HOST_DEVICE_CALLABLE computeSingleScatterInLOR.
// Uses RawImageConst objects (works on both CPU and GPU).
HOST_DEVICE_CALLABLE inline float computeSingleScatterInLOR(
    const Line3D& lor, float /*tof_ps*/, int numSamples, const float* xSamples,
    const float* ySamples, const float* zSamples, float energyLLD,
    float sigmaEnergy, float crystalDepth, float axialFOV,
    float collimatorRadius, CrystalMaterial crystalMaterial,
    const Cylinder3D& cyl1, const Cylinder3D& cyl2, const Plane3D& endPlate1,
    const Plane3D& endPlate2, RawImageConst muImg, RawImageConst lambdaImg)
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

	RawImageParams muParams = muImg.rawParams;
	const float* muData = muImg.rawPointer;
	RawImageParams lambdaParams = lambdaImg.rawParams;
	const float* lambdaData = lambdaImg.rawPointer;

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
	const float3 muImageOffset =
	    make_float3(muParams.off_x, muParams.off_y, muParams.off_z);
	const float3 lambdaImageOffset =
	    make_float3(lambdaParams.off_x, lambdaParams.off_y, lambdaParams.off_z);
#endif

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
			const float x = ps.x - muParams.off_x;
			const float y = ps.y - muParams.off_y;
			const float z = ps.z - muParams.off_z;
			const float dx = (x + muParams.length_x / 2.0f) /
			                 muParams.length_x *
			                 static_cast<float>(muParams.nx);
			const float dy = (y + muParams.length_y / 2.0f) /
			                 muParams.length_y *
			                 static_cast<float>(muParams.ny);
			const float dz = (z + muParams.length_z / 2.0f) /
			                 muParams.length_z *
			                 static_cast<float>(muParams.nz);
			const ssize_t ix = static_cast<ssize_t>(dx);
			const ssize_t iy = static_cast<ssize_t>(dy);
			const ssize_t iz = static_cast<ssize_t>(dz);
			if (ix < 0 || ix >= muParams.nx || iy < 0 || iy >= muParams.ny ||
			    iz < 0 || iz >= muParams.nz)
				vatt = 0.0f;
			else
				vatt = muData[iz * muParams.nx * muParams.ny +
				              iy * muParams.nx + ix];
		}
		dsigcompdomega = getKleinNishina(cosa);

		// Forward projection for path to detector 1 (mu at 511keV)
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
		{
			float3 p1_f = make_float3(lor_1_s.point1.x, lor_1_s.point1.y,
			                          lor_1_s.point1.z);
			float3 p2_f = make_float3(lor_1_s.point2.x, lor_1_s.point2.y,
			                          lor_1_s.point2.z);
			p1_f -= muImageOffset;
			p2_f -= muImageOffset;
			RawScannerParams scannerParams{};
			projectSiddonDefault<true, false>(
			    att_s_1_511, const_cast<float*>(muData), nullptr, p1_f, p2_f,
			    make_float3(0, 0, 0), make_float3(0, 0, 0), 0, nullptr, 0.0f,
			    scannerParams, muParams, 1, 0);
			att_s_1_511 *= 0.1f;
		}
#else
		{
			ProjectorSiddon::projection<true, false, false>(
			    RawImage{muParams, const_cast<float*>(muData)}, lor_1_s,
			    att_s_1_511);
			att_s_1_511 *= 0.1f;
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
			p1_f -= lambdaImageOffset;
			p2_f -= lambdaImageOffset;
			RawScannerParams scannerParams{};
			projectSiddonDefault<true, false>(
			    lamb_s_1, const_cast<float*>(lambdaData), nullptr, p1_f, p2_f,
			    make_float3(0, 0, 0), make_float3(0, 0, 0), 0, nullptr, 0.0f,
			    scannerParams, lambdaParams, 1, 0);
		}
#else
		{
			ProjectorSiddon::projection<true, false, false>(
			    RawImage{lambdaParams, const_cast<float*>(lambdaData)}, lor_1_s,
			    lamb_s_1);
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
			p1_f -= muImageOffset;
			p2_f -= muImageOffset;
			RawScannerParams scannerParams{};
			projectSiddonDefault<true, false>(
			    att_s_2_511, const_cast<float*>(muData), nullptr, p1_f, p2_f,
			    make_float3(0, 0, 0), make_float3(0, 0, 0), 0, nullptr, 0.0f,
			    scannerParams, muParams, 1, 0);
			att_s_2_511 *= 0.1f;
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

			ProjectorSiddon::projection<true, false, false>(
			    RawImage{muParams, const_cast<float*>(muData)}, lor_2_s,
			    att_s_2_511);
			att_s_2_511 *= 0.1f;
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
			p1_f -= lambdaImageOffset;
			p2_f -= lambdaImageOffset;
			RawScannerParams scannerParams{};
			projectSiddonDefault<true, false>(
			    lamb_s_2, const_cast<float*>(lambdaData), nullptr, p1_f, p2_f,
			    make_float3(0, 0, 0), make_float3(0, 0, 0), 0, nullptr, 0.0f,
			    scannerParams, lambdaParams, 1, 0);
		}
#else
		{
			ProjectorSiddon::projection<true, false, false>(
			    RawImage{lambdaParams, const_cast<float*>(lambdaData)}, lor_2_s,
			    lamb_s_2);
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
//  the unified HOST_DEVICE_CALLABLE version.
inline float computeSingleScatterInLOR(
    const Line3D& lor, float tof_ps, int numSamples, const float* xSamples,
    const float* ySamples, const float* zSamples, float energyLLD,
    float sigmaEnergy, float crystalDepth, float axialFOV,
    float collimatorRadius, CrystalMaterial crystalMaterial,
    const Cylinder3D& cyl1, const Cylinder3D& cyl2, const Plane3D& endPlate1,
    const Plane3D& endPlate2, const Image& mu, const Image& lambda)
{
	const RawImageConst muImg = getRawImage(mu);
	const RawImageConst lambdaImg = getRawImage(lambda);

	return computeSingleScatterInLOR(
	    lor, tof_ps, numSamples, xSamples, ySamples, zSamples, energyLLD,
	    sigmaEnergy, crystalDepth, axialFOV, collimatorRadius, crystalMaterial,
	    cyl1, cyl2, endPlate1, endPlate2, muImg, lambdaImg);
}

#endif

}  // namespace yrt::scatter
