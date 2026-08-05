/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageUtils.cuh"
#include "yrt-pet/utils/GPUUtils.cuh"

#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/recon/RawParameters.hpp"
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
//  scatter point.
HOST_DEVICE_CALLABLE inline float computeRaySum(const RawImageConst& img,
                                                const Line3D& ray)
{
	float raySum = 0;
	const auto params = img.rawParams;
	const float* imageData = img.rawPointer;

	const ssize_t nx = params.nx;
	const ssize_t ny = params.ny;
	const ssize_t nz = params.nz;

	// Virtual detector location
	const float xd = ray.point1.x;
	const float yd = ray.point1.y;
	const float zd = ray.point1.z;

	// Get the image indices corresponding to the scatter point location
	// This function assumes that the "point2" of the "ray" given is inside the
	//  image, since it is the scatter point.
	const float xs = ray.point2.x;
	const float ys = ray.point2.y;
	const float zs = ray.point2.z;
	const ssize_t jxs = rintf(util::positionToIndex(
	    ray.point2.x, params.vx, params.length_x, params.off_x));
	const ssize_t jys = rintf(util::positionToIndex(
	    ray.point2.y, params.vy, params.length_y, params.off_y));
	const ssize_t jzs = rintf(util::positionToIndex(
	    ray.point2.z, params.vz, params.length_z, params.off_z));

	// Compute the direction cosines (l,m,n) for the ray connecting the two
	// locations
	const float d = sqrt(GET_SQ(xd - xs) + GET_SQ(yd - ys) + GET_SQ(zd - zs));
	const float l = (xd - xs) / d;
	const float m = (yd - ys) / d;
	const float n = (zd - zs) / d;

	// Distance to step in ray sum
	const float delta = fminf(fminf(params.vx, params.vy), params.vz);

	// Begin the ray sum
	float x = xs;
	float y = ys;
	float z = zs;
	ssize_t jx = jxs;
	ssize_t jy = jys;
	ssize_t jz = jzs;

	while ((jx >= 0) && (jx < nx) && (jy >= 0) && (jy < ny) && (jz >= 0) &&
	       (jz < nz))
	{
		const ssize_t flatIdx = jz * nx * ny + jy * nx + jx;
		raySum += imageData[flatIdx] * delta;

		x += l * delta;
		y += m * delta;
		z += n * delta;

		jx = rintf(
		    util::positionToIndex(x, params.vx, params.length_x, params.off_x));
		jy = rintf(
		    util::positionToIndex(y, params.vy, params.length_y, params.off_y));
		jz = rintf(
		    util::positionToIndex(z, params.vz, params.length_z, params.off_z));
	}

	return raySum;
}

HOST_DEVICE_CALLABLE inline float probDetect(float energyLLD,
                                             float energyResolution,
                                             float photonEnergy,
                                             float energyBackground = 0.0f)
{
	// These usually come from the scanner parameters
	const float lld = energyLLD;
	const float fwhm = energyResolution * sqrt(511.0 * photonEnergy);
	const float sig = fwhm / SIGMA_TO_FWHM_FLT;
	const float background = energyBackground;

	// Assume the energy measured has a Gaussian distribution with mean
	// photonEnergy and FWHM. Then, the probability of detection (i.e. measured
	// as above lld) is the integral from lld to infinity of a Gaussian with
	// mean "photonEnergy" and standard deviation "sig"

	// Variables:
	//  - $p_e$ : photonEnergy
	//  - $\sigma$ : sig
	//  - $E_{LLD}$ : lld
	// The integral being solved is:
	// $ \int_{E_{LLD}}^{\infty}\frac{1}{\sqrt{2\pi}\sigma}
	//   \exp({-\frac{(x-p_e)^2}{2\sigma^2}})dx $

	// Substitution: $y = \frac{x - p_e}{\sqrt{2}\sigma}$
	//  Evaluates to: $\frac{1}{\sqrt{\pi}}\int_l^\infty \exp(-y^2)dy$
	//  where lower bound is $l = \frac{E_{LLD} - p_e}{\sqrt{2}\sigma}$
	const float l = (lld - photonEnergy) / (sqrt(2.0f) * sig);

	// Error functions:
	//  - $\text{erfc}(l) = \frac{2}{\sqrt{\pi}} \int_l^\infty \exp(-t^2)dt$
	//  - $\text{erf}(l) = \frac{2}{\sqrt{\pi}} \int_0^l \exp(-t^2)dt$
	// If $l > 0$: probability is $\frac{1}{2} \text{erfc}(l)$.
	// If $l < 0$: probability is $\frac{1}{2} (\text{erf}(-l) + 1)$.

	// The final result includes the energy background weighting. This might be
	//  used in the future
	if (l > 0.0)
	{
		return 0.5f * ((1.0f - background) * erfc(l)) + background;
	}
	return 0.5f * ((1.0f - background) * (erf(-l) + 1.0f)) + background;
}

HOST_DEVICE_CALLABLE inline float getAttenScaleFactor(float scatterAngleCosine)
{
	// Models the difference of attenuation that lower energy photons are
	//  subjected to.
	// Lower energy photons are more likely to be attenuated. This function
	//  scales the mu-map accordingly.
	constexpr float attenScaleParam0 = 1.3167f;
	constexpr float attenScaleParam1 = -0.2371f;
	constexpr float attenScaleParam2 = -0.0784f;
	return attenScaleParam0 + attenScaleParam1 * scatterAngleCosine +
	       attenScaleParam2 * GET_SQ(scatterAngleCosine);
}

HOST_DEVICE_CALLABLE inline float getDetectionEff(float scatterAngleCosine)
{
	// Lower energy photons are more likely to be attenuated by the crystal
	//  itself. This function scales the attenuation probability accordingly.
	// TODO: These values change depending on the crystal material.
	//  It should not be hard-coded.
	constexpr float detectionEffParam0 = 1.57266f;
	constexpr float detectionEffParam1 = -0.342749f;
	constexpr float detectionEffParam2 = -0.229914f;
	return detectionEffParam0 + detectionEffParam1 * scatterAngleCosine +
	       detectionEffParam2 * GET_SQ(scatterAngleCosine);
}

// Trace a line through the image and stop whenever the image has voxels with a
//  value higher than 'threshold'.
HOST_DEVICE_CALLABLE inline bool
    doesLineIntersectImageThreshold(const Line3D& line,
                                    const RawImageConst& image, float threshold)
{
	const Vector3D& p1 = line.point1;
	const Vector3D& p2 = line.point2;
	const Vector3D pDiff = p1 - p2;
	const RawImageParams& params = image.rawParams;
	const ssize_t nx = params.nx;
	const ssize_t ny = params.ny;
	const ssize_t nz = params.nz;
	const float* imageData = image.rawPointer;

	Vector3D pIdx1, pIdx2;  // The same points (p1 and p2) but in image indices
	pIdx1.x = rintf(
	    util::positionToIndex(p1.x, params.vx, params.length_x, params.off_x));
	pIdx1.y = rintf(
	    util::positionToIndex(p1.y, params.vy, params.length_y, params.off_y));
	pIdx1.z = rintf(
	    util::positionToIndex(p1.z, params.vz, params.length_z, params.off_z));
	pIdx2.x = rintf(
	    util::positionToIndex(p2.x, params.vx, params.length_x, params.off_x));
	pIdx2.y = rintf(
	    util::positionToIndex(p2.y, params.vy, params.length_y, params.off_y));
	pIdx2.z = rintf(
	    util::positionToIndex(p2.z, params.vz, params.length_z, params.off_z));

	// Direction cosines (l,m,n) for the line
	float distance1 = GET_SQ(pDiff.x) + GET_SQ(pDiff.y) + GET_SQ(pDiff.z);
	float distance2 = 0.0;
	const float d = sqrt(distance1);
	const float l = (p2.x - p1.x) / d;
	const float m = (p2.y - p1.y) / d;
	const float n = (p2.z - p1.z) / d;

	// Begin the ray tracing
	constexpr float delta = 1.0;  // distance to step in ray sum
	ssize_t jx = pIdx1.x;
	ssize_t jy = pIdx1.y;
	ssize_t jz = pIdx1.z;

	// Iterators
	ssize_t i = 0;

	// Iterate through the line
	while ((jx != pIdx2.x) || (jy != pIdx2.y) || (jz != pIdx2.z))
	{
		if ((jx >= 0 && jx < nx) && (jy >= 0 && jy < ny) &&
		    (jz >= 0 && jz < nz))
		{
			const ssize_t flatIdx = jz * nx * ny + jy * nx + jx;

			if (imageData[flatIdx] > threshold)
			{
				return true;
			}
		}

		++i;

		// Current physical position (mm)
		const float x = p1.x + i * l * delta;
		const float y = p1.y + i * m * delta;
		const float z = p1.z + i * n * delta;

		distance2 = GET_SQ(x - p2.x) + GET_SQ(y - p2.y) + GET_SQ(z - p2.z);
		// If the distance from the current point to point2 starts increasing
		//  again, then we have reached beyond point2
		if (distance2 > distance1)
		{
			// Reached end of the line
			return false;
		}
		distance1 = distance2;

		// Current position in image indices (voxels)
		const ssize_t indexX = rintf(
		    util::positionToIndex(x, params.vx, params.length_x, params.off_x));
		const ssize_t indexY = rintf(
		    util::positionToIndex(y, params.vy, params.length_y, params.off_y));
		const ssize_t indexZ = rintf(
		    util::positionToIndex(z, params.vz, params.length_z, params.off_z));

		jx = indexX;
		jy = indexY;
		jz = indexZ;
	}
	// Reached end of the image
	return false;
}

// Uses RawImageConst objects (works on both CPU and GPU).
HOST_DEVICE_CALLABLE inline float computeSingleScatterInLOR(
    const Line3D& lor, float /*tof_ps*/, int numSamples, const float* xSamples,
    const float* ySamples, const float* zSamples, float detectionThreshold,
    float energyLLD, float energyResolution, RawImageConst muImg,
    RawImageConst lambdaImg)
{
	// Total scatter estimate associated to the current LOR
	double lorScatterEstimate = 0.0;

	Line3D lineAS, lineBS;
	Vector3D pS, vAS, vBS;
	const Vector3D pA = lor.point1;
	const Vector3D pB = lor.point2;

	RawImageParams muParams = muImg.rawParams;
	const float* muData = muImg.rawPointer;

	for (int i = 0; i < numSamples; i++)
	{
		pS.update(xSamples[i], ySamples[i], zSamples[i]);

		// Get the attenuation at the current scatter point
		ssize_t iS, jS, kS;
		bool scatterPointIsInside = util::getNearestNeighborIdx(
		    pS.x, pS.y, pS.z, muParams.length_x, muParams.length_y,
		    muParams.length_z, muParams.off_x, muParams.off_y, muParams.off_z,
		    muParams.nx, muParams.ny, muParams.nz, muParams.nt, &iS, &jS, &kS);
		if (!scatterPointIsInside)
		{
			// This should never happen
			continue;
		}
		size_t flatIdx = iS + jS * muParams.nx + kS * muParams.nx * muParams.ny;
		float curLocalMu = muData[flatIdx] * 0.1f;

		// Segments AS and BS
		lineAS.update(pA, pS);
		lineBS.update(pB, pS);

		// Difference vector
		vAS.update(pS - pA);
		vBS.update(pS - pB);

		float numerator = vAS.scalProd(vBS);

		if (numerator > 0.0)
		{
			// Ignore cases where the photon scattered by more than 90 degrees
			continue;
		}

		float rAS2 = vAS.getNormSquared();
		float rBS2 = vBS.getNormSquared();
		float rAS = sqrt(rAS2);
		float rBS = sqrt(rBS2);
		float denominator = rAS * rBS;
		float scatterAngleCosine = fabs(numerator / denominator);

		float photonEnergy = 511.0f / (2.0f - scatterAngleCosine);
		float scatterEnergyEff =
		    probDetect(energyLLD, energyResolution, photonEnergy);

		// Exclude scatters that have a detection probability below the
		//  detection threshold
		if (scatterEnergyEff < detectionThreshold)
		{
			continue;
		}

		float attenScaleFactor = getAttenScaleFactor(scatterAngleCosine);
		float emissionRaySumAS = computeRaySum(lambdaImg, lineAS);
		float emissionRaySumBS = computeRaySum(lambdaImg, lineBS);
		float muRaySumAS = computeRaySum(muImg, lineAS);
		float muRaySumBS = computeRaySum(muImg, lineBS);
		float scatterMuRaySumAS = muRaySumAS * attenScaleFactor;
		float scatterMuRaySumBS = muRaySumBS * attenScaleFactor;
		scatterEnergyEff *= getDetectionEff(scatterAngleCosine);

		// Here, we could add, as a numerator, the crystal area. However, this
		//  would only amount to a difference of scaling in the SSS estimation
		//  that would be erased by the tail-fitting
		float unscaledGeometricCrossSectionAS = 1.0f / rAS2;
		float unscaledGeometricCrossSectionBS = 1.0f / rBS2;

		// Compute differential cross section for 511 keV photons at this
		//  scatter angle (Klein Nishina)
		float diffCrossSection = getKleinNishina(scatterAngleCosine);

		// Compute the total photon flux at A and B due to scatter point S
		float Ias = util::getAttenuationCoefficientFactor(muRaySumAS +
		                                                  scatterMuRaySumBS) *
		            emissionRaySumAS;
		float Ibs = util::getAttenuationCoefficientFactor(scatterMuRaySumAS +
		                                                  muRaySumBS) *
		            emissionRaySumBS;
		float Iasb = scatterEnergyEff * (Ias + Ibs);

		// Compute the scatter contribution from this ASB combination
		float bigFactor =
		    unscaledGeometricCrossSectionAS * unscaledGeometricCrossSectionBS;
		bigFactor *= curLocalMu * diffCrossSection;
		// The Compton scatter coefficient is ignored since the tail-fitting
		//  would replace it. Similarly, here "bigFactor" could be divided by
		//  $4\pi$, but isn't.

		double sampleScatterEstimate = bigFactor * Iasb;
		lorScatterEstimate += sampleScatterEstimate;
	}

	return static_cast<float>(lorScatterEstimate);
}

#ifndef __CUDACC__

// CPU convenience overload: takes Image objects, extracts raw data, and calls
//  the unified HOST_DEVICE_CALLABLE version.
inline float computeSingleScatterInLOR(const Line3D& lor, float tof_ps,
                                       int numSamples, const float* xSamples,
                                       const float* ySamples,
                                       const float* zSamples,
                                       float detectionThreshold,
                                       float energyLLD, float energyResolution,
                                       const Image& mu, const Image& lambda)
{
	const RawImageConst muImg = getRawImage(mu);
	const RawImageConst lambdaImg = getRawImage(lambda);

	return computeSingleScatterInLOR(
	    lor, tof_ps, numSamples, xSamples, ySamples, zSamples,
	    detectionThreshold, energyLLD, energyResolution, muImg, lambdaImg);
}

#endif

}  // namespace yrt::scatter
