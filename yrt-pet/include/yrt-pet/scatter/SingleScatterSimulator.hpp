/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/scatter/ScatterSpace.hpp"

#include <vector>

namespace yrt
{

class Histogram3D;
class Scanner;
class Image;

namespace scatter
{

class SingleScatterSimulator
{
public:
	// Attenuation in 1/cm
	static constexpr float DefaultAttThresholdSampling = 0.01;
	static constexpr int DefaultSeed = 13;
	// Fraction between 0 and 1
	static constexpr float DefaultNumSampFrac = 2.f / 3.f;
	// Probability between 0 and 1
	static constexpr float DefaultDetectionThreshold = 0.05;

	SingleScatterSimulator(
	    const Scanner& pr_scanner, const Image& pr_mu, const Image& pr_lambda,
	    int p_seed = DefaultSeed,
	    float p_attThresholdSampling = DefaultAttThresholdSampling,
	    float p_numSampFrac = DefaultNumSampFrac,
	    float p_detectionThreshold = DefaultDetectionThreshold);

	void runSSS(ScatterSpace& outScatterSpace,
	            bool onlyDirectPlanes = false) const;

#if BUILD_CUDA
	void runSSSDevice(ScatterSpace& outScatterSpace,
	                  bool onlyDirectPlanes = false,
	                  size_t maxVRAM_bytes = 0ull,
	                  const cudaStream_t* stream0 = nullptr,
	                  const cudaStream_t* stream1 = nullptr) const;
#endif

	float computeSingleScatterInLOR(const Line3D& lor, float tof_ps) const;

	Vector3D getSamplePoint(int i) const;
	int getNumSamples() const;
	const Image& getAttenuationImage() const;

private:
	static float ran1(int* idum);

	// Attenuation image samples
	int m_numSamples;
	std::vector<float> m_xSamples, m_ySamples, m_zSamples;
	// Threshold on the attenuation image to consider a voxel as a potential
	//  scatter point
	float m_attThresholdSampling;
	float m_detectionThreshold;

	float m_energyLLD, m_energyResolution;
	float m_scannerRadius;
	const Scanner& mr_scanner;
	const Image& mr_mu;      // Attenuation image
	const Image& mr_lambda;  // Image from 2 MLEM iterations
};

}  // namespace scatter
}  // namespace yrt
