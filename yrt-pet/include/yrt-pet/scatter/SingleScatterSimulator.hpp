/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/Cylinder.hpp"
#include "yrt-pet/geometry/Plane.hpp"
#include "yrt-pet/scatter/Crystal.hpp"

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
	SingleScatterSimulator(const Scanner& pr_scanner, const Image& pr_mu,
	                       const Image& pr_lambda,
	                       CrystalMaterial p_crystalMaterial, int seedi);

	void runSSS(size_t numberZ, size_t numberPhi, size_t numberR,
	            Histogram3D& scatterHisto);

	float computeSingleScatterInLOR(const Line3D& lor, const Vector3D& n1,
	                                const Vector3D& n2) const;

	Vector3D getSamplePoint(int i) const;
	int getNumSamples() const;
	bool passCollimator(const Line3D& lor) const;

private:
	static float ran1(int* idum);
	static float getKleinNishina(float cosa);
	static float getMuScalingFactor(float energy);

	float getIntersectionLengthLORCrystal(const Line3D& lor) const;

	// Attenuation image samples
	int m_numSamples;
	std::vector<float> m_xSamples, m_ySamples, m_zSamples;
	// Histogram samples
	std::vector<size_t> m_zBinSamples, m_phiSamples, m_rSamples;

	float m_energyLLD, m_sigmaEnergy;
	float m_scannerRadius, m_crystalDepth, m_axialFOV, m_collimatorRadius;
	const Scanner& mr_scanner;
	const Image& mr_mu;      // Attenuation image
	const Image& mr_lambda;  // Image from 2 MLEM iterations
	CrystalMaterial m_crystalMaterial;
	Cylinder m_cyl1, m_cyl2;
	Plane m_endPlate1, m_endPlate2;
};

}  // namespace scatter
}  // namespace yrt
