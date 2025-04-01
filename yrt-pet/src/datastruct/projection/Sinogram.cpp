/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/Sinogram.hpp"
#include "geometry/Constants.hpp"

#include <fstream>
#include <stdexcept>

Sinogram::Sinogram(const std::string& configPath)
{
	// Load JSON configuration
	std::ifstream configFile(configPath);
	if (!configFile)
		throw std::runtime_error("Could not open config file");
	json config;
	configFile >> config;

	// Parse parameters
	m_sinoParams.rMax = config["rMax"];
	m_sinoParams.axialFOV = config["axialFOV"];
	m_sinoParams.thetaMax = config["thetaMax"];
	m_sinoParams.maxTOF = config["maxTOF"];
	m_sinoParams.numR = config["numR"];
	m_sinoParams.numPhi = config["numPhi"];
	m_sinoParams.numZ = config["numZ"];
	m_sinoParams.numTheta = config["numTheta"];
	m_sinoParams.numTOFBins = config["numTOFBins"];
}

float Sinogram::getValue(size_t rIdx, size_t phiIdx, size_t zIdx,
                         size_t thetaIdx, size_t tofIdx) const
{
	const size_t index = tofIdx * (m_sinoParams.numTheta * m_sinoParams.numZ *
	                               m_sinoParams.numPhi * m_sinoParams.numR) +
	                     thetaIdx * (m_sinoParams.numZ * m_sinoParams.numPhi *
	                                 m_sinoParams.numR) +
	                     zIdx * (m_sinoParams.numPhi * m_sinoParams.numR) +
	                     phiIdx * m_sinoParams.numR + rIdx;
	return mp_data->getFlat(index);
}

LORRegion Sinogram::getLORCoordinates(size_t rIdx, size_t phiIdx, size_t zIdx,
                                      size_t thetaIdx, size_t tofIdx) const
{
	// Convert indices to physical values
	const float r = (m_sinoParams.numR > 1) ?
	                    m_sinoParams.rMax * rIdx / (m_sinoParams.numR - 1) :
	                    0.0f;
	const float phi =
	    (180.0f * phiIdx) / (m_sinoParams.numPhi - 1) * PI / 180.0f;
	const float z = -m_sinoParams.axialFOV / 2.0f +
	                m_sinoParams.axialFOV * zIdx / (m_sinoParams.numZ - 1);
	const float theta = (m_sinoParams.thetaMax * thetaIdx) /
	                    (m_sinoParams.numTheta - 1.0f) * PI / 180.0f;
	const float tof =
	    (-m_sinoParams.maxTOF + 2.0f * m_sinoParams.maxTOF * tofIdx) /
	    (m_sinoParams.numTOFBins - 1.0f);

	// Calculate transaxial coordinates
	const float t = std::sqrt(m_sinoParams.rMax * m_sinoParams.rMax - r * r);
	const float x1 = r * std::cos(phi) - t * std::sin(phi);
	const float y1 = r * std::sin(phi) + t * std::cos(phi);
	const float x2 = r * std::cos(phi) + t * std::sin(phi);
	const float y2 = r * std::sin(phi) - t * std::cos(phi);

	// Calculate axial coordinates
	const float transaxialLength = 2 * t;
	const float deltaZ = transaxialLength * std::tan(theta);
	const float zStart = z - deltaZ / 2;
	const float zEnd = z + deltaZ / 2;

	return LORRegion{{x1, y1, zStart, x2, y2, zEnd}, tof};
}

float Sinogram::interpolate(const LORRegion& lor) const
{
	// Convert LOR to parameters
	auto [r, phi, z, theta, tof] = convertLORToParameters(lor);

	// Calculate continuous indices
	float rIdx = (m_sinoParams.numR > 1) ?
	                 (r / m_sinoParams.rMax) * (m_sinoParams.numR - 1) :
	                 0.0f;
	float phiIdx = (phi * 180.0f / PI) / 180.0f * (m_sinoParams.numPhi - 1);
	float zIdx = (z + m_sinoParams.axialFOV / 2.0f) / m_sinoParams.axialFOV *
	             (m_sinoParams.numZ - 1);
	float thetaIdx = (theta * 180.0f / PI) / m_sinoParams.thetaMax *
	                 (m_sinoParams.numTheta - 1);
	float tofIdx = (tof + m_sinoParams.maxTOF) / (2 * m_sinoParams.maxTOF) *
	               (m_sinoParams.numTOFBins - 1);

	rIdx = std::clamp(rIdx, 0.0f, m_sinoParams.numR - 1.0f);
	phiIdx = std::clamp(phiIdx, 0.0f, m_sinoParams.numPhi - 1.0f);
	zIdx = std::clamp(zIdx, 0.0f, m_sinoParams.numZ - 1.0f);
	thetaIdx = std::clamp(thetaIdx, 0.0f, m_sinoParams.numTheta - 1.0f);
	tofIdx = std::clamp(tofIdx, 0.0f, m_sinoParams.numTOFBins - 1.0f);

	// Perform 5D linear interpolation
	return interpolate5D(rIdx, phiIdx, zIdx, thetaIdx, tofIdx);
}

size_t Sinogram::getNumR() const
{
	return m_sinoParams.numR;
}

size_t Sinogram::getNumPhi() const
{
	return m_sinoParams.numPhi;
}

size_t Sinogram::getNumZ() const
{
	return m_sinoParams.numZ;
}

size_t Sinogram::getNumTheta() const
{
	return m_sinoParams.numTheta;
}

size_t Sinogram::getNumTOFBins() const
{
	return m_sinoParams.numTOFBins;
}

float Sinogram::getRMax() const
{
	return m_sinoParams.rMax;
}

float Sinogram::getAxialFOV() const
{
	return m_sinoParams.axialFOV;
}

float Sinogram::getThetaMax() const
{
	return m_sinoParams.thetaMax;
}

float Sinogram::getMaxTOF() const
{
	return m_sinoParams.maxTOF;
}

std::tuple<float, float, float, float, float>
    Sinogram::convertLORToParameters(const LORRegion& sinogramPos) const
{
	float r = std::sqrt(sinogramPos.lor.point1.x * sinogramPos.lor.point1.x +
	                    sinogramPos.lor.point1.y * sinogramPos.lor.point1.y);
	float phi = std::atan2(sinogramPos.lor.point1.y, sinogramPos.lor.point1.x) *
	            180.0f / PI;
	float z = (sinogramPos.lor.point1.z + sinogramPos.lor.point2.z) / 2.0f;
	float theta =
	    std::atan2(sinogramPos.lor.point2.z - sinogramPos.lor.point1.z,
	               m_sinoParams.axialFOV) *
	    180.0 / PI;
	float tof = sinogramPos.tof;
	return {r, phi, z, theta, tof};
}

float Sinogram::interpolate5D(float rIdx, float phiIdx, float zIdx,
                              float thetaIdx, float tofIdx) const
{
	// Get neighboring indices and weights
	const auto [r0, r1, wr] = getNeighbors(rIdx, m_sinoParams.numR);
	const auto [phi0, phi1, wphi] = getNeighbors(phiIdx, m_sinoParams.numPhi);
	const auto [z0, z1, wz] = getNeighbors(zIdx, m_sinoParams.numZ);
	const auto [theta0, theta1, wtheta] =
	    getNeighbors(thetaIdx, m_sinoParams.numTheta);
	const auto [tof0, tof1, wtof] =
	    getNeighbors(tofIdx, m_sinoParams.numTOFBins);

	float sum = 0.0f;
	for (const int dr : {0, 1})
	{
		for (const int dphi : {0, 1})
		{
			for (const int dz : {0, 1})
			{
				for (const int dtheta : {0, 1})
				{
					for (const int dtof : {0, 1})
					{
						const size_t ri =
						    std::min(r0 + dr, m_sinoParams.numR - 1);
						const size_t phii =
						    std::min(phi0 + dphi, m_sinoParams.numPhi - 1);
						const size_t zi =
						    std::min(z0 + dz, m_sinoParams.numZ - 1);
						const size_t thetai = std::min(
						    theta0 + dtheta, m_sinoParams.numTheta - 1);
						const size_t tofi =
						    std::min(tof0 + dtof, m_sinoParams.numTOFBins - 1);

						const float weight =
						    ((dr ? wr : 1 - wr) * (dphi ? wphi : 1 - wphi) *
						     (dz ? wz : 1 - wz) *
						     (dtheta ? wtheta : 1 - wtheta) *
						     (dtof ? wtof : 1 - wtof));

						sum += getValue(ri, phii, zi, thetai, tofi) * weight;
					}
				}
			}
		}
	}
	return sum;
}

std::tuple<size_t, size_t, float> Sinogram::getNeighbors(float idx, size_t max)
{
	if (max == 0)
		throw std::runtime_error("Invalid dimension size");
	const auto i0 = static_cast<size_t>(idx);
	size_t i1 = std::min(i0 + 1, max - 1);
	float frac = idx - i0;
	return {i0, i1, frac};
}

SinogramOwned::SinogramOwned(const std::string& configPath,
                             const std::string& dataPath)
    : Sinogram(configPath)
{
	auto data = std::make_unique<Array5D<float>>();
	data->readFromFile(dataPath, {m_sinoParams.numR, m_sinoParams.numPhi,
	                              m_sinoParams.numZ, m_sinoParams.numTheta,
	                              m_sinoParams.numTOFBins});
	mp_data = std::move(data);
}
