/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/ScatterSpace.hpp"

#include "yrt-pet/geometry/Constants.hpp"

namespace yrt
{


ScatterSpace::ScatterSpace(const Scanner& pr_scanner, const std::string& fname)
    : Histogram(pr_scanner), mp_values(nullptr)
{
	readFromFile(fname);

	const auto dims = mp_values->getDims();

	m_numTOFBins = dims[0];
	m_numPlanes = dims[1];
	m_numAngles = dims[2];

	initStepSizes();
}

ScatterSpace::ScatterSpace(const Scanner& pr_scanner, size_t p_numTOFBins,
                           size_t p_numPlanes, size_t p_numAngles)
    : Histogram(pr_scanner),
      m_numTOFBins(p_numTOFBins),
      m_numPlanes(p_numPlanes),
      m_numAngles(p_numAngles)
{
	ASSERT_MSG(m_numTOFBins > 0, "Number of TOF bins must be non-null");
	ASSERT_MSG(m_numPlanes > 0, "Number of planes must be non-null");
	ASSERT_MSG(m_numAngles > 1, "Number of angles must be more than 1");

	initStepSizes();

	// Allocate the scatter space
	mp_values = std::make_unique<Array5D<float>>();
	mp_values->allocate(m_numTOFBins, m_numPlanes, m_numAngles, m_numPlanes,
	                    m_numAngles);
	mp_values->fill(0.0f);
}

void ScatterSpace::readFromFile(const std::string& fname)
{
	if (mp_values == nullptr)
	{
		mp_values = std::make_unique<Array5D<float>>();
	}
	mp_values->readFromFile(fname);
}

void ScatterSpace::writeToFile(const std::string& fname) const
{
	mp_values->writeToFile(fname);
}

ScatterSpace::ScatterSpaceIndex
    ScatterSpace::nearestNeighbor(const ScatterSpacePosition& pos) const
{

	// Wrap angles to [0, 2pi)
    float a1 = wrapAngle(pos.angle1);
    float a2 = wrapAngle(pos.angle2);

    // Clamp z coordinates to axial FOV
    float z1 = clampPlanePosition(pos.planePosition1);
    float z2 = clampPlanePosition(pos.planePosition2);

    // Convert to continuous grid indices
    float a1_idx = a1 / angle_step_;
    float z1_idx = (z1 - z_min_) / z_step_;
    float a2_idx = a2 / angle_step_;
    float z2_idx = (z2 - z_min_) / z_step_;

    // Get integer parts and fractional parts
    int a1_i = static_cast<int>(std::floor(a1_idx));
    int z1_i = static_cast<int>(std::floor(z1_idx));
    int a2_i = static_cast<int>(std::floor(a2_idx));
    int z2_i = static_cast<int>(std::floor(z2_idx));

    float a1_frac = a1_idx - a1_i;
    float z1_frac = z1_idx - z1_i;
    float a2_frac = a2_idx - a2_i;
    float z2_frac = z2_idx - z2_i;

    // Wrap angle indices (periodic boundary)
    uint16_t a1_i0 = wrapAngleIndex(a1_i);
    uint16_t a1_i1 = wrapAngleIndex(a1_i + 1);
    uint16_t a2_i0 = wrapAngleIndex(a2_i);
    uint16_t a2_i1 = wrapAngleIndex(a2_i + 1);

    // Clamp z indices (non-periodic boundary)
    uint16_t z1_i0 = clampZIndex(z1_i);
    uint16_t z1_i1 = clampZIndex(z1_i + 1);
    uint16_t z2_i0 = clampZIndex(z2_i);
    uint16_t z2_i1 = clampZIndex(z2_i + 1);

    // Get the 16 neighboring grid points
    float v0000 = getValue(a1_i0, z1_i0, a2_i0, z2_i0);
    float v0001 = getValue(a1_i0, z1_i0, a2_i0, z2_i1);
    float v0010 = getValue(a1_i0, z1_i0, a2_i1, z2_i0);
    float v0011 = getValue(a1_i0, z1_i0, a2_i1, z2_i1);

    float v0100 = getValue(a1_i0, z1_i1, a2_i0, z2_i0);
    float v0101 = getValue(a1_i0, z1_i1, a2_i0, z2_i1);
    float v0110 = getValue(a1_i0, z1_i1, a2_i1, z2_i0);
    float v0111 = getValue(a1_i0, z1_i1, a2_i1, z2_i1);

    float v1000 = getValue(a1_i1, z1_i0, a2_i0, z2_i0);
    float v1001 = getValue(a1_i1, z1_i0, a2_i0, z2_i1);
    float v1010 = getValue(a1_i1, z1_i0, a2_i1, z2_i0);
    float v1011 = getValue(a1_i1, z1_i0, a2_i1, z2_i1);

    float v1100 = getValue(a1_i1, z1_i1, a2_i0, z2_i0);
    float v1101 = getValue(a1_i1, z1_i1, a2_i0, z2_i1);
    float v1110 = getValue(a1_i1, z1_i1, a2_i1, z2_i0);
    float v1111 = getValue(a1_i1, z1_i1, a2_i1, z2_i1);

    // 4D linear interpolation (multilinear)
    // Interpolate along z2 direction first
    float v000 = v0000 * (1 - z2_frac) + v0001 * z2_frac;
    float v001 = v0010 * (1 - z2_frac) + v0011 * z2_frac;
    float v010 = v0100 * (1 - z2_frac) + v0101 * z2_frac;
    float v011 = v0110 * (1 - z2_frac) + v0111 * z2_frac;
    float v100 = v1000 * (1 - z2_frac) + v1001 * z2_frac;
    float v101 = v1010 * (1 - z2_frac) + v1011 * z2_frac;
    float v110 = v1100 * (1 - z2_frac) + v1101 * z2_frac;
    float v111 = v1110 * (1 - z2_frac) + v1111 * z2_frac;

    // Then along a2 direction
    float v00 = v000 * (1 - a2_frac) + v001 * a2_frac;
    float v01 = v010 * (1 - a2_frac) + v011 * a2_frac;
    float v10 = v100 * (1 - a2_frac) + v101 * a2_frac;
    float v11 = v110 * (1 - a2_frac) + v111 * a2_frac;

    // Then along z1 direction
    float v0 = v00 * (1 - z1_frac) + v01 * z1_frac;
    float v1 = v10 * (1 - z1_frac) + v11 * z1_frac;

    // Finally along a1 direction
    return v0 * (1 - a1_frac) + v1 * a1_frac;
}

float ScatterSpace::getTOF_ps(size_t TOFBin) const
{
	return m_TOFBinStep_ps * (static_cast<float>(TOFBin) + 0.5f);
}

float ScatterSpace::getPlanePosition(size_t planeIndex) const
{
	return m_planeStep * (static_cast<float>(planeIndex) + 0.5f);
}

float ScatterSpace::getAngle(size_t angleIndex) const
{
	return m_angleStep * (static_cast<float>(angleIndex) + 0.5f);
}

float ScatterSpace::getTOFBinStep_ps() const
{
	return m_TOFBinStep_ps;
}

float ScatterSpace::getAngleStep() const
{
	return m_angleStep;
}

float ScatterSpace::getPlaneStep() const
{
	return m_planeStep;
}

float ScatterSpace::getValue(const ScatterSpaceIndex& idx) const
{
	return getValue(idx.tofBin, idx.planeIndex1, idx.angleIndex1,
	                idx.planeIndex2, idx.angleIndex2);
}

float ScatterSpace::getValue(size_t tofBin, size_t planeIndex1,
                             size_t angleIndex1, size_t planeIndex2,
                             size_t angleIndex2) const
{
	return mp_values->get(
	    {tofBin, planeIndex1, angleIndex1, planeIndex2, angleIndex2});
}

void ScatterSpace::symmetrize()
{
	// TODO NOW: Here, make it so that for all elements of this scatter space,
	//  the value at ((a1,z1), (a2,z2), tof) is the same as for
	//  ((a2,z2), (a1,z1), tof)
}

float ScatterSpace::wrapAngle(float angle)
{
	// Wrap to [0, 2pi)
	angle = std::fmod(angle, TWOPI_FLT);
	if (angle < 0)
	{
		angle += TWOPI_FLT;
	}
	return angle;
}

float ScatterSpace::clampPlanePosition(float planePosition) const
{
	const float maxPlanePosition = getAxialFOV() / 2 + getPlaneStep() / 2;
	const float minPlanePosition = -maxPlanePosition;
	return std::clamp(planePosition, minPlanePosition, maxPlanePosition);
}

float ScatterSpace::clampTOF(float tof_ps) const
{
	const float maxTOF = getMaxTOF_ps() + getPlaneStep() / 2;
	const float minTOF = -maxPlanePosition;
	return std::clamp(planePosition, minPlanePosition, maxPlanePosition);
}

void ScatterSpace::initStepSizes()
{
	m_TOFBinStep_ps = getMaxTOF_ps() / static_cast<float>(m_numTOFBins);
	m_angleStep = TWOPI_FLT / static_cast<float>(m_numAngles);
	m_planeStep = getAxialFOV() / static_cast<float>(m_numPlanes);
}

float ScatterSpace::getAxialFOV() const
{
	return mr_scanner.axialFOV;
}

float ScatterSpace::getRadius() const
{
	return mr_scanner.scannerRadius;
}

float ScatterSpace::getDiameter() const
{
	return getRadius() * 2;
}

float ScatterSpace::getMaxTOF_ps() const
{
	return getDiameter() / SPEED_OF_LIGHT_MM_PS_FLT;
}

size_t ScatterSpace::getNumPlanes() const
{
	return m_numPlanes;
}

size_t ScatterSpace::getNumAngles() const
{
	return m_numAngles;
}

}  // namespace yrt
