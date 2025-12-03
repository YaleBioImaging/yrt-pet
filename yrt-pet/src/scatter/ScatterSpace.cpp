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
	mp_values->allocate(m_numTOFBins, m_numPlanes, m_numAngles,
	                           m_numPlanes, m_numAngles);
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

float ScatterSpace::getTOF_mm(size_t TOFBin) const
{
	return m_TOFBinStep_mm * (static_cast<float>(TOFBin) + 0.5f);
}

float ScatterSpace::getPlanePosition(size_t planeIndex) const
{
	return m_planeStep * (static_cast<float>(planeIndex) + 0.5f);
}

float ScatterSpace::getAngle(size_t angleIndex) const
{
	return m_angleStep * (static_cast<float>(angleIndex) + 0.5f);
}

float ScatterSpace::getTOFBinStep_mm() const
{
	return m_TOFBinStep_mm;
}

float ScatterSpace::getAngleStep() const
{
	return m_angleStep;
}

float ScatterSpace::getPlaneStep() const
{
	return m_planeStep;
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

void ScatterSpace::initStepSizes()
{
	m_TOFBinStep_mm = getRadius() / static_cast<float>(m_numTOFBins);
	m_angleStep = TWOPI_FLT / static_cast<float>(m_numAngles);
	m_planeStep = getAxialFOV() / static_cast<float>(m_numPlanes);
}

float ScatterSpace::getAxialFOV() const
{
	return mr_scanner.axialFOV;
}

size_t ScatterSpace::getNumPlanes() const
{
	return m_numPlanes;
}

size_t ScatterSpace::getNumAngles() const
{
	return m_numAngles;
}

float ScatterSpace::getRadius() const
{
	return mr_scanner.scannerRadius;
}

}  // namespace yrt
