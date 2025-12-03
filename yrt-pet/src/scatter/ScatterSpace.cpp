/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/ScatterSpace.hpp"

#include "yrt-pet/geometry/Constants.hpp"

namespace yrt
{

ScatterSpace::ScatterSpace(const Scanner& pr_scanner, size_t p_numPlanes,
                           size_t p_numAngles)
    : Histogram(pr_scanner), m_numPlanes(p_numPlanes), m_numAngles(p_numAngles)
{
}

float ScatterSpace::getPlanePosition(size_t planeIndex) const
{
	const float axialFOV = getAxialFOV();
	const float planeSpacing = axialFOV / getNumPlanes();
	return planeIndex * planeSpacing + planeSpacing * 0.5f;
}

float ScatterSpace::getAngle(size_t angleIndex) const
{
	constexpr float fullCircle = TWOPI_FLT;
	const float angleSpacing = fullCircle / getNumAngles();
	return angleIndex * angleSpacing + angleSpacing * 0.5f;
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

}  // namespace yrt
