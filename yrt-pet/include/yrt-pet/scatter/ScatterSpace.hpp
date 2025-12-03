/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/Histogram.hpp"

namespace yrt
{

class ScatterSpace : public Histogram
{
public:
	ScatterSpace(const Scanner& pr_scanner, size_t p_numPlanes, size_t p_numAngles);

	float getAxialFOV() const;
	size_t getNumPlanes() const;
	size_t getNumAngles() const;

	float getPlanePosition(size_t planeIndex) const;
	float getAngle(size_t angleIndex) const; // In radians

protected:
	size_t m_numPlanes;
	size_t m_numAngles;

};

}
