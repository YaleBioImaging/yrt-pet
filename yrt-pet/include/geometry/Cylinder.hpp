/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/Line3D.hpp"
#include "geometry/Vector3D.hpp"

class Cylinder
{
public:
	Vector3D center;
	float length_z;
	float radius;

public:
	Cylinder();
	Cylinder(const Vector3D& cent, float lz, float r);
	bool doesLineIntersectCylinderInfinite(const Line3D& l, Vector3D& p1,
	                                  Vector3D& p2) const;
	bool doesLineIntersectCylinder(const Line3D& l, Vector3D& p1,
	                         Vector3D& p2) const;
	bool clipLine(Line3D& l) const;
	bool clipLineInfinite(Line3D& l) const;
};
