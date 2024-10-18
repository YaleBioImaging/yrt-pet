/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/Cylinder.hpp"

#include "geometry/Constants.hpp"
#include "geometry/Vector3D.hpp"

#include <cmath>

Cylinder::Cylinder() : center{}, length_z{0.}, radius{0.} {}

// constructors:
Cylinder::Cylinder(const Vector3D& cent, float lz, float r)
    : center{cent}, length_z{lz}, radius{r}
{
}

bool Cylinder::doesLineIntersectCylinderInfinite(const Line3D& l, Vector3D& p1,
                                                 Vector3D& p2) const
{
	const float lb = l.point1.x;
	const float ld = l.point1.y;
	const float lf = l.point1.z;
	const float la = l.point2.x - lb;
	const float lc = l.point2.y - ld;
	const float le = l.point2.z - lf;

	const float a = la, b = lb, c = lc, d = ld, e = le, f = lf;
	const float A = a * a + c * c;
	const float B = 2 * (a * (b - center.x) + c * (d - center.y));
	const float C =
	    GET_SQ(b - center.x) + GET_SQ(d - center.y) - GET_SQ(radius);

	const float delta = B * B - 4 * A * C;
	if (delta < 0)
	{
		return false;
	}

	const float t1 = (-B - sqrt(delta)) / (2 * A);
	const float t2 = (-B + sqrt(delta)) / (2 * A);

	p1.x = a * t1 + b;
	p1.y = c * t1 + d;
	p1.z = e * t1 + f;

	p2.x = a * t2 + b;
	p2.y = c * t2 + d;
	p2.z = e * t2 + f;
	return true;
}

bool Cylinder::doesLineIntersectCylinder(const Line3D& l, Vector3D& p1,
                                         Vector3D& p2) const
{
	const bool infinite_ret = doesLineIntersectCylinderInfinite(l, p1, p2);
	if (!infinite_ret)
	{
		return false;
	}
	if (std::abs(p1.z - center.z) > length_z / 2 ||
	    std::abs(p2.z - center.z) > length_z / 2)
	{
		return false;
	}
	return true;
}
bool Cylinder::clipLine(Line3D& l) const
{
	Vector3D p1, p2;
	if (!doesLineIntersectCylinder(l, p1, p2))
		return false;
	l.update(p1, p2);
	return true;
}

bool Cylinder::clipLineInfinite(Line3D& l) const
{
	Vector3D p1, p2;
	if (!doesLineIntersectCylinderInfinite(l, p1, p2))
		return false;
	l.update(p1, p2);
	return true;
}
