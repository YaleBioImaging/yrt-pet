/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cmath>

#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/geometry/Vector3D.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"

namespace yrt
{
class Cylinder
{
public:
	Vector3D center;
	float length_z;
	float radius;

public:
	HOST_DEVICE_CALLABLE Cylinder() : center{}, length_z{0}, radius{0} {}

	HOST_DEVICE_CALLABLE Cylinder(const Vector3D& cent, float lz, float r)
	    : center{cent}, length_z{lz}, radius{r}
	{
	}

	HOST_DEVICE_CALLABLE bool doesLineIntersectCylinderInfinite(
	    const Line3D& l, Vector3D& p1, Vector3D& p2) const
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

	HOST_DEVICE_CALLABLE bool doesLineIntersectCylinder(
	    const Line3D& l, Vector3D& p1, Vector3D& p2) const
	{
		const bool infinite_ret = doesLineIntersectCylinderInfinite(l, p1, p2);
		if (!infinite_ret)
		{
			return false;
		}
		const auto abs_z1 = p1.z >= center.z ? p1.z - center.z
		                                     : center.z - p1.z;
		const auto abs_z2 = p2.z >= center.z ? p2.z - center.z
		                                     : center.z - p2.z;
		if (abs_z1 > length_z / 2 || abs_z2 > length_z / 2)
		{
			return false;
		}
		return true;
	}

	HOST_DEVICE_CALLABLE bool clipLine(Line3D& l) const
	{
		Vector3D p1, p2;
		if (!doesLineIntersectCylinder(l, p1, p2))
			return false;
		l.update(p1, p2);
		return true;
	}

	HOST_DEVICE_CALLABLE bool clipLineInfinite(Line3D& l) const
	{
		Vector3D p1, p2;
		if (!doesLineIntersectCylinderInfinite(l, p1, p2))
			return false;
		l.update(p1, p2);
		return true;
	}
};
}  // namespace yrt
