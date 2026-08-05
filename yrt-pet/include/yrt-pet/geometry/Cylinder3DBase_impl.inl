/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 *
 * Implementation for Cylinder3DBase. Included by Cylinder3D.cpp (CPU) and
 * Cylinder3D.cuh (GPU). Do not include directly.
 */

#pragma once

#include <cmath>

#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/geometry/Line3D_impl.inl"

namespace yrt
{

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool Cylinder3DBase<TFloat>::isValid() const
{
	return length_z > static_cast<TFloat>(0) &&
	       radius > static_cast<TFloat>(0);
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool
    Cylinder3DBase<TFloat>::doesLineIntersectCylinderInfinite(
        const Line3DBase<TFloat>& l, Vector3DBase<TFloat>& p1,
        Vector3DBase<TFloat>& p2) const
{
	const TFloat lb = l.point1.x;
	const TFloat ld = l.point1.y;
	const TFloat lf = l.point1.z;
	const TFloat la = l.point2.x - lb;
	const TFloat lc = l.point2.y - ld;
	const TFloat le = l.point2.z - lf;

	const TFloat a = la, b = lb, c = lc, d = ld, e = le, f = lf;
	const TFloat A = a * a + c * c;
	const TFloat B = 2 * (a * (b - center.x) + c * (d - center.y));
	const TFloat C =
	    GET_SQ(b - center.x) + GET_SQ(d - center.y) - GET_SQ(radius);

	const TFloat delta = B * B - 4 * A * C;
	if (delta < 0)
	{
		return false;
	}

	const TFloat t1 = (-B - sqrt(delta)) / (2 * A);
	const TFloat t2 = (-B + sqrt(delta)) / (2 * A);

	p1.x = a * t1 + b;
	p1.y = c * t1 + d;
	p1.z = e * t1 + f;

	p2.x = a * t2 + b;
	p2.y = c * t2 + d;
	p2.z = e * t2 + f;
	return true;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool
    Cylinder3DBase<TFloat>::doesLineIntersectCylinder(
        const Line3DBase<TFloat>& l, Vector3DBase<TFloat>& p1,
        Vector3DBase<TFloat>& p2) const
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

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool
    Cylinder3DBase<TFloat>::clipLine(Line3DBase<TFloat>& l) const
{
	Vector3DBase<TFloat> p1, p2;
	if (!doesLineIntersectCylinder(l, p1, p2))
		return false;
	l.update(p1, p2);
	return true;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool
    Cylinder3DBase<TFloat>::clipLineInfinite(Line3DBase<TFloat>& l) const
{
	Vector3DBase<TFloat> p1, p2;
	if (!doesLineIntersectCylinderInfinite(l, p1, p2))
		return false;
	l.update(p1, p2);
	return true;
}

}  // namespace yrt
