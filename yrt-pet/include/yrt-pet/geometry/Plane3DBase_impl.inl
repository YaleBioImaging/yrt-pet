/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 *
 * Implementation for Plane3DBase. Included by Plane3D.cpp (CPU) and
 * Plane3D.cuh (GPU). Do not include directly.
 */

#pragma once

#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/geometry/Vector3D_impl.inl"

namespace yrt
{

template <typename TFloat>
HOST_DEVICE_CALLABLE inline void
    Plane3DBase<TFloat>::update_eq(const Vector3DBase<TFloat>& pt1,
                                   const Vector3DBase<TFloat>& pt2,
                                   const Vector3DBase<TFloat>& pt3)
{
	const TFloat x1 = pt1.x;
	const TFloat x2 = pt2.x;
	const TFloat x3 = pt3.x;
	const TFloat y1 = pt1.y;
	const TFloat y2 = pt2.y;
	const TFloat y3 = pt3.y;
	const TFloat z1 = pt1.z;
	const TFloat z2 = pt2.z;
	const TFloat z3 = pt3.z;
	a = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2);
	b = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2);
	c = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
	d = -x1 * a - y1 * b - z1 * c;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool Plane3DBase<TFloat>::isValid() const
{
	return dir.getNorm() >= static_cast<TFloat>(EPS_FLT);
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool
    Plane3DBase<TFloat>::isCoplanar(const Vector3DBase<TFloat>& pt) const
{
	const Vector3DBase<TFloat> vector_pos = point3 - pt;
	const TFloat scalProd = dir.scalProd(vector_pos);
	const auto abs_sp = scalProd >= 0 ? scalProd : -scalProd;
	return abs_sp < static_cast<TFloat>(EPS_FLT);
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool
    Plane3DBase<TFloat>::isParallel(const Line3DBase<TFloat>& l) const
{
	const TFloat la = l.point2.x - l.point1.x;
	const TFloat lc = l.point2.y - l.point1.y;
	const TFloat le = l.point2.z - l.point1.z;

	const TFloat test = a * la + b * lc + c * le;
	const auto abs_test = test >= 0 ? test : -test;
	return abs_test < static_cast<TFloat>(1e-8);
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>
    Plane3DBase<TFloat>::findInterLine(const Line3DBase<TFloat>& line) const
{
	const TFloat lb = line.point1.x;
	const TFloat ld = line.point1.y;
	const TFloat lf = line.point1.z;
	const TFloat la = line.point2.x - lb;
	const TFloat lc = line.point2.y - ld;
	const TFloat le = line.point2.z - lf;

	const TFloat denom = a * la + b * lc + c * le;
	Vector3DBase<TFloat> tmp;
	const auto abs_denom = denom >= 0 ? denom : -denom;
	if (abs_denom < static_cast<TFloat>(1e-8))
	{
		tmp.update(static_cast<TFloat>(LARGE_VALUE + 1),
		           static_cast<TFloat>(LARGE_VALUE + 1),
		           static_cast<TFloat>(LARGE_VALUE + 1));
	}
	else
	{
		TFloat t = -(a * lb + b * ld + c * lf + d) / denom;
		tmp.x = la * t + lb;
		tmp.y = lc * t + ld;
		tmp.z = le * t + lf;
	}
	return tmp;
}

}  // namespace yrt
