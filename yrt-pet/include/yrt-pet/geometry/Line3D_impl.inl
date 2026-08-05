/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cmath>

#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/geometry/Vector3D_impl.inl"

namespace yrt
{

/*
 * Implementation for Line3DBase. Included by Line3D.cpp (CPU) and
 * Line3D.cuh (GPU). Do not include directly.
 */

template <typename TFloat>
HOST_DEVICE_CALLABLE inline TFloat Line3DBase<TFloat>::getNorm() const
{
	return (point1 - point2).getNorm();
}

template <typename TFloat>
template <typename TargetType>
HOST_DEVICE_CALLABLE inline Line3DBase<TargetType>
    Line3DBase<TFloat>::to() const
{
	const Vector3DBase<TargetType> newPoint1{static_cast<TargetType>(point1.x),
	                                         static_cast<TargetType>(point1.y),
	                                         static_cast<TargetType>(point1.z)};
	const Vector3DBase<TargetType> newPoint2{static_cast<TargetType>(point2.x),
	                                         static_cast<TargetType>(point2.y),
	                                         static_cast<TargetType>(point2.z)};
	return Line3DBase<TargetType>{newPoint1, newPoint2};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool Line3DBase<TFloat>::isValid() const
{
	return detail::abs_(point1.x - point2.x) > static_cast<TFloat>(EPS_FLT) ||
	       detail::abs_(point1.y - point2.y) > static_cast<TFloat>(EPS_FLT) ||
	       detail::abs_(point1.z - point2.z) > static_cast<TFloat>(EPS_FLT);
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Line3DBase<TFloat> Line3DBase<TFloat>::nullLine()
{
	return Line3DBase{Vector3DBase<TFloat>{0, 0, 0},
	                  Vector3DBase<TFloat>{0, 0, 0}};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline void
    Line3DBase<TFloat>::update(const Vector3DBase<TFloat>& pt1,
                               const Vector3DBase<TFloat>& pt2)
{
	point1 = pt1;
	point2 = pt2;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool
    Line3DBase<TFloat>::isEqual(Line3DBase<TFloat>& line) const
{
	const Vector3DBase point1Diff = point1 - line.point1;
	const TFloat distPointSquared1 = point1Diff.getNormSquared();
	if (distPointSquared1 > static_cast<TFloat>(EPS))
	{
		return false;
	}
	const Vector3DBase point2Diff = point2 - line.point2;
	const TFloat distPointSquared2 = point2Diff.getNormSquared();
	if (distPointSquared2 > static_cast<TFloat>(EPS))
	{
		return false;
	}
	return true;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool
    Line3DBase<TFloat>::isParallel(Line3DBase<TFloat>& line) const
{
	const TFloat b = point1.x;
	const TFloat d = point1.y;
	const TFloat f = point1.z;
	const TFloat a = point2.x - b;
	const TFloat c = point2.y - d;
	const TFloat e = point2.z - f;

	const TFloat lb = line.point1.x;
	const TFloat ld = line.point1.y;
	const TFloat lf = line.point1.z;
	const TFloat la = line.point2.x - lb;
	const TFloat lc = line.point2.y - ld;
	const TFloat le = line.point2.z - lf;

	Vector3DBase<TFloat> tmp1{a, c, e};
	tmp1.normalize();
	Vector3DBase<TFloat> tmp2{la, lc, le};
	tmp2.normalize();
	const Vector3DBase<TFloat> crossProd{tmp1.y * tmp2.z - tmp1.z * tmp2.y,
	                                     tmp1.z * tmp2.x - tmp1.x * tmp2.z,
	                                     tmp1.x * tmp2.y - tmp1.y * tmp2.x};
	const TFloat norm = crossProd.getNorm();
	return norm <= static_cast<TFloat>(EPS);
}

}  // namespace yrt
