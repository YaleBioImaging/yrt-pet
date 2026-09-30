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

#ifndef __CUDACC__
#include <string>
#endif

namespace yrt
{

template <typename TFloat>
class Plane3DBase
{
public:
	Vector3DBase<TFloat> point1;
	Vector3DBase<TFloat> point2;
	Vector3DBase<TFloat> point3;
	Vector3DBase<TFloat> dir;
	TFloat a;
	TFloat b;
	TFloat c;
	TFloat d;
	// equation of the plane: a.x + b.y + c.z + d = 0

	HOST_DEVICE_CALLABLE void
	    update_eq(const Vector3DBase<TFloat>& pt1,
	              const Vector3DBase<TFloat>& pt2,
	              const Vector3DBase<TFloat>& pt3);

	HOST_DEVICE_CALLABLE bool isValid() const;

	HOST_DEVICE_CALLABLE bool isCoplanar(const Vector3DBase<TFloat>& pt) const;
	HOST_DEVICE_CALLABLE bool isParallel(const Line3DBase<TFloat>& l) const;
	HOST_DEVICE_CALLABLE Vector3DBase<TFloat>
	    findInterLine(const Line3DBase<TFloat>& line) const;
};

#ifndef __CUDACC__
template <typename TFloat>
Plane3DBase<TFloat> makePlane3D(const Vector3DBase<TFloat>& pt1,
                                const Vector3DBase<TFloat>& pt2,
                                const Vector3DBase<TFloat>& pt3);
#endif

using Plane3D = Plane3DBase<float>;

// Created to avoid type-castings
using Plane3DDouble = Plane3DBase<double>;

}  // namespace yrt
