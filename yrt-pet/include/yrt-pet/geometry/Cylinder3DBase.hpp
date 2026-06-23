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
class Cylinder3DBase
{
public:
	Vector3DBase<TFloat> center;
	TFloat length_z;
	TFloat radius;

	HOST_DEVICE_CALLABLE bool isValid() const;

	HOST_DEVICE_CALLABLE bool doesLineIntersectCylinderInfinite(
	    const Line3DBase<TFloat>& l, Vector3DBase<TFloat>& p1,
	    Vector3DBase<TFloat>& p2) const;

	HOST_DEVICE_CALLABLE bool doesLineIntersectCylinder(
	    const Line3DBase<TFloat>& l, Vector3DBase<TFloat>& p1,
	    Vector3DBase<TFloat>& p2) const;

	HOST_DEVICE_CALLABLE bool clipLine(Line3DBase<TFloat>& l) const;
	HOST_DEVICE_CALLABLE bool clipLineInfinite(Line3DBase<TFloat>& l) const;
};

#ifndef __CUDACC__
template <typename TFloat>
Cylinder3DBase<TFloat> makeCylinder3D(const Vector3DBase<TFloat>& center,
                                      TFloat length_z, TFloat radius);
#endif

using Cylinder3D = Cylinder3DBase<float>;

// Created to avoid type-castings
using Cylinder3DDouble = Cylinder3DBase<double>;

}  // namespace yrt
