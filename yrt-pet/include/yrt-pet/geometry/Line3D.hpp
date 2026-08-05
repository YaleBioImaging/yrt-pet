/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/Vector3D.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"

#ifndef __CUDACC__
#include <ostream>
#endif

namespace yrt
{

template <typename TFloat>
class Line3DBase
{
public:
	HOST_DEVICE_CALLABLE void update(const Vector3DBase<TFloat>& pt1,
	                                 const Vector3DBase<TFloat>& pt2);

	HOST_DEVICE_CALLABLE bool isEqual(Line3DBase<TFloat>& line) const;
	HOST_DEVICE_CALLABLE bool isParallel(Line3DBase<TFloat>& line) const;
	HOST_DEVICE_CALLABLE TFloat getNorm() const;

	template <typename TargetType>
	HOST_DEVICE_CALLABLE Line3DBase<TargetType> to() const;

	// Return false if both endpoints of the line are the the same
	HOST_DEVICE_CALLABLE bool isValid() const;

	HOST_DEVICE_CALLABLE static Line3DBase<TFloat> nullLine();

public:
	Vector3DBase<TFloat> point1;
	Vector3DBase<TFloat> point2;
};

#ifndef __CUDACC__
template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const Line3DBase<TFloat>& l);
#endif

using Line3D = Line3DBase<float>;

// Created to avoid type-castings
using Line3DDouble = Line3DBase<double>;

}  // namespace yrt
