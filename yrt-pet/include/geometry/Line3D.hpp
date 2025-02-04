/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/Vector3D.hpp"

template<typename TFloat>
class Line3DBase
{
public:
	void update(const Vector3DBase<TFloat>& pt1, const Vector3DBase<TFloat>& pt2);

	bool isEqual(Line3DBase<TFloat>& line) const;
	bool isParallel(Line3DBase<TFloat>& line) const;
	TFloat getNorm() const;

	template<typename TargetType>
	Line3DBase<TargetType> to() const;

	static Line3DBase<TFloat> nullLine();

public:
	Vector3DBase<TFloat> point1;
	Vector3DBase<TFloat> point2;
};

template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const Line3DBase<TFloat>& l);

using Line3D = Line3DBase<float>;

// Created to avoid type-castings
using Line3DDouble = Line3DBase<double>;

