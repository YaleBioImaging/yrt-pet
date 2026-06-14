/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cmath>

#ifndef __CUDACC__
#include <stdexcept>
#include <string>
#endif

#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/geometry/Vector3D.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"

namespace yrt
{

class Plane
{
public:
	Vector3D point1;
	Vector3D point2;
	Vector3D point3;
	Vector3D dir;
	double a;
	double b;
	double c;
	double d;
	// equation of the plane: a.x + b.y + c.z + d = 0

public:

	HOST_DEVICE_CALLABLE Plane() {}

#ifndef __CUDACC__
	HOST_DEVICE_CALLABLE Plane(const Vector3D& pt1, const Vector3D& pt2,
	                           const Vector3D& pt3);
	HOST_DEVICE_CALLABLE void update(const Vector3D& pt1, const Vector3D& pt2,
	                                 const Vector3D& pt3);
#endif

	HOST_DEVICE_CALLABLE void
	    update_eq(const Vector3D& pt1, const Vector3D& pt2, const Vector3D& pt3)
	{
		const float x1 = pt1.x;
		const float x2 = pt2.x;
		const float x3 = pt3.x;
		const float y1 = pt1.y;
		const float y2 = pt2.y;
		const float y3 = pt3.y;
		const float z1 = pt1.z;
		const float z2 = pt2.z;
		const float z3 = pt3.z;
		a = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2);
		b = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2);
		c = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
		d = -x1 * a - y1 * b - z1 * c;
	}

	HOST_DEVICE_CALLABLE bool isCoplanar(const Vector3D& pt) const
	{
		const Vector3D vector_pos = point3 - pt;
		const float scalProd = dir.scalProd(vector_pos);
		const auto abs_sp = scalProd >= 0 ? scalProd : -scalProd;
		return abs_sp < EPS_FLT;
	}

	HOST_DEVICE_CALLABLE bool isParallel(const Line3D& l) const
	{
		const float la = l.point2.x - l.point1.x;
		const float lc = l.point2.y - l.point1.y;
		const float le = l.point2.z - l.point1.z;

		const float test = a * la + b * lc + c * le;
		const auto abs_test = test >= 0 ? test : -test;
		return abs_test < static_cast<float>(1e-8);
	}

	HOST_DEVICE_CALLABLE Vector3D findInterLine(const Line3D& line) const
	{
		const float lb = line.point1.x;
		const float ld = line.point1.y;
		const float lf = line.point1.z;
		const float la = line.point2.x - lb;
		const float lc = line.point2.y - ld;
		const float le = line.point2.z - lf;

		const float denom = a * la + b * lc + c * le;
		Vector3D tmp;
		const auto abs_denom = denom >= 0 ? denom : -denom;
		if (abs_denom < static_cast<float>(1e-8))
		{
			tmp.update(LARGE_VALUE + 1, LARGE_VALUE + 1, LARGE_VALUE + 1);
		}
		else
		{
			float t = -(a * lb + b * ld + c * lf + d) / denom;
			tmp.x = la * t + lb;
			tmp.y = lc * t + ld;
			tmp.z = le * t + lf;
		}
		return tmp;
	}
};

#ifndef __CUDACC__
// These throw exceptions and cannot be on GPU
HOST_DEVICE_CALLABLE inline Plane::Plane(const Vector3D& pt1,
                                         const Vector3D& pt2,
                                         const Vector3D& pt3)
    : point1(pt1), point2(pt2), point3(pt3)
{
	Vector3D vector_pos1, vector_pos2;
	vector_pos1.update(pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z);
	vector_pos2.update(pt3.x - pt2.x, pt3.y - pt2.y, pt3.z - pt2.z);
	dir = vector_pos1 * vector_pos2;
	if (dir.getNorm() < EPS_FLT)
	{
		std::string errorMessage =
		    "The 3 points in input of Plane::Plane() do not define a plane.";
		errorMessage += "point1 = ( " + std::to_string(pt1.x) + ", " +
		                std::to_string(pt1.y) + ", " + std::to_string(pt1.z) +
		                " )  point2 = ( " + std::to_string(pt2.x) + ", " +
		                std::to_string(pt2.y) + ", " + std::to_string(pt2.z) +
		                " )  point3 = ( " + std::to_string(pt3.x) + ", " +
		                std::to_string(pt3.y) + ", " + std::to_string(pt3.z) +
		                " )";
		throw std::runtime_error(errorMessage);
	}
	update_eq(pt1, pt2, pt3);
}

HOST_DEVICE_CALLABLE inline void
    Plane::update(const Vector3D& pt1, const Vector3D& pt2, const Vector3D& pt3)
{
	Vector3D vector_pos1, vector_pos2;
	point1.update(pt1.x, pt1.y, pt1.z);
	point2.update(pt2.x, pt2.y, pt2.z);
	point3.update(pt3.x, pt3.y, pt3.z);
	vector_pos1.update(pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z);
	vector_pos2.update(pt3.x - pt2.x, pt3.y - pt2.y, pt3.z - pt2.z);
	const Vector3D dir_tmp = vector_pos1 * vector_pos2;
	dir.update(dir_tmp.x, dir_tmp.y, dir_tmp.z);
	if (dir.getNorm() < EPS_FLT)
	{
		std::string errorMessage =
		    "The 3 points in input of Plane::update() do not define a plane.";
		errorMessage += "point1 = ( " + std::to_string(pt1.x) + ", " +
		                std::to_string(pt1.y) + ", " + std::to_string(pt1.z) +
		                " )  point2 = ( " + std::to_string(pt2.x) + ", " +
		                std::to_string(pt2.y) + ", " + std::to_string(pt2.z) +
		                " )  point3 = ( " + std::to_string(pt3.x) + ", " +
		                std::to_string(pt3.y) + ", " + std::to_string(pt3.z) +
		                " )";
		throw std::runtime_error(errorMessage);
	}
	update_eq(pt1, pt2, pt3);
}
#endif

}  // namespace yrt
