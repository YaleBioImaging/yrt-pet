/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/Plane.hpp"

#include "geometry/Constants.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

Plane::Plane() {}


Plane::Plane(const Vector3D& pt1, const Vector3D& pt2, const Vector3D& pt3)
    : point1(pt1), point2(pt2), point3(pt3)
{
	Vector3D vector_pos1, vector_pos2;
	// check that these points define a plane:
	vector_pos1.update(pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z);
	vector_pos2.update(pt3.x - pt2.x, pt3.y - pt2.y, pt3.z - pt2.z);
	dir = vector_pos1 * vector_pos2;
	if (dir.getNorm() < SMALL_FLT)
	{
		std::cout << "\nThe 3 points in input of Plane::Plane() do not "
		             "define a plane.\n";
		std::cout << "point1 = ( " << pt1.x << ", " << pt1.y << ", " << pt1.z
		          << " )  point2 = ( " << pt2.x << ", " << pt2.y << ", "
		          << pt2.z << " )  point3 = ( " << pt3.x << ", " << pt3.y
		          << ", " << pt3.z << " )\n\n";
		exit(-1);
	}
	// find equation of the plane:
	update_eq(pt1, pt2, pt3);
}


// update function:
void Plane::update(const Vector3D& pt1, const Vector3D& pt2,
                   const Vector3D& pt3)
{
	Vector3D vector_pos1, vector_pos2;
	point1.update(pt1.x, pt1.y, pt1.z);
	point2.update(pt2.x, pt2.y, pt2.z);
	point3.update(pt3.x, pt3.y, pt3.z);
	// check that these points define a plane:
	vector_pos1.update(pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z);
	vector_pos2.update(pt3.x - pt2.x, pt3.y - pt2.y, pt3.z - pt2.z);
	const Vector3D dir_tmp = vector_pos1 * vector_pos2;
	dir.update(dir_tmp.x, dir_tmp.y, dir_tmp.z);
	if (dir.getNorm() < SMALL_FLT)
	{
		std::cout << "\nThe 3 points in input of Plane::Plane() do not "
		             "define a plane.\n";
		std::cout << "point1 = ( " << pt1.x << ", " << pt1.y << ", " << pt1.z
		          << " )  point2 = ( " << pt2.x << ", " << pt2.y << ", "
		          << pt2.z << " )  point3 = ( " << pt3.x << ", " << pt3.y
		          << ", " << pt3.z << " )\n\n",
		    getchar();
		exit(-1);
	}
	// update equation:
	update_eq(pt1, pt2, pt3);
}

// update equation of the plane using these 3 points:
void Plane::update_eq(const Vector3D& pt1, const Vector3D& pt2,
                      const Vector3D& pt3)
{
	// find equation of the plane
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


// returns true if "point" is in the plane
bool Plane::isCoplanar(const Vector3D& pt) const
{
	bool isCop = false;
	const Vector3D vector_pos = point3 - pt;
	const float scalProd = dir.scalProd(vector_pos);
	if (std::abs(scalProd) < SMALL_FLT)
	{
		isCop = true;
	}
	return isCop;
}


// returns true if "line" is parrallel to the current plan
bool Plane::isParrallel(const Line3D& l) const
{
	const float la = l.point2.x - l.point1.x;
	const float lc = l.point2.y - l.point1.y;
	const float le = l.point2.z - l.point1.z;

	const float test = a * la + b * lc + c * le;
	if (std::abs(test) < 10e-8)
	{
		return true;
	}
	return false;
}


// this function returns the intersection point of the current plane
// with a line defined by two points "point1" and "point2".
// The coords. of the inter point are equal to LARGE_VALUE+1
// if the line is parrallel to the plane.
Vector3D Plane::findInterLine(const Line3D& line) const
{
	const float lb = line.point1.x;
	const float ld = line.point1.y;
	const float lf = line.point1.z;
	const float la = line.point2.x - lb;
	const float lc = line.point2.y - ld;
	const float le = line.point2.z - lf;

	const float denom = a * la + b * lc + c * le;
	float t = a * lb + b * ld + c * lf + d;
	Vector3D tmp;
	if (std::abs(denom) < 10e-8)
	{
		tmp.update(LARGE_VALUE + 1, LARGE_VALUE + 1, LARGE_VALUE + 1);
	}
	else
	{
		t /= -denom;
		tmp.x = la * t + lb;
		tmp.y = lc * t + ld;
		tmp.z = le * t + lf;
	}
	return tmp;
}
