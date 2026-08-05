/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/geometry/Plane3DBase.hpp"
#include "yrt-pet/geometry/Plane3DBase_impl.inl"

#ifndef __CUDACC__
#include <stdexcept>

namespace yrt
{

template <typename TFloat>
Plane3DBase<TFloat> makePlane3D(const Vector3DBase<TFloat>& pt1,
                                const Vector3DBase<TFloat>& pt2,
                                const Vector3DBase<TFloat>& pt3)
{
	Plane3DBase<TFloat> plane;
	plane.point1 = pt1;
	plane.point2 = pt2;
	plane.point3 = pt3;

	Vector3DBase<TFloat> vector_pos1, vector_pos2;
	vector_pos1 = pt3 - pt1;
	vector_pos2 = pt3 - pt2;
	plane.dir = vector_pos1 * vector_pos2;

	if (!plane.isValid())
	{
		std::string errorMessage =
		    "The 3 points in input of makePlane3D() do not define a plane.";
		errorMessage += "point1 = ( " + std::to_string(pt1.x) + ", " +
		                std::to_string(pt1.y) + ", " + std::to_string(pt1.z) +
		                " )  point2 = ( " + std::to_string(pt2.x) + ", " +
		                std::to_string(pt2.y) + ", " + std::to_string(pt2.z) +
		                " )  point3 = ( " + std::to_string(pt3.x) + ", " +
		                std::to_string(pt3.y) + ", " + std::to_string(pt3.z) +
		                " )";
		throw std::runtime_error(errorMessage);
	}
	plane.update_eq(pt1, pt2, pt3);
	return plane;
}

template Plane3DBase<float> makePlane3D(const Vector3DBase<float>&,
                                        const Vector3DBase<float>&,
                                        const Vector3DBase<float>&);
template Plane3DBase<double> makePlane3D(const Vector3DBase<double>&,
                                         const Vector3DBase<double>&,
                                         const Vector3DBase<double>&);

}  // namespace yrt
#endif

namespace yrt
{

template class Plane3DBase<double>;
template class Plane3DBase<float>;

static_assert(std::is_trivially_constructible<Plane3DBase<double>>());
static_assert(std::is_trivially_destructible<Plane3DBase<double>>());
static_assert(std::is_trivially_copyable<Plane3DBase<double>>());
static_assert(std::is_trivially_copy_constructible<Plane3DBase<double>>());
static_assert(std::is_trivially_copy_assignable<Plane3DBase<double>>());
static_assert(std::is_trivially_default_constructible<Plane3DBase<double>>());
static_assert(std::is_trivially_move_assignable<Plane3DBase<double>>());
static_assert(std::is_trivially_move_constructible<Plane3DBase<double>>());

static_assert(std::is_trivially_constructible<Plane3DBase<float>>());
static_assert(std::is_trivially_destructible<Plane3DBase<float>>());
static_assert(std::is_trivially_copyable<Plane3DBase<float>>());
static_assert(std::is_trivially_copy_constructible<Plane3DBase<float>>());
static_assert(std::is_trivially_copy_assignable<Plane3DBase<float>>());
static_assert(std::is_trivially_default_constructible<Plane3DBase<float>>());
static_assert(std::is_trivially_move_assignable<Plane3DBase<float>>());
static_assert(std::is_trivially_move_constructible<Plane3DBase<float>>());

}  // namespace yrt
