/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/geometry/Cylinder3DBase.hpp"
#include "yrt-pet/geometry/Cylinder3DBase_impl.inl"

#ifndef __CUDACC__
#include <stdexcept>

namespace yrt
{

template <typename TFloat>
Cylinder3DBase<TFloat> makeCylinder3D(const Vector3DBase<TFloat>& center,
                                      TFloat length_z, TFloat radius)
{
	Cylinder3DBase<TFloat> cyl;
	cyl.center = center;
	cyl.length_z = length_z;
	cyl.radius = radius;
	if (!cyl.isValid())
	{
		throw std::runtime_error(
		    "Invalid Cylinder3D: length_z and radius must be positive.");
	}
	return cyl;
}

template Cylinder3DBase<float> makeCylinder3D(const Vector3DBase<float>&,
                                              float, float);
template Cylinder3DBase<double> makeCylinder3D(const Vector3DBase<double>&,
                                               double, double);

}  // namespace yrt
#endif

namespace yrt
{

template class Cylinder3DBase<double>;
template class Cylinder3DBase<float>;

static_assert(std::is_trivially_constructible<Cylinder3DBase<double>>());
static_assert(std::is_trivially_destructible<Cylinder3DBase<double>>());
static_assert(std::is_trivially_copyable<Cylinder3DBase<double>>());
static_assert(std::is_trivially_copy_constructible<Cylinder3DBase<double>>());
static_assert(std::is_trivially_copy_assignable<Cylinder3DBase<double>>());
static_assert(std::is_trivially_default_constructible<Cylinder3DBase<double>>());
static_assert(std::is_trivially_move_assignable<Cylinder3DBase<double>>());
static_assert(std::is_trivially_move_constructible<Cylinder3DBase<double>>());

static_assert(std::is_trivially_constructible<Cylinder3DBase<float>>());
static_assert(std::is_trivially_destructible<Cylinder3DBase<float>>());
static_assert(std::is_trivially_copyable<Cylinder3DBase<float>>());
static_assert(std::is_trivially_copy_constructible<Cylinder3DBase<float>>());
static_assert(std::is_trivially_copy_assignable<Cylinder3DBase<float>>());
static_assert(std::is_trivially_default_constructible<Cylinder3DBase<float>>());
static_assert(std::is_trivially_move_assignable<Cylinder3DBase<float>>());
static_assert(std::is_trivially_move_constructible<Cylinder3DBase<float>>());

}  // namespace yrt
