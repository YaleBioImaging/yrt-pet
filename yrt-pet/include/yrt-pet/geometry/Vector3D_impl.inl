/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 *
 * Implementation for Vector3DBase. Included by Vector3D.cpp (CPU) and
 * Vector3D.cuh (GPU). Do not include directly.
 */

#pragma once

#include <cmath>
#include <cstdlib>

#include "yrt-pet/geometry/Constants.hpp"

namespace yrt
{

namespace detail
{
template <typename T>
HOST_DEVICE_CALLABLE inline T abs_(T x)
{
	return x >= 0 ? x : -x;
}
}  // namespace detail

template <typename TFloat>
HOST_DEVICE_CALLABLE inline TFloat Vector3DBase<TFloat>::getNorm() const
{
	return sqrt(getNormSquared());
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline TFloat Vector3DBase<TFloat>::getNormSquared() const
{
	return x * x + y * y + z * z;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline void Vector3DBase<TFloat>::update(TFloat xi,
                                                              TFloat yi,
                                                              TFloat zi)
{
	x = xi;
	y = yi;
	z = zi;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline void Vector3DBase<TFloat>::update(
    const Vector3DBase& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>&
    Vector3DBase<TFloat>::normalize()
{
	const TFloat norm = getNorm();
	if (norm > static_cast<TFloat>(DOUBLE_PRECISION))
	{
		x /= norm;
		y /= norm;
		z /= norm;
	}
	else
	{
		x = 0;
		y = 0;
		z = 0;
	}
	return *this;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>
    Vector3DBase<TFloat>::getNormalized()
{
	const TFloat norm = getNorm();
	if (norm > static_cast<TFloat>(DOUBLE_PRECISION))
	{
		return Vector3DBase{x / norm, y / norm, z / norm};
	}
	return Vector3DBase{0, 0, 0};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool Vector3DBase<TFloat>::isNormalized() const
{
	return detail::abs_(static_cast<TFloat>(1.0) - getNorm()) <
	       static_cast<TFloat>(EPS);
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>
    Vector3DBase<TFloat>::operator-(const Vector3DBase& v) const
{
	return Vector3DBase{x - v.x, y - v.y, z - v.z};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>
    Vector3DBase<TFloat>::operator+(const Vector3DBase& v) const
{
	return Vector3DBase{x + v.x, y + v.y, z + v.z};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>&
    Vector3DBase<TFloat>::operator-=(const Vector3DBase& v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>&
    Vector3DBase<TFloat>::operator+=(const Vector3DBase& v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline TFloat
    Vector3DBase<TFloat>::scalProd(const Vector3DBase& vector) const
{
	return x * vector.x + y * vector.y + z * vector.z;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>
    Vector3DBase<TFloat>::crossProduct(const Vector3DBase& B) const
{
	return Vector3DBase{y * B.z - z * B.y, z * B.x - x * B.z,
	                    x * B.y - y * B.x};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline void Vector3DBase<TFloat>::linearTransformation(
    const Vector3DBase& i, const Vector3DBase& j, const Vector3DBase& k)
{
	const TFloat new_x = i.x * x + j.x * y + k.x * z;
	const TFloat new_y = i.y * x + j.y * y + k.y * z;
	const TFloat new_z = i.z * x + j.z * y + k.z * z;
	x = new_x;
	y = new_y;
	z = new_z;
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline int Vector3DBase<TFloat>::argmax()
{
	return (x > y) ? ((x > z) ? 0 : 2) : ((y > z) ? 1 : 2);
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline TFloat Vector3DBase<TFloat>::operator[](
    int idx) const
{
	if (idx == 0)
		return x;
	if (idx == 1)
		return y;
	if (idx == 2)
		return z;
	return static_cast<TFloat>(NAN);
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline TFloat Vector3DBase<TFloat>::operator[](int idx)
{
	if (idx == 0)
		return x;
	if (idx == 1)
		return y;
	if (idx == 2)
		return z;
	return static_cast<TFloat>(NAN);
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>
    Vector3DBase<TFloat>::operator*(const Vector3DBase& vector) const
{
	return Vector3DBase{y * vector.z - z * vector.y,
	                    z * vector.x - x * vector.z,
	                    x * vector.y - y * vector.x};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>
    Vector3DBase<TFloat>::operator+(TFloat scal) const
{
	return Vector3DBase{x + scal, y + scal, z + scal};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>
    Vector3DBase<TFloat>::operator-(TFloat scal) const
{
	return Vector3DBase{x - scal, y - scal, z - scal};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>
    Vector3DBase<TFloat>::operator*(TFloat scal) const
{
	return Vector3DBase{x * scal, y * scal, z * scal};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline Vector3DBase<TFloat>
    Vector3DBase<TFloat>::operator/(TFloat scal) const
{
	if (detail::abs_(scal) > static_cast<TFloat>(DOUBLE_PRECISION))
	{
		return Vector3DBase<TFloat>{x / scal, y / scal, z / scal};
	}
	return Vector3DBase<TFloat>{static_cast<TFloat>(LARGE_VALUE),
	                            static_cast<TFloat>(LARGE_VALUE),
	                            static_cast<TFloat>(LARGE_VALUE)};
}

template <typename TFloat>
HOST_DEVICE_CALLABLE inline bool Vector3DBase<TFloat>::operator==(
    const Vector3DBase& vector) const
{
	const Vector3DBase tmp{x - vector.x, y - vector.y, z - vector.z};
	return tmp.getNorm() < static_cast<TFloat>(EPS);
}

template <typename TFloat>
template <typename TargetType>
HOST_DEVICE_CALLABLE inline Vector3DBase<TargetType>
    Vector3DBase<TFloat>::to() const
{
	return Vector3DBase<TargetType>{static_cast<TargetType>(x),
	                                static_cast<TargetType>(y),
	                                static_cast<TargetType>(z)};
}

}  // namespace yrt
