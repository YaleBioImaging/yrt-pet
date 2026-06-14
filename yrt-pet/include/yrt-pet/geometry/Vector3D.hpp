/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/GPUUtils.cuh"

#ifndef __CUDACC__
#include <ostream>
#endif

namespace yrt
{

template <typename TFloat>
class Vector3DBase
{
public:
	HOST_DEVICE_CALLABLE TFloat getNorm() const;
	HOST_DEVICE_CALLABLE TFloat getNormSquared() const;
	HOST_DEVICE_CALLABLE void update(TFloat xi, TFloat yi, TFloat zi);
	HOST_DEVICE_CALLABLE void update(const Vector3DBase& v);
	HOST_DEVICE_CALLABLE Vector3DBase& normalize();
	HOST_DEVICE_CALLABLE Vector3DBase getNormalized();
	HOST_DEVICE_CALLABLE bool isNormalized() const;
	HOST_DEVICE_CALLABLE Vector3DBase operator-(const Vector3DBase& v) const;
	HOST_DEVICE_CALLABLE Vector3DBase operator+(const Vector3DBase& v) const;
	HOST_DEVICE_CALLABLE Vector3DBase& operator-=(const Vector3DBase& v);
	HOST_DEVICE_CALLABLE Vector3DBase& operator+=(const Vector3DBase& v);
	HOST_DEVICE_CALLABLE TFloat scalProd(const Vector3DBase& vector) const;
	HOST_DEVICE_CALLABLE Vector3DBase crossProduct(const Vector3DBase& B) const;
	HOST_DEVICE_CALLABLE void linearTransformation(const Vector3DBase& i,
	                                               const Vector3DBase& j,
	                                               const Vector3DBase& k);
	HOST_DEVICE_CALLABLE int argmax();
	HOST_DEVICE_CALLABLE Vector3DBase operator*(const Vector3DBase& vector) const;
	HOST_DEVICE_CALLABLE Vector3DBase operator+(TFloat scal) const;
	HOST_DEVICE_CALLABLE Vector3DBase operator-(TFloat scal) const;
	HOST_DEVICE_CALLABLE Vector3DBase operator*(TFloat scal) const;
	HOST_DEVICE_CALLABLE Vector3DBase operator/(TFloat scal) const;
	HOST_DEVICE_CALLABLE TFloat operator[](int idx) const;
	HOST_DEVICE_CALLABLE TFloat operator[](int idx);
	HOST_DEVICE_CALLABLE bool operator==(const Vector3DBase& vector) const;

	template <typename TargetType>
	HOST_DEVICE_CALLABLE Vector3DBase<TargetType> to() const;

public:
	TFloat x;
	TFloat y;
	TFloat z;
};

#ifndef __CUDACC__
template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const Vector3DBase<TFloat>& v);
#endif

using Vector3D = Vector3DBase<float>;

// Created to avoid type-castings
using Vector3DDouble = Vector3DBase<double>;

}  // namespace yrt
