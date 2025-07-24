/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/Constants.hpp"

#include <cmath>
#include <ostream>

namespace yrt
{

template <typename TFloat>
class Vector3DBase
{
public:
	TFloat getNorm() const { return sqrt(getNormSquared()); }
	TFloat getNormSquared() const { return x * x + y * y + z * z; }
	void update(TFloat xi, TFloat yi, TFloat zi)
	{
		x = xi;
		y = yi;
		z = zi;
	}
	void update(const Vector3DBase& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}
	Vector3DBase& normalize()
	{
		TFloat norm;

		norm = getNorm();

		if (norm > DOUBLE_PRECISION)
		{
			x = x / norm;
			y = y / norm;
			z = z / norm;
		}
		else
		{
			x = 0.;
			y = 0.;
			z = 0.;
		}

		return *this;
	}

	Vector3DBase getNormalized()
	{
		const TFloat norm = getNorm();

		if (norm > DOUBLE_PRECISION)
		{
			return Vector3DBase{x / norm, y / norm, z / norm};
		}
		return Vector3DBase{0., 0., 0.};
	}

	bool isNormalized() const
	{
		return std::abs(1.0 - getNorm()) < SMALL_FLT;
	}
	Vector3DBase operator-(const Vector3DBase& v) const
	{
		return Vector3DBase{x - v.x, y - v.y, z - v.z};
	}

	Vector3DBase operator+(const Vector3DBase& v) const
	{
		return Vector3DBase{x + v.x, y + v.y, z + v.z};
	}

	TFloat scalProd(const Vector3DBase& vector) const
	{
		return x * vector.x + y * vector.y + z * vector.z;
	}
	Vector3DBase crossProduct(const Vector3DBase& B) const
	{
		return Vector3DBase{y * B.z - z * B.y, z * B.x - x * B.z,
							x * B.y - y * B.x};
	}
	void linearTransformation(const Vector3DBase& i,
												const Vector3DBase& j,
												const Vector3DBase& k)
	{
		this->x = i.x * this->x + j.x * this->y + k.x * this->z;
		this->y = i.y * this->x + j.y * this->y + k.y * this->z;
		this->z = i.z * this->x + j.z * this->y + k.z * this->z;
	}
	int argmax()
	{
		return (x > y) ? ((x > z) ? 0 : 2) : ((y > z) ? 1 : 2);
	}
	TFloat operator[](int idx) const
	{
		if (idx == 0)
			return x;
		else if (idx == 1)
			return y;
		else if (idx == 2)
			return z;
		else
			return NAN;
	}
	TFloat operator[](int idx)
	{
		if (idx == 0)
		{
			return x;
		}
		if (idx == 1)
		{
			return y;
		}
		if (idx == 2)
		{
			return z;
		}
		return NAN;
	}

	Vector3DBase operator*(const Vector3DBase& vector) const
	{
		return Vector3DBase{y * vector.z - z * vector.y,
							z * vector.x - x * vector.z,
							x * vector.y - y * vector.x};
	}
	Vector3DBase operator+(TFloat scal) const
	{
		return Vector3DBase{x + scal, y + scal, z + scal};
	}

	Vector3DBase operator-(TFloat scal) const
	{
		return Vector3DBase{x - scal, y - scal, z - scal};
	}

	Vector3DBase operator*(TFloat scal) const
	{
		return Vector3DBase{x * scal, y * scal, z * scal};
	}

	Vector3DBase operator/(TFloat scal) const
	{
		if (std::abs(scal) > DOUBLE_PRECISION)
		{
			return Vector3DBase<TFloat>{x / scal, y / scal, z / scal};
		}
		return Vector3DBase<TFloat>{LARGE_VALUE, LARGE_VALUE, LARGE_VALUE};
	}
	bool operator==(const Vector3DBase& vector) const
	{
		const Vector3DBase tmp{x - vector.x, y - vector.y, z - vector.z};
		if (tmp.getNorm() < SMALL_FLT)
		{
			return true;
		}
		return false;
	}

	template <typename TargetType>
	Vector3DBase<TargetType> to() const
	{
		return Vector3DBase<TargetType>{static_cast<TargetType>(x),
										static_cast<TargetType>(y),
										static_cast<TargetType>(z)};
	}

public:
	TFloat x;
	TFloat y;
	TFloat z;
};

template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const Vector3DBase<TFloat>& v)
{
	oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
	return oss;
}

static_assert(std::is_trivially_constructible<Vector3DBase<double>>());
static_assert(std::is_trivially_destructible<Vector3DBase<double>>());
static_assert(std::is_trivially_copyable<Vector3DBase<double>>());
static_assert(std::is_trivially_copy_constructible<Vector3DBase<double>>());
static_assert(std::is_trivially_copy_assignable<Vector3DBase<double>>());
static_assert(std::is_trivially_default_constructible<Vector3DBase<double>>());
static_assert(std::is_trivially_move_assignable<Vector3DBase<double>>());
static_assert(std::is_trivially_move_constructible<Vector3DBase<double>>());

static_assert(std::is_trivially_constructible<Vector3DBase<float>>());
static_assert(std::is_trivially_destructible<Vector3DBase<float>>());
static_assert(std::is_trivially_copyable<Vector3DBase<float>>());
static_assert(std::is_trivially_copy_constructible<Vector3DBase<float>>());
static_assert(std::is_trivially_copy_assignable<Vector3DBase<float>>());
static_assert(std::is_trivially_default_constructible<Vector3DBase<float>>());
static_assert(std::is_trivially_move_assignable<Vector3DBase<float>>());
static_assert(std::is_trivially_move_constructible<Vector3DBase<float>>());

using Vector3D = Vector3DBase<float>;
// Created to avoid type-castings
using Vector3DDouble = Vector3DBase<double>;

}  // namespace yrt
