/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <ostream>

template <typename TFloat>
class Vector3DBase
{
public:
	TFloat getNorm() const;
	TFloat getNormSquared() const;
	void update(TFloat xi, TFloat yi, TFloat zi);
	void update(const Vector3DBase& v);
	Vector3DBase& normalize();
	Vector3DBase getNormalized();
	bool isNormalized() const;
	Vector3DBase operator-(const Vector3DBase& v) const;
	Vector3DBase operator+(const Vector3DBase& v) const;
	TFloat scalProd(const Vector3DBase& vector) const;
	Vector3DBase crossProduct(const Vector3DBase& B) const;
	void linearTransformation(const Vector3DBase& i, const Vector3DBase& j,
	                          const Vector3DBase& k);
	int argmax();
	Vector3DBase operator*(const Vector3DBase& vector) const;
	Vector3DBase operator+(TFloat scal) const;
	Vector3DBase operator-(TFloat scal) const;
	Vector3DBase operator*(TFloat scal) const;
	Vector3DBase operator/(TFloat scal) const;
	TFloat operator[](int idx) const;
	TFloat operator[](int idx);
	bool operator==(const Vector3DBase& vector) const;

	template <typename TargetType>
	Vector3DBase<TargetType> to() const;

public:
	TFloat x;
	TFloat y;
	TFloat z;
};

template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const Vector3DBase<TFloat>& v);

using Vector3D = Vector3DBase<float>;

// Created to avoid type-castings
using Vector3DDouble = Vector3DBase<double>;
