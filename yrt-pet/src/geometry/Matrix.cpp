/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/Matrix.hpp"

#include "geometry/Constants.hpp"

#include <cmath>
#include <iostream>


Matrix::Matrix(float a00, float a01, float a02, float a10, float a11, float a12,
               float a20, float a21, float a22)
    : m_a00(a00),
      m_a01(a01),
      m_a02(a02),
      m_a10(a10),
      m_a11(a11),
      m_a12(a12),
      m_a20(a20),
      m_a21(a21),
      m_a22(a22)
{
}

Matrix::Matrix(const Matrix& v)
{
	m_a00 = v.m_a00;
	m_a01 = v.m_a01;
	m_a02 = v.m_a02;
	m_a10 = v.m_a10;
	m_a11 = v.m_a11;
	m_a12 = v.m_a12;
	m_a20 = v.m_a20;
	m_a21 = v.m_a21;
	m_a22 = v.m_a22;
}

Matrix::Matrix()
    : m_a00(0.0),
      m_a01(0.0),
      m_a02(0.0),
      m_a10(0.0),
      m_a11(0.0),
      m_a12(0.0),
      m_a20(0.0),
      m_a21(0.0),
      m_a22(0.0)
{
}

Matrix Matrix::identity()
{
	return Matrix{1, 0, 0, 0, 1, 0, 0, 0, 1};
}

template <int row, int col>
float Matrix::element() const
{
	static_assert(row >= 0 && col >= 0);
	static_assert(row < 3 && col < 3);
	if constexpr (row == 0 && col == 0)
		return m_a00;
	if constexpr (row == 0 && col == 1)
		return m_a01;
	if constexpr (row == 0 && col == 2)
		return m_a02;
	if constexpr (row == 1 && col == 0)
		return m_a10;
	if constexpr (row == 1 && col == 1)
		return m_a11;
	if constexpr (row == 1 && col == 2)
		return m_a12;
	if constexpr (row == 2 && col == 0)
		return m_a20;
	if constexpr (row == 2 && col == 1)
		return m_a21;
	if constexpr (row == 2 && col == 2)
		return m_a22;
	throw std::runtime_error("Unexpected error");
}
template float Matrix::element<0, 0>() const;
template float Matrix::element<0, 1>() const;
template float Matrix::element<0, 2>() const;
template float Matrix::element<1, 0>() const;
template float Matrix::element<1, 1>() const;
template float Matrix::element<1, 2>() const;
template float Matrix::element<2, 0>() const;
template float Matrix::element<2, 1>() const;
template float Matrix::element<2, 2>() const;

void Matrix::update(float a00, float a01, float a02, float a10, float a11,
                    float a12, float a20, float a21, float a22)
{
	m_a00 = a00;
	m_a01 = a01;
	m_a02 = a02;
	m_a10 = a10;
	m_a11 = a11;
	m_a12 = a12;
	m_a20 = a20;
	m_a21 = a21;
	m_a22 = a22;
}

// update 3:
void Matrix::update(const Matrix& v)
{
	m_a00 = v.m_a00;
	m_a01 = v.m_a01;
	m_a02 = v.m_a02;
	m_a10 = v.m_a10;
	m_a11 = v.m_a11;
	m_a12 = v.m_a12;
	m_a20 = v.m_a20;
	m_a21 = v.m_a21;
	m_a22 = v.m_a22;
}

Matrix Matrix::operator-(Matrix v) const
{

	Matrix res;

	res.m_a00 = m_a00 - v.m_a00;
	res.m_a01 = m_a01 - v.m_a01;
	res.m_a02 = m_a02 - v.m_a02;
	res.m_a10 = m_a10 - v.m_a10;
	res.m_a11 = m_a11 - v.m_a11;
	res.m_a12 = m_a12 - v.m_a12;
	res.m_a20 = m_a20 - v.m_a20;
	res.m_a21 = m_a21 - v.m_a21;
	res.m_a22 = m_a22 - v.m_a22;

	return res;
}

Matrix Matrix::operator+(Matrix v) const
{

	Matrix res;

	res.m_a00 = m_a00 + v.m_a00;
	res.m_a01 = m_a01 + v.m_a01;
	res.m_a02 = m_a02 + v.m_a02;
	res.m_a10 = m_a10 + v.m_a10;
	res.m_a11 = m_a11 + v.m_a11;
	res.m_a12 = m_a12 + v.m_a12;
	res.m_a20 = m_a20 + v.m_a20;
	res.m_a21 = m_a21 + v.m_a21;
	res.m_a22 = m_a22 + v.m_a22;

	return res;
}

Matrix Matrix::operator*(Matrix matrix) const
{
	Matrix res;

	res.m_a00 = m_a00 * matrix.m_a00;
	res.m_a01 = m_a01 * matrix.m_a01;
	res.m_a02 = m_a02 * matrix.m_a02;
	res.m_a10 = m_a10 * matrix.m_a10;
	res.m_a11 = m_a11 * matrix.m_a11;
	res.m_a12 = m_a12 * matrix.m_a12;
	res.m_a20 = m_a20 * matrix.m_a20;
	res.m_a21 = m_a21 * matrix.m_a21;
	res.m_a22 = m_a22 * matrix.m_a22;

	return res;
}

Vector3D Matrix::operator*(const Vector3D& vector) const
{
	Vector3D res;

	res.x = m_a00 * vector.x + m_a01 * vector.y + m_a02 * vector.z;
	res.y = m_a10 * vector.x + m_a11 * vector.y + m_a12 * vector.z;
	res.z = m_a20 * vector.x + m_a21 * vector.y + m_a22 * vector.z;

	return res;
}


Matrix Matrix::operator+(float scal) const
{
	Matrix res;

	res.m_a00 = m_a00 + scal;
	res.m_a01 = m_a01 + scal;
	res.m_a02 = m_a02 + scal;
	res.m_a10 = m_a10 + scal;
	res.m_a11 = m_a11 + scal;
	res.m_a12 = m_a12 + scal;
	res.m_a20 = m_a20 + scal;
	res.m_a21 = m_a21 + scal;
	res.m_a22 = m_a22 + scal;

	return res;
}

Matrix Matrix::operator-(float scal) const
{
	Matrix res;

	res.m_a00 = m_a00 - scal;
	res.m_a01 = m_a01 - scal;
	res.m_a02 = m_a02 - scal;
	res.m_a10 = m_a10 - scal;
	res.m_a11 = m_a11 - scal;
	res.m_a12 = m_a12 - scal;
	res.m_a20 = m_a20 - scal;
	res.m_a21 = m_a21 - scal;
	res.m_a22 = m_a22 - scal;

	return res;
}

Matrix Matrix::operator*(float scal) const
{
	Matrix res;

	res.m_a00 = m_a00 * scal;
	res.m_a01 = m_a01 * scal;
	res.m_a02 = m_a02 * scal;
	res.m_a10 = m_a10 * scal;
	res.m_a11 = m_a11 * scal;
	res.m_a12 = m_a12 * scal;
	res.m_a20 = m_a20 * scal;
	res.m_a21 = m_a21 * scal;
	res.m_a22 = m_a22 * scal;

	return res;
}

Matrix Matrix::operator/(float scal) const
{
	Matrix res;

	if (std::abs(scal) > SMALL_FLT)
	{
		res.m_a00 = m_a00 / scal;
		res.m_a01 = m_a01 / scal;
		res.m_a02 = m_a02 / scal;
		res.m_a10 = m_a10 / scal;
		res.m_a11 = m_a11 / scal;
		res.m_a12 = m_a12 / scal;
		res.m_a20 = m_a20 / scal;
		res.m_a21 = m_a21 / scal;
		res.m_a22 = m_a22 / scal;
	}
	else
	{
		res.m_a00 = LARGE_VALUE;
		res.m_a01 = LARGE_VALUE;
		res.m_a02 = LARGE_VALUE;
		res.m_a10 = LARGE_VALUE;
		res.m_a11 = LARGE_VALUE;
		res.m_a12 = LARGE_VALUE;
		res.m_a20 = LARGE_VALUE;
		res.m_a21 = LARGE_VALUE;
		res.m_a22 = LARGE_VALUE;
	}

	return res;
}

// return true if matrices are the same:
bool Matrix::operator==(Matrix matrix) const
{
	float sqr_norm = (m_a00 - matrix.m_a00) * (m_a00 - matrix.m_a00) +
	                 (m_a01 - matrix.m_a01) * (m_a01 - matrix.m_a01) +
	                 (m_a02 - matrix.m_a02) * (m_a02 - matrix.m_a02) +
	                 (m_a10 - matrix.m_a10) * (m_a10 - matrix.m_a10) +
	                 (m_a11 - matrix.m_a11) * (m_a11 - matrix.m_a11) +
	                 (m_a12 - matrix.m_a12) * (m_a12 - matrix.m_a12) +
	                 (m_a20 - matrix.m_a20) * (m_a20 - matrix.m_a20) +
	                 (m_a21 - matrix.m_a21) * (m_a21 - matrix.m_a21) +
	                 (m_a22 - matrix.m_a22) * (m_a22 - matrix.m_a22);

	return sqrt(sqr_norm) < SMALL_FLT;
}

std::ostream& operator<<(std::ostream& oss, const Matrix& v)
{
	oss << "("
	    << "(" << v.m_a00 << ", " << v.m_a01 << ", " << v.m_a02 << "), ("
	    << v.m_a10 << ", " << v.m_a11 << ", " << v.m_a12 << "), (" << v.m_a20
	    << ", " << v.m_a21 << ", " << v.m_a22 << ")";
	return oss;
}
