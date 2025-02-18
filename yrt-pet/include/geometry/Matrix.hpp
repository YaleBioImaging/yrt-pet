/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/Vector3D.hpp"
#include <ostream>

class Matrix
{
public:
	Matrix(float a00, float a01, float a02, float a10, float a11, float a12,
	       float a20, float a21, float a22);
	Matrix(const Matrix& v);
	Matrix();
	static Matrix identity();

	void update(float a00, float a01, float a02, float a10, float a11,
	            float a12, float a20, float a21, float a22);
	void update(const Matrix& v);
	Matrix operator-(Matrix v) const;
	Matrix operator+(Matrix matrix) const;
	Matrix operator*(Matrix matrix) const;
	Vector3D operator*(const Vector3D& vector) const;
	Matrix operator+(float scal) const;
	Matrix operator-(float scal) const;
	Matrix operator*(float scal) const;
	Matrix operator/(float scal) const;
	bool operator==(Matrix matrix) const;
	friend std::ostream& operator<<(std::ostream& oss, const Matrix& v);

private:
	float m_a00;
	float m_a01;
	float m_a02;
	float m_a10;
	float m_a11;
	float m_a12;
	float m_a20;
	float m_a21;
	float m_a22;
};
