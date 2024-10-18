/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/Vector3D.hpp"

#include "geometry/Constants.hpp"

#include <cmath>

#if BUILD_PYBIND11
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;

template <typename TFloat>
void py_setup_vector3dbase(py::module& m)
{
	std::string className = "Vector3D";
	if (typeid(TFloat) == typeid(double))
	{
		className += "double";
	}

	auto c = py::class_<Vector3DBase<TFloat>>(m, className.c_str());
	c.def(py::init(
	    []()
	    {
		    return std::unique_ptr<Vector3DBase<TFloat>>(
		        new Vector3DBase<TFloat>{0., 0., 0.});
	    }));
	c.def(py::init(
	    [](TFloat x, TFloat y, TFloat z)
	    {
		    return std::unique_ptr<Vector3DBase<TFloat>>(
		        new Vector3DBase<TFloat>{x, y, z});
	    }));
	c.def("getNorm", &Vector3DBase<TFloat>::getNorm);
	c.def("update",
	      static_cast<void (Vector3DBase<TFloat>::*)(TFloat, TFloat, TFloat)>(
	          &Vector3DBase<TFloat>::update));
	c.def("update",
	      static_cast<void (Vector3DBase<TFloat>::*)(
	          const Vector3DBase<TFloat>&)>(&Vector3DBase<TFloat>::update));
	c.def("normalize", &Vector3DBase<TFloat>::normalize);
	c.def("__sub__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(
	          const Vector3DBase<TFloat>&) const>(
	          &Vector3DBase<TFloat>::operator-),
	      py::is_operator());
	c.def("__add__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(
	          const Vector3DBase<TFloat>&) const>(
	          &Vector3DBase<TFloat>::operator+),
	      py::is_operator());
	c.def("__mul__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(
	          const Vector3DBase<TFloat>&) const>(
	          &Vector3DBase<TFloat>::operator*),
	      py::is_operator());
	c.def("__sub__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
	                      const>(&Vector3DBase<TFloat>::operator-),
	      py::is_operator());
	c.def("__add__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
	                      const>(&Vector3DBase<TFloat>::operator+),
	      py::is_operator());
	c.def("__mul__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
	                      const>(&Vector3DBase<TFloat>::operator*),
	      py::is_operator());
	c.def("__div__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
	                      const>(&Vector3DBase<TFloat>::operator/),
	      py::is_operator());
	c.def(
	    "__eq__",
	    static_cast<bool (Vector3DBase<TFloat>::*)(const Vector3DBase<TFloat>&)
	                    const>(&Vector3DBase<TFloat>::operator==),
	    py::is_operator());
	c.def("__repr__",
	      [](const Vector3DBase<TFloat>& self)
	      {
		      std::stringstream ss;
		      ss << self;
		      return ss.str();
	      });
	c.def_readwrite("x", &Vector3DBase<TFloat>::x);
	c.def_readwrite("y", &Vector3DBase<TFloat>::y);
	c.def_readwrite("z", &Vector3DBase<TFloat>::z);
	c.def("isNormalized", &Vector3DBase<TFloat>::isNormalized);
	c.def("getNormalized", &Vector3DBase<TFloat>::getNormalized);
}

void py_setup_vector3dall(py::module& m)
{
	py_setup_vector3dbase<float>(m);
	py_setup_vector3dbase<double>(m);
}

#endif

template <typename TFloat>
TFloat Vector3DBase<TFloat>::getNorm() const
{
	return sqrt(getNormSquared());
}

template <typename TFloat>
TFloat Vector3DBase<TFloat>::getNormSquared() const
{
	return x * x + y * y + z * z;
}

template <typename TFloat>
void Vector3DBase<TFloat>::update(TFloat xi, TFloat yi, TFloat zi)
{
	x = xi;
	y = yi;
	z = zi;
}

// update 3:
template <typename TFloat>
void Vector3DBase<TFloat>::update(const Vector3DBase& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
}

template <typename TFloat>
Vector3DBase<TFloat>& Vector3DBase<TFloat>::normalize()
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

template <typename TFloat>
Vector3DBase<TFloat> Vector3DBase<TFloat>::getNormalized()
{
	const TFloat norm = getNorm();

	if (norm > DOUBLE_PRECISION)
	{
		return Vector3DBase{x / norm, y / norm, z / norm};
	}
	return Vector3DBase{0., 0., 0.};
}

template <typename TFloat>
bool Vector3DBase<TFloat>::isNormalized() const
{
	return std::abs(1.0 - getNorm()) < SMALL_FLT;
}

template <typename TFloat>
Vector3DBase<TFloat>
    Vector3DBase<TFloat>::operator-(const Vector3DBase& v) const
{
	return Vector3DBase{x - v.x, y - v.y, z - v.z};
}

template <typename TFloat>
Vector3DBase<TFloat>
    Vector3DBase<TFloat>::operator+(const Vector3DBase& v) const
{
	return Vector3DBase{x + v.x, y + v.y, z + v.z};
}

template <typename TFloat>
TFloat Vector3DBase<TFloat>::scalProd(const Vector3DBase& vector) const
{
	return x * vector.x + y * vector.y + z * vector.z;
}

template <typename TFloat>
Vector3DBase<TFloat>
    Vector3DBase<TFloat>::crossProduct(const Vector3DBase& B) const
{
	return Vector3DBase{y * B.z - z * B.y, z * B.x - x * B.z,
	                    x * B.y - y * B.x};
}

template <typename TFloat>
void Vector3DBase<TFloat>::linearTransformation(const Vector3DBase& i,
                                                const Vector3DBase& j,
                                                const Vector3DBase& k)
{
	this->x = i.x * this->x + j.x * this->y + k.x * this->z;
	this->y = i.y * this->x + j.y * this->y + k.y * this->z;
	this->z = i.z * this->x + j.z * this->y + k.z * this->z;
}

template <typename TFloat>
int Vector3DBase<TFloat>::argmax()
{
	return (x > y) ? ((x > z) ? 0 : 2) : ((y > z) ? 1 : 2);
}

template <typename TFloat>
TFloat Vector3DBase<TFloat>::operator[](int idx) const
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

template <typename TFloat>
TFloat Vector3DBase<TFloat>::operator[](int idx)
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


template <typename TFloat>
Vector3DBase<TFloat>
    Vector3DBase<TFloat>::operator*(const Vector3DBase& vector) const
{
	return Vector3DBase{y * vector.z - z * vector.y,
	                    z * vector.x - x * vector.z,
	                    x * vector.y - y * vector.x};
}

template <typename TFloat>
Vector3DBase<TFloat> Vector3DBase<TFloat>::operator+(TFloat scal) const
{
	return Vector3DBase{x + scal, y + scal, z + scal};
}

template <typename TFloat>
Vector3DBase<TFloat> Vector3DBase<TFloat>::operator-(TFloat scal) const
{
	return Vector3DBase{x - scal, y - scal, z - scal};
}

template <typename TFloat>
Vector3DBase<TFloat> Vector3DBase<TFloat>::operator*(TFloat scal) const
{
	return Vector3DBase{x * scal, y * scal, z * scal};
}

template <typename TFloat>
Vector3DBase<TFloat> Vector3DBase<TFloat>::operator/(TFloat scal) const
{
	if (std::abs(scal) > DOUBLE_PRECISION)
	{
		return Vector3DBase<TFloat>{x / scal, y / scal, z / scal};
	}
	return Vector3DBase<TFloat>{LARGE_VALUE, LARGE_VALUE, LARGE_VALUE};
}

// return true if vectors are the same:
template <typename TFloat>
bool Vector3DBase<TFloat>::operator==(const Vector3DBase& vector) const
{
	const Vector3DBase tmp{x - vector.x, y - vector.y, z - vector.z};
	if (tmp.getNorm() < SMALL_FLT)
	{
		return true;
	}
	return false;
}

template <typename TFloat>
template <typename TargetType>
Vector3DBase<TargetType> Vector3DBase<TFloat>::to() const
{
	return Vector3DBase<TargetType>{static_cast<TargetType>(x),
	                                static_cast<TargetType>(y),
	                                static_cast<TargetType>(z)};
}
template Vector3DBase<double> Vector3DBase<float>::to() const;
template Vector3DBase<float> Vector3DBase<double>::to() const;

template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const Vector3DBase<TFloat>& v)
{
	oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
	return oss;
}

template std::ostream& operator<<(std::ostream& oss,
                                  const Vector3DBase<double>& v);
template std::ostream& operator<<(std::ostream& oss,
                                  const Vector3DBase<float>& v);

template class Vector3DBase<double>;
template class Vector3DBase<float>;

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
