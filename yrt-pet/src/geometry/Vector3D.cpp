/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/geometry/Vector3D.hpp"
#include "yrt-pet/geometry/Vector3D_impl.inl"

#if BUILD_PYBIND11
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
template <typename TFloat>
void py_setup_vector3dbase(py::module& m)
{
	std::string className = "Vector3D";
	if (typeid(TFloat) == typeid(double))
	{
		className += "Double";
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
	          }),
	      "x"_a, "y"_a, "z"_a);
	c.def("getNorm", &Vector3DBase<TFloat>::getNorm);
	c.def("update",
	      static_cast<void (Vector3DBase<TFloat>::*)(TFloat, TFloat, TFloat)>(
	          &Vector3DBase<TFloat>::update),
	      "x"_a, "y"_a, "z"_a);
	c.def("update",
	      static_cast<void (Vector3DBase<TFloat>::*)(
	          const Vector3DBase<TFloat>&)>(&Vector3DBase<TFloat>::update),
	      "v"_a);
	c.def("normalize", &Vector3DBase<TFloat>::normalize);
	c.def("__sub__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(
	          const Vector3DBase<TFloat>&) const>(
	          &Vector3DBase<TFloat>::operator-),
	      py::is_operator(), "other"_a);
	c.def("__isub__",
	      static_cast<Vector3DBase<TFloat>& (Vector3DBase<TFloat>::*)(
	          const Vector3DBase<TFloat>&)>(
	          &Vector3DBase<TFloat>::operator-=),
	      py::is_operator(), "other"_a);
	c.def("__add__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(
	          const Vector3DBase<TFloat>&) const>(
	          &Vector3DBase<TFloat>::operator+),
	      py::is_operator(), "other"_a);
	c.def("__iadd__",
	      static_cast<Vector3DBase<TFloat>& (Vector3DBase<TFloat>::*)(
	          const Vector3DBase<TFloat>&)>(
	          &Vector3DBase<TFloat>::operator+=),
	      py::is_operator(), "other"_a);
	c.def("__mul__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(
	          const Vector3DBase<TFloat>&) const>(
	          &Vector3DBase<TFloat>::operator*),
	      py::is_operator(), "other"_a);
	c.def("__sub__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
	                      const>(&Vector3DBase<TFloat>::operator-),
	      py::is_operator(), "scalar"_a);
	c.def("__add__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
	                      const>(&Vector3DBase<TFloat>::operator+),
	      py::is_operator(), "scalar"_a);
	c.def("__mul__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
	                      const>(&Vector3DBase<TFloat>::operator*),
	      py::is_operator(), "scalar"_a);
	c.def("__truediv__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
	                      const>(&Vector3DBase<TFloat>::operator/),
	      py::is_operator(), "scalar"_a);
	c.def(
	    "__eq__",
	    static_cast<bool (Vector3DBase<TFloat>::*)(const Vector3DBase<TFloat>&)
	                    const>(&Vector3DBase<TFloat>::operator==),
	    py::is_operator(), "other"_a);
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
}  // namespace yrt

#endif

namespace yrt
{
template Vector3DBase<double> Vector3DBase<float>::to() const;
template Vector3DBase<float> Vector3DBase<double>::to() const;

#ifndef __CUDACC__
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
#endif

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

}  // namespace yrt
