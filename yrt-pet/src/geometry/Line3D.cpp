/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/geometry/Line3D_impl.inl"

#if BUILD_PYBIND11
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{

template <typename TFloat>
void py_setup_line3dbase(py::module& m)
{
	std::string className = "Line3D";
	if (typeid(TFloat) == typeid(double))
	{
		className += "Double";
	}

	auto c = py::class_<Line3DBase<TFloat>>(m, className.c_str());
	c.def(py::init(
	    []()
	    {
		    return std::unique_ptr<Line3DBase<TFloat>>(
		        new Line3DBase<TFloat>{Line3DBase<TFloat>::nullLine()});
	    }));
	c.def(py::init(
	          [](Vector3DBase<TFloat> point1, Vector3DBase<TFloat> point2)
	          {
		          return std::unique_ptr<Line3DBase<TFloat>>(
		              new Line3DBase<TFloat>{point1, point2});
	          }),
	      "p1"_a, "p2"_a);
	c.def("getNorm", &Line3DBase<TFloat>::getNorm);
	c.def("isEqual", &Line3DBase<TFloat>::isEqual);
	c.def("isParallel", &Line3DBase<TFloat>::isParallel);
	c.def("update", &Line3DBase<TFloat>::update, py::arg("pt1"),
	      py::arg("pt2"));
	c.def("__repr__",
	      [](const Line3DBase<TFloat>& self)
	      {
		      std::stringstream ss;
		      ss << self;
		      return ss.str();
	      });
	c.def_readwrite("point1", &Line3DBase<TFloat>::point1);
	c.def_readwrite("point2", &Line3DBase<TFloat>::point2);
	c.def("toTuple",
	      [](Line3DBase<TFloat>& self)
	      {
		      return py::make_tuple(
		          py::make_tuple(self.point1.x, self.point1.y, self.point1.z),
		          py::make_tuple(self.point2.x, self.point2.y, self.point2.z));
	      });
}

void py_setup_line3dall(py::module& m)
{
	py_setup_line3dbase<float>(m);
	py_setup_line3dbase<double>(m);
}

}  // namespace yrt
#endif

namespace yrt
{
template Line3DBase<double> Line3DBase<float>::to() const;
template Line3DBase<float> Line3DBase<double>::to() const;

#ifndef __CUDACC__
template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const Line3DBase<TFloat>& l)
{
	oss << l.point1 << ", " << l.point2 << std::endl;
	return oss;
}

template std::ostream& operator<<(std::ostream& oss,
                                  const Line3DBase<double>& l);
template std::ostream& operator<<(std::ostream& oss,
                                  const Line3DBase<float>& l);
#endif

template class Line3DBase<double>;
template class Line3DBase<float>;

static_assert(std::is_trivially_constructible<Line3DBase<double>>());
static_assert(std::is_trivially_destructible<Line3DBase<double>>());
static_assert(std::is_trivially_copyable<Line3DBase<double>>());
static_assert(std::is_trivially_copy_constructible<Line3DBase<double>>());
static_assert(std::is_trivially_copy_assignable<Line3DBase<double>>());
static_assert(std::is_trivially_default_constructible<Line3DBase<double>>());
static_assert(std::is_trivially_move_assignable<Line3DBase<double>>());
static_assert(std::is_trivially_move_constructible<Line3DBase<double>>());

static_assert(std::is_trivially_constructible<Line3DBase<float>>());
static_assert(std::is_trivially_destructible<Line3DBase<float>>());
static_assert(std::is_trivially_copyable<Line3DBase<float>>());
static_assert(std::is_trivially_copy_constructible<Line3DBase<float>>());
static_assert(std::is_trivially_copy_assignable<Line3DBase<float>>());
static_assert(std::is_trivially_default_constructible<Line3DBase<float>>());
static_assert(std::is_trivially_move_assignable<Line3DBase<float>>());
static_assert(std::is_trivially_move_constructible<Line3DBase<float>>());

}  // namespace yrt
