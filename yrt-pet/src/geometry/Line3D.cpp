/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/Line3D.hpp"
#include "geometry/Constants.hpp"

#if BUILD_PYBIND11
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;

template <typename TFloat>
void py_setup_line3dbase(py::module& m)
{
	std::string className = "Line3D";
	if (typeid(TFloat) == typeid(double))
	{
		className += "double";
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
	    }));
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
}

void py_setup_line3dall(py::module& m)
{
	py_setup_line3dbase<float>(m);
	py_setup_line3dbase<double>(m);
}

#endif

template <typename TFloat>
TFloat Line3DBase<TFloat>::getNorm() const
{
	return (point1 - point2).getNorm();
}

template <typename TFloat>
template <typename TargetType>
Line3DBase<TargetType> Line3DBase<TFloat>::to() const
{
	const Vector3DBase<TargetType> newPoint1{static_cast<TargetType>(point1.x),
	                                         static_cast<TargetType>(point1.y),
	                                         static_cast<TargetType>(point1.z)};
	const Vector3DBase<TargetType> newPoint2{static_cast<TargetType>(point2.x),
	                                         static_cast<TargetType>(point2.y),
	                                         static_cast<TargetType>(point2.z)};

	return Line3DBase<TargetType>{newPoint1, newPoint2};
}
template Line3DBase<double> Line3DBase<float>::to() const;
template Line3DBase<float> Line3DBase<double>::to() const;

template <typename TFloat>
Line3DBase<TFloat> Line3DBase<TFloat>::nullLine()
{
	return Line3DBase{Vector3DBase<TFloat>{0., 0., 0.},
	                  Vector3DBase<TFloat>{0., 0., 0.}};
}

template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const Line3DBase<TFloat>& l)
{
	oss << l.point1 << ", " << l.point2 << std::endl;
	return oss;
}

template <typename TFloat>
void Line3DBase<TFloat>::update(const Vector3DBase<TFloat>& pt1,
                                const Vector3DBase<TFloat>& pt2)
{
	point1 = pt1;
	point2 = pt2;
}

template <typename TFloat>
bool Line3DBase<TFloat>::isEqual(Line3DBase<TFloat>& line) const
{
	const Vector3DBase point1Diff = point1 - line.point1;
	const TFloat distPointSquared1 = point1Diff.getNormSquared();
	if (distPointSquared1 > SMALL)
	{
		return false;
	}
	const Vector3DBase point2Diff = point2 - line.point2;
	const TFloat distPointSquared2 = point2Diff.getNormSquared();
	if (distPointSquared2 > SMALL)
	{
		return false;
	}
	return true;
}

template <typename TFloat>
bool Line3DBase<TFloat>::isParallel(Line3DBase<TFloat>& line) const
{
	const TFloat b = point1.x;
	const TFloat d = point1.y;
	const TFloat f = point1.z;
	const TFloat a = point2.x - b;
	const TFloat c = point2.y - d;
	const TFloat e = point2.z - f;

	const TFloat lb = line.point1.x;
	const TFloat ld = line.point1.y;
	const TFloat lf = line.point1.z;
	const TFloat la = line.point2.x - lb;
	const TFloat lc = line.point2.y - ld;
	const TFloat le = line.point2.z - lf;

	Vector3DBase<TFloat> tmp1{a, c, e};  // Orientation of current line
	tmp1.normalize();
	Vector3DBase<TFloat> tmp2{la, lc, le};  // Orientation of other line
	tmp2.normalize();
	const Vector3DBase<TFloat> crossProd{tmp1.y * tmp2.z - tmp1.z * tmp2.y,
	                                     tmp1.z * tmp2.x - tmp1.x * tmp2.z,
	                                     tmp1.x * tmp2.y - tmp1.y * tmp2.x};
	const TFloat norm = crossProd.getNorm();
	return norm <= SMALL;
}

template class Line3DBase<double>;
template class Line3DBase<float>;

template std::ostream& operator<<(std::ostream& oss,
                                  const Line3DBase<double>& l);
template std::ostream& operator<<(std::ostream& oss,
                                  const Line3DBase<float>& l);

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
