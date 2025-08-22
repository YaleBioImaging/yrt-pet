/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Array.hpp"
#include <vector>

#if BUILD_PYBIND11
#include "yrt-pet/utils/pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace yrt
{

template <class T, typename U, int ndim>
void declare_array(pybind11::module& m, std::string type_str)
{
	std::string pyclass_name = std::string("Array") + std::to_string(ndim) +
	                           std::string("D") + type_str;
	auto c = pybind11::class_<T>(m, pyclass_name.c_str(),
	                             pybind11::buffer_protocol(),
	                             pybind11::dynamic_attr());
	c.def(pybind11::init<>());
	c.def_buffer(
	    [](T& d) -> pybind11::buffer_info
	    {
		    return pybind11::buffer_info(
		        d.getRawPointer(), sizeof(U),
		        pybind11::format_descriptor<U>::format(), ndim, d.getDims(),
		        d.getStrides());
	    });
	c.def("allocate", &T::allocate);
	c.def("readFromFile",
	      static_cast<void (T::*)(const std::string&)>(&T::readFromFile));
	c.def("readFromFile",
	      static_cast<void (T::*)(const std::string&,
	                              const std::array<size_t, ndim>&)>(
	          &T::readFromFile));
	c.def("writeToFile", &T::writeToFile);
	c.def("getSize", &T::getSize);
	c.def("getStrides", &T::getStrides);
	c.def("getSizeTotal", &T::getSizeTotal);
	c.def("getMaxValue", &T::getMaxValue);
	c.def("getFlatIdx",
	      static_cast<size_t (T::*)(const std::array<size_t, ndim>&) const>(
	          &T::getFlatIdx));
	c.def("getFlat", &T::getFlat);
	c.def("setFlat", &T::setFlat);
	c.def(
	    "__getitem__",
	    static_cast<U& (T::*)(const std::array<size_t, ndim>&) const>(&T::get));
	c.def("__setitem__",
	      static_cast<void (T::*)(const std::array<size_t, ndim>&, U val)>(
	          &T::set));
	c.def("fill", &T::fill);
}

void py_setup_array(pybind11::module& m)
{
	// TODO: Add Python bindings for the "Alias" type

	// Add common array types
	PY_DECLARE_ARRAY(float, 1);
	PY_DECLARE_ARRAY(float, 2);
	PY_DECLARE_ARRAY(float, 3);
	PY_DECLARE_ARRAY(double, 1);
	PY_DECLARE_ARRAY(double, 2);
	PY_DECLARE_ARRAY(double, 3);
	PY_DECLARE_ARRAY(int, 1);
	PY_DECLARE_ARRAY(int, 2);
	PY_DECLARE_ARRAY(int, 3);
}

}  // namespace yrt

#endif  // if BUILD_PYBIND11
