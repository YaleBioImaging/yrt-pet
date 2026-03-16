/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Array.hpp"
#include <vector>

#if BUILD_PYBIND11
#include "yrt-pet/utils/pybind11.hpp"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{

template <typename T>
void declare_array_base(py::module& m, const std::string& dtype_name)
{
	std::string pyclass_name =
	    "Array" + std::to_string(T::NDim) + "D" + dtype_name + "Base";

	auto c = py::class_<T>(m, pyclass_name.c_str(), py::buffer_protocol(),
	                       py::dynamic_attr());
	c.def_buffer(
	    [](T& d) -> py::buffer_info
	    {
		    return py::buffer_info(
		        d.getRawPointer(), sizeof(typename T::DType),
		        py::format_descriptor<typename T::DType>::format(), T::NDim,
		        d.getDims(), d.getStrides());
	    });

	c.def("isMemoryValid", &T::isMemoryValid);
	c.def("getFlatIdx", &T::getFlatIdx);
	c.def("unravelIdx", &T::unravelIdx);
	if constexpr (T::NDim != 1)
	{
		c.def("__getitem__",
		      static_cast<T::DType& (T::*)(const std::array<size_t, T::NDim>&)
		                      const>(&T::get));
		c.def("__setitem__",
		      static_cast<void (T::*)(const std::array<size_t, T::NDim>&,
		                              typename T::DType val)>(&T::set));
	}
	else
	{
		c.def("__getitem__",
		      static_cast<T::DType& (T::*)(size_t) const>(&T::getFlat));
		c.def("__setitem__",
		      static_cast<void (T::*)(size_t, typename T::DType)>(&T::setFlat));
	}
	c.def("incrementFlat", &T::incrementFlat, "idx"_a, "val"_a);
	c.def("setFlat", &T::setFlat, "idx"_a, "val"_a);
	c.def("getFlat", &T::getFlat, "idx"_a);
	c.def("getSize", &T::getSize, "dim"_a);
	c.def("getSizeTotal", &T::getSizeTotal);
	c.def("getDims",
	      static_cast<std::array<size_t, T::NDim> (T::*)() const>(&T::getDims));
	c.def("getStrides", &T::getStrides);
	c.def("getStridesIdx", &T::getStridesIdx);
	c.def("fill", &T::fill, "val"_a);
	c.def("writeToFile", &T::writeToFile, "fname"_a);
	c.def("readFromFile",
	      static_cast<void (T::*)(const std::string&)>(&T::readFromFile));
	c.def("readFromFile",
	      static_cast<void (T::*)(const std::string&,
	                              const std::array<size_t, T::NDim>&)>(
	          &T::readFromFile));
	c.def(py::self += py::self);
	c.def(py::self += typename T::DType());
	if constexpr (!std::is_same_v<typename T::DType, bool>)
	{
		c.def(py::self *= py::self);
		c.def(py::self *= typename T::DType());
	}
	c.def(py::self /= py::self);
	c.def(py::self /= typename T::DType());
	c.def(py::self -= py::self);
	c.def(py::self -= typename T::DType());
	c.def("invert", &T::invert);
	c.def("getMaxValue", &T::getMaxValue);
	c.def("sum", &T::sum);
	c.def("copy", [](T& self, const T& src) { self.copy(src); }, "src"_a);
}

template <typename T>
void declare_array_child(py::module& m, const std::string& dtype_name)
{
	std::string pyclass_name =
	    "Array" + std::to_string(T::NDim) + "D" + dtype_name;
	if constexpr (T::IsOwned)
	{
		pyclass_name += "Owned";
	}
	else
	{
		pyclass_name += "Alias";
	}

	auto c = py::class_<T, typename T::BaseClass>(
	    m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	c.def(py::init<>());

	if constexpr (T::IsOwned)
	{
		c.def("allocate", &T::allocate);
	}
	else
	{
		c.def("bind",
		      static_cast<void (T::*)(const typename T::BaseClass&)>(&T::bind));

		c.def("bind", &T::bindFromNumpy, "numpy_array"_a);
	}
}

template <typename DType>
void declare_array(py::module& m, const std::string& dtype_name)
{
	declare_array_base<Array1DBase<DType>>(m, dtype_name);
	declare_array_child<Array1DOwned<DType>>(m, dtype_name);
	declare_array_child<Array1DAlias<DType>>(m, dtype_name);

	declare_array_base<Array2DBase<DType>>(m, dtype_name);
	declare_array_child<Array2DOwned<DType>>(m, dtype_name);
	declare_array_child<Array2DAlias<DType>>(m, dtype_name);

	declare_array_base<Array3DBase<DType>>(m, dtype_name);
	declare_array_child<Array3DOwned<DType>>(m, dtype_name);
	declare_array_child<Array3DAlias<DType>>(m, dtype_name);

	declare_array_base<Array4DBase<DType>>(m, dtype_name);
	declare_array_child<Array4DOwned<DType>>(m, dtype_name);
	declare_array_child<Array4DAlias<DType>>(m, dtype_name);

	declare_array_base<Array5DBase<DType>>(m, dtype_name);
	declare_array_child<Array5DOwned<DType>>(m, dtype_name);
	declare_array_child<Array5DAlias<DType>>(m, dtype_name);
}

void py_setup_array(py::module& m)
{
	declare_array<float>(m, "Float");
	declare_array<double>(m, "Double");
	declare_array<int>(m, "Int");
	declare_array<bool>(m, "Bool");
}

}  // namespace yrt

#endif  // if BUILD_PYBIND11
