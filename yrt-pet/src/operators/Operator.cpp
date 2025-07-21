/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/Operator.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace yrt
{
void py_setup_operator(py::module& m)
{
	auto c = py::class_<Operator>(m, "Operator");
}
}  // namespace yrt

#endif
