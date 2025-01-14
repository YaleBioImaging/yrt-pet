/*
* This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/Operator.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;

void py_setup_operator(py::module& m)
{
	auto c = py::class_<Operator>(m, "Operator");
}

#endif
