/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/Variable.hpp"

#if BUILD_PYBIND11

#include <pybind11/pybind11.h>

namespace py = pybind11;

void py_setup_variable(py::module& m)
{
	// Variable is added here because the class is empty
	auto c = py::class_<Variable>(m, "Variable");
}

#endif
