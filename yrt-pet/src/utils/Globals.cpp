/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/Globals.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

void py_setup_globals(pybind11::module& m)
{
	m.def("setNumThreads", &Globals::setNumThreads);
	m.def("getNumThreads", &Globals::getNumThreads);

	m.def("isPinnedMemoryEnabled", &GlobalsCuda::isPinnedMemoryEnabled);
	m.def("setPinnedMemoryEnabled", &GlobalsCuda::setPinnedMemoryEnabled);
}

#endif  // if BUILD_PYBIND11
