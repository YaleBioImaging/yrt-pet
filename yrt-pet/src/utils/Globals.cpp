/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/Globals.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

void py_setup_globals(pybind11::module& m)
{
	m.def("set_num_threads", &Globals::set_num_threads);
	m.def("get_num_threads", &Globals::get_num_threads);

	m.def("isPinnedMemoryEnabled", &GlobalsCuda::isPinnedMemoryEnabled);
	m.def("setPinnedMemoryEnabled", &GlobalsCuda::setPinnedMemoryEnabled);
	m.def("getMaxVRAM", &GlobalsCuda::getMaxVRAM);
	m.def("setMaxVRAM", &GlobalsCuda::setMaxVRAM);
}

#endif  // if BUILD_PYBIND11
