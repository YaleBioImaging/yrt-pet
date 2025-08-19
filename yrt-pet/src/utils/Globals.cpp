/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Globals.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace yrt
{
void py_setup_globals(pybind11::module& m)
{
	m.def("setVerbosityLevel", &globals::setVerbosityLevel);
	m.def("getVerbosityLevel", &globals::getVerbosityLevel);

	m.def("setNumThreads", &globals::setNumThreads);
	m.def("getNumThreads", &globals::getNumThreads);

	m.def("isPinnedMemoryEnabled", &globals::isPinnedMemoryEnabled);
	m.def("setPinnedMemoryEnabled", &globals::setPinnedMemoryEnabled);
}
}  // namespace yrt

#endif  // if BUILD_PYBIND11
