/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Version.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace yrt
{
void py_setup_version(py::module& m)
{
	m.def("getVersionString", &version::getVersionString);
	m.def("getGitHash", &version::getGitHash);
	m.def("isDirty", &version::isDirty);
	m.def("printVersion", &version::printVersion);
	m.attr("versionString") = version::versionString;
}
}  // namespace yrt

#endif
