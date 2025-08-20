/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Globals.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_globals(pybind11::module& m)
{
	py::enum_<globals::VerbositySection>(m, "VerbositySection")
	    .value("GENERAL", globals::VerbositySection::GENERAL)
	    .value("ALLOCATION", globals::VerbositySection::ALLOCATION)
	    .value("FILESYSTEM", globals::VerbositySection::FILESYSTEM)
	    .value("IMAGE", globals::VerbositySection::IMAGE)
	    .value("PROJECTOR", globals::VerbositySection::PROJECTOR)
	    .value("CORRECTOR", globals::VerbositySection::CORRECTOR)
	    .value("OPTIMIZER", globals::VerbositySection::OPTIMIZER)
	    .export_values();

	m.def("setVerbosityLevel", &globals::setVerbosityLevel, "section"_a,
	      "level"_a);
	m.def("getVerbosityLevel", &globals::getVerbosityLevel, "section"_a);

	m.def("getNumThreads", &globals::getNumThreads);

	m.def("isPinnedMemoryEnabled", &globals::isPinnedMemoryEnabled);
	m.def("setPinnedMemoryEnabled", &globals::setPinnedMemoryEnabled);
}
}  // namespace yrt

#endif  // if BUILD_PYBIND11
