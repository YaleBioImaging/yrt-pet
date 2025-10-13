/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/TimeOfFlight.hpp"

#include "yrt-pet/utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_timeofflight(py::module& m)
{
	auto c = py::class_<TimeOfFlightHelper>(m, "TimeOfFlightHelper");
	c.def(py::init<float, int>(), "tof_width_ps"_a, "tof_n_std"_a);
	c.def("getAlphaRange", &TimeOfFlightHelper::getAlphaRange);
	c.def("getWeight", &TimeOfFlightHelper::getWeight);
	c.def("getSigma", &TimeOfFlightHelper::getSigma);
	c.def("getTruncWidth", &TimeOfFlightHelper::getTruncWidth);
	c.def("getNorm", &TimeOfFlightHelper::getNorm);
}
}  // namespace yrt

#endif

namespace yrt
{


}  // namespace yrt
