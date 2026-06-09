/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/Histogram.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_histogram(py::module& m)
{
	auto c = py::class_<Histogram, ProjectionData>(m, "Histogram");
	c.def("getProjectionValueFromHistogramBin",
	      &Histogram::getProjectionValueFromHistogramBin, "histo_bin_id"_a);
}
}  // namespace yrt

#endif

namespace yrt
{
Histogram::Histogram(const Scanner& pr_scanner) : ProjectionData{pr_scanner} {}
}  // namespace yrt
