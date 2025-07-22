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

TimeOfFlightHelper::TimeOfFlightHelper(float tof_width_ps, int tof_n_std)
{
	const double tof_width_mm = tof_width_ps * SPEED_OF_LIGHT_MM_PS * 0.5;
	// FWHM = sigma 2 sqrt(2 ln 2)
	m_sigma = tof_width_mm / (2 * sqrtf(2 * logf(2)));
	if (tof_n_std <= 0)
	{
		m_truncWidth_mm = -1.f;
	}
	else
	{
		m_truncWidth_mm = tof_n_std * m_sigma;
	}
	m_norm = 1 / (std::sqrt(2 * PI) * m_sigma);
}

float TimeOfFlightHelper::getSigma() const
{
	return m_sigma;
}

float TimeOfFlightHelper::getTruncWidth() const
{
	return m_truncWidth_mm;
}

float TimeOfFlightHelper::getNorm() const
{
	return m_norm;
}
}  // namespace yrt
