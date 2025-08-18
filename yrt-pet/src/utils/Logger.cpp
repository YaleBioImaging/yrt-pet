/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Logger.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_log(pybind11::module& m)
{
	m.def("log1", &log<1>);
	m.def("log2", &log<2>);
	m.def("log3", &log<3>);
	m.def("log4", &log<4>);
	m.def("log5", &log<5>);
}
}  // namespace yrt
#endif

namespace yrt
{

template <int LEVEL>
Logger<LEVEL>& Logger<LEVEL>::operator<<(std::ostream& (*manip)(std::ostream&))
{
	if (globals::getVerbosityLevel() >= LEVEL)
	{
		std::cout << manip;
	}
	return *this;
}

template class Logger<1>;  // Standard
template class Logger<2>;
template class Logger<3>;
template class Logger<4>;
template class Logger<5>;

}  // namespace yrt
