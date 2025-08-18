/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Logger.hpp"
#include "yrt-pet/utils/Assert.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_log(pybind11::module& m)
{
	m.def("log",
	      [](int level, const std::string& message)
	      {
		      ASSERT_MSG(level >= 1 && level <= 5,
		                 "Log level has to be inclusively between 1 and 5");
		      if (level == 1)
		      {
			      log<1> << message << std::endl;
		      }
		      else if (level == 2)
		      {
			      log<2> << message << std::endl;
		      }
		      else if (level == 3)
		      {
			      log<3> << message << std::endl;
		      }
		      else if (level == 4)
		      {
			      log<4> << message << std::endl;
		      }
		      else if (level == 5)
		      {
			      log<5> << message << std::endl;
		      }
	      });
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
