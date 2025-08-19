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
	      [](int level, globals::VerbositySection section,
	         const std::string& message)
	      {
		      ASSERT_MSG(level >= 1 && level <= 5,
		                 "Log level has to be inclusively between 1 and 5");
		      switch (level)
		      {
		      case 1: log<1>(section, message); break;
		      case 2: log<2>(section, message); break;
		      case 3: log<3>(section, message); break;
		      case 4: log<4>(section, message); break;
		      case 5: log<5>(section, message); break;
		      default:
			      // Should never reach here.
			      throw std::runtime_error("Unknown error");
		      }
	      });
}
}  // namespace yrt
#endif
