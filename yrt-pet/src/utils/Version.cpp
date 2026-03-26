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
}
}  // namespace yrt

#endif

namespace yrt
{
namespace version
{

std::string getVersionString()
{
	return versionString;
}

std::string getGitHash()
{
	std::string version = versionString;

	if (!version.ends_with("-dirty"))
	{
		size_t dashPos = version.rfind('-');
		if (dashPos != std::string::npos && dashPos + 1 < version.length())
		{
			return version.substr(dashPos + 1);
		}
	}
	else
	{
		std::string versionNoDirty = version.substr(0, version.length() - 6);
		size_t dashPos = versionNoDirty.rfind('-');
		if (dashPos != std::string::npos &&
		    dashPos + 1 < versionNoDirty.length())
		{
			return versionNoDirty.substr(dashPos + 1);
		}
	}

	return "";
}

bool isDirty()
{
	std::string version = versionString;
	return version.ends_with("-dirty");
}

void printVersion()
{
	std::cout << versionString << std::endl;
}


}  // namespace version
}  // namespace yrt
