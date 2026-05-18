/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Version.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_version(py::module& m)
{
	auto c = py::class_<version::VersionStruct>(m, "VersionStruct");
	c.def(py::init<>());
	c.def(py::init<int, int, int, std::string, bool>(), "major"_a, "minor"_a,
	      "patch"_a, "hash"_a, "isDirty"_a = false);
	c.def_readonly("major", &version::VersionStruct::major);
	c.def_readonly("minor", &version::VersionStruct::minor);
	c.def_readonly("patch", &version::VersionStruct::patch);
	c.def_readonly("hash", &version::VersionStruct::hash);
	c.def_readonly("isDirty", &version::VersionStruct::isDirty);
	c.def("__str__", &version::encodeVersion);
	c.def("__repr__", &version::encodeVersion);
	c.def("__eq__", &version::VersionStruct::operator==);
	c.def("__gt__", &version::VersionStruct::operator>);
	c.def("__lt__", &version::VersionStruct::operator<);
	c.def("__ge__", &version::VersionStruct::operator>=);
	c.def("__le__", &version::VersionStruct::operator<=);

	auto cs =
	    py::class_<version::SimpleVersionStruct>(m, "SimpleVersionStruct");
	cs.def(py::init<>());
	cs.def(py::init<int, int>(), "major"_a, "minor"_a);
	cs.def_readonly("major", &version::SimpleVersionStruct::major);
	cs.def_readonly("minor", &version::SimpleVersionStruct::minor);
	cs.def("__str__", &version::encodeVersionSimple);
	cs.def("__repr__", &version::encodeVersionSimple);
	cs.def("__eq__", &version::SimpleVersionStruct::operator==);
	cs.def("__gt__", &version::SimpleVersionStruct::operator>);
	cs.def("__lt__", &version::SimpleVersionStruct::operator<);
	cs.def("__ge__", &version::SimpleVersionStruct::operator>=);
	cs.def("__le__", &version::SimpleVersionStruct::operator<=);

	m.def("getVersionString", &version::getVersionString);
	m.def("getGitHash", &version::getGitHash);
	m.def("isDirty", &version::isDirty);
	m.def("printVersion", &version::printVersion);
	m.def("getVersion", &version::getVersion);
	m.def("decodeVersion", &version::decodeVersion);
	m.def("encodeVersion", &version::encodeVersion);
	m.def("decodeVersionSimple", &version::decodeVersionSimple);
	m.def("encodeVersion", &version::encodeVersion);
}
}  // namespace yrt

#endif

namespace yrt::version
{

bool VersionStruct::operator==(const VersionStruct& other) const
{
	return major == other.major && minor == other.minor && patch == other.patch;
}

bool VersionStruct::operator>(const VersionStruct& other) const
{
	if (major != other.major)
	{
		return major > other.major;
	}
	if (minor != other.minor)
	{
		return minor > other.minor;
	}
	return patch > other.patch;
}

bool VersionStruct::operator<(const VersionStruct& other) const
{
	if (major != other.major)
	{
		return major < other.major;
	}
	if (minor != other.minor)
	{
		return minor < other.minor;
	}
	return patch < other.patch;
}

bool VersionStruct::operator>=(const VersionStruct& other) const
{
	return !(*this < other);
}

bool VersionStruct::operator<=(const VersionStruct& other) const
{
	return !(*this > other);
}

std::string getVersionString()
{
	return versionString;
}

VersionStruct getVersion()
{
	return decodeVersion(getVersionString());
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

VersionStruct decodeVersion(const std::string& pr_versionString)
{
	VersionStruct vs;
	vs.major = 0;
	vs.minor = 0;
	vs.patch = 0;
	vs.hash = "";
	vs.isDirty = false;

	std::string version = pr_versionString;

	vs.isDirty = version.ends_with("-dirty");
	if (vs.isDirty)
	{
		version = version.substr(0, version.length() - 6);
	}

	const size_t dashPos = version.rfind('-');
	if (dashPos != std::string::npos && dashPos + 1 < version.length())
	{
		vs.hash = version.substr(dashPos + 1);
		version = version.substr(0, dashPos);
	}

	const size_t firstDot = version.find('.');
	const size_t secondDot = version.find('.', firstDot + 1);

	if (firstDot != std::string::npos)
	{
		vs.major = std::stoi(version.substr(0, firstDot));
	}

	if (secondDot != std::string::npos)
	{
		vs.minor =
		    std::stoi(version.substr(firstDot + 1, secondDot - firstDot - 1));
		vs.patch = std::stoi(version.substr(secondDot + 1));
	}
	else if (firstDot != std::string::npos)
	{
		vs.minor = std::stoi(version.substr(firstDot + 1));
	}

	return vs;
}

std::string encodeVersion(const VersionStruct& vs)
{
	std::string result = std::to_string(vs.major) + "." +
	                     std::to_string(vs.minor) + "." +
	                     std::to_string(vs.patch);
	if (!vs.hash.empty())
	{
		result += "-" + vs.hash;
	}
	if (vs.isDirty)
	{
		result += "-dirty";
	}
	return result;
}

SimpleVersionStruct decodeVersionSimple(const std::string& pr_versionString)
{
	SimpleVersionStruct vs;
	vs.major = 0;
	vs.minor = 0;

	std::string version = pr_versionString;

	const size_t dotPos = version.find('.');
	if (dotPos != std::string::npos)
	{
		vs.major = std::stoi(version.substr(0, dotPos));
		vs.minor = std::stoi(version.substr(dotPos + 1));
	}

	return vs;
}

std::string encodeVersionSimple(const SimpleVersionStruct& vs)
{
	return std::to_string(vs.major) + "." + std::to_string(vs.minor);
}

bool SimpleVersionStruct::operator==(const SimpleVersionStruct& other) const
{
	return major == other.major && minor == other.minor;
}

bool SimpleVersionStruct::operator>(const SimpleVersionStruct& other) const
{
	if (major != other.major)
	{
		return major > other.major;
	}
	return minor > other.minor;
}

bool SimpleVersionStruct::operator<(const SimpleVersionStruct& other) const
{
	if (major != other.major)
	{
		return major < other.major;
	}
	return minor < other.minor;
}

bool SimpleVersionStruct::operator>=(const SimpleVersionStruct& other) const
{
	return !(*this < other);
}

bool SimpleVersionStruct::operator<=(const SimpleVersionStruct& other) const
{
	return !(*this > other);
}

void printVersion()
{
	std::cout << versionString << std::endl;
}


}  // namespace yrt::version
