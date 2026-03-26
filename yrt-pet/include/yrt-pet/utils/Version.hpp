/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <iostream>
#include <string>

namespace yrt
{

namespace version
{

static constexpr const char* versionString = YRTPET_VERSION_STRING;

inline std::string getVersionString()
{
	return versionString;
}

inline std::string getGitHash()
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

inline bool isDirty()
{
	std::string version = versionString;
	return version.ends_with("-dirty");
}

inline void printVersion()
{
	std::cout << versionString << std::endl;
}

};  // namespace version

};  // namespace yrt
