/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Globals.hpp"

namespace yrt
{

// A logging proxy object for a given LEVEL
template <int LEVEL>
class Logger
{
public:
	// Custom stream buffer that writes to std::cout if verbosity allows
	template <typename T>
	Logger& operator<<(const T& value)
	{
		if (globals::getVerbosityLevel() >= LEVEL)
		{
			std::cout << value;
		}
		return *this;
	}

	// Special case for manipulators like std::endl
	Logger& operator<<(std::ostream& (*manip)(std::ostream&));

	// We let verbosity level 0 be complete silence
	static_assert(LEVEL > 0);

	// We shouldn't allow for more than 5 levels of verbosity
	static_assert(LEVEL <= 5);
};

// Convenient alias
template <int LEVEL>
inline Logger<LEVEL> log{};

/*
 * Verbosity logger usage:
 * yrt::log<1> << "My message" << std::endl; // Standard level
 * yrt::log<2> << "Number of things: " << 45 << std::endl; // Level 2
 * yrt::log<3> << "Detailed number of things: " << 92.5 << std::endl; // Level 3
 * ...
 * yrt::log<5> << "Debug stuff" << std::endl; // Level 5
 */

}  // namespace yrt
