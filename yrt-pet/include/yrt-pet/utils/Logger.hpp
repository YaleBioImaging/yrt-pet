/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Globals.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

namespace yrt
{

// A logging proxy object for a given LEVEL
template <int LEVEL>
class Logger
{
public:
	// Custom stream buffer that writes to std::cout if verbosity allows
	template <typename T>
	Logger& operator()(globals::VerbositySection section, const T& value)
	{
		if (globals::getVerbosityLevel(section) >= LEVEL)
		{
			const auto now = std::chrono::system_clock::now();
			const std::time_t t = std::chrono::system_clock::to_time_t(now);

			// Convert to local time
			const std::tm* tm = std::localtime(&t);

			// Print with i/o manipulators
			std::cout << "[" << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "] "
			          << value << std::endl;
		}
		return *this;
	}

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
 * yrt::log<1>(GENERAL, "My message"); // Standard level
 * yrt::log<2>(ALLOCATION, "Number of things: " + std::to_string(45)); // Level
 * 2 yrt::log<3>(GENERAL, "Detailed info: " + std::to_string(415)); // Level 3
 * ...
 * yrt::log<5>(PROJECTOR, "Debug stuff"); // Level 5
 */

}  // namespace yrt
