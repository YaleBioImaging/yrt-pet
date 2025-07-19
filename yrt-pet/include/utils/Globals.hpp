/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "omp.h"
#include "utils/Utilities.hpp"

#include <cstddef>
#include <iostream>

namespace Globals
{
	inline int& num_threads()
	{
		static int s_num_threads = []()
		{
			// Initialization
			return omp_get_max_threads();
		}();
		return s_num_threads;
	}

	inline int getNumThreads()
	{
		return num_threads();
	}

	inline void setNumThreads(int t)
	{
		if (t <= 0)
		{
			num_threads() = omp_get_max_threads();
		}
		else
		{
			num_threads() = t;
		}
		std::cout << "Using " << num_threads() << " threads." << std::endl;
		omp_set_num_threads(num_threads());
	}

};  // namespace Globals

namespace GlobalsCuda
{
	// TODO NOW: Document this
	static constexpr char DisablePinnedMemoryEnvVar[] =
	    "YRTPET_DISABLE_PINNED_MEMORY";

	inline bool& usePinnedMemory()
	{
		static bool s_usePinnedMemory = []()
		{
			const auto disablePinnedMemoryValue_opt =
			    Util::getEnv(DisablePinnedMemoryEnvVar);
			if (disablePinnedMemoryValue_opt.has_value())
			{
				const std::string& disablePinnedMemoryValue =
				    disablePinnedMemoryValue_opt.value();

				if (disablePinnedMemoryValue == "1" ||
				    Util::toLower(disablePinnedMemoryValue) == "yes" ||
				    Util::toLower(disablePinnedMemoryValue) == "on" ||
				    Util::toLower(disablePinnedMemoryValue) == "true")
				{
					return false;
				}
			}
			return true;
		}();

		return s_usePinnedMemory;
	}

	inline void setPinnedMemoryEnabled(bool enabled)
	{
		usePinnedMemory() = enabled;
	}

	inline bool isPinnedMemoryEnabled()
	{
		return usePinnedMemory();
	}

	// TODO: Add an option to set the max VRAM

	static constexpr size_t ThreadsPerBlockData = 256;
	static constexpr size_t ThreadsPerBlockImg3d = 8;
	static constexpr size_t ThreadsPerBlockImg2d = 32;
};  // namespace GlobalsCuda
