/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Utilities.hpp"

#include "omp.h"

#include <array>
#include <cstddef>
#include <iostream>

namespace yrt
{
namespace globals
{

enum class VerbositySection
{
	GENERAL = 0,
	ALLOCATION,
	FILESYSTEM,
	IMAGE,
	PROJECTOR,
	CORRECTOR,
	OPTIMIZER,
	COUNT
};

static constexpr int DefaultVerbosityLevel = 1;

inline int& verbosityLevel(VerbositySection section)
{
	static constexpr int NumVerbositySections =
	    static_cast<int>(VerbositySection::COUNT);

	static std::array<int, NumVerbositySections> s_verbosityLevels = []
	{
		std::array<int, NumVerbositySections> arr{};
		arr.fill(DefaultVerbosityLevel);  // initialize all sections
		return arr;
	}();

	return s_verbosityLevels[static_cast<int>(section)];
}

inline int getVerbosityLevel(VerbositySection section)
{
	return verbosityLevel(section);
}

inline void setVerbosityLevel(VerbositySection section, int v)
{
	if (v <= 0)
	{
		verbosityLevel(section) = 0;  // clamp to zero
	}
	else
	{
		ASSERT_MSG(v <= 5, "Maximum verbosity level allowed is 5");
		verbosityLevel(section) = v;
	}
}

inline int& numThreads()
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
	return numThreads();
}

inline void setNumThreads(int t)
{
	if (t <= 0)
	{
		numThreads() = omp_get_max_threads();
	}
	else
	{
		numThreads() = t;
	}
	std::cout << "Using " << numThreads() << " threads." << std::endl;
	omp_set_num_threads(numThreads());
}

static constexpr char DisablePinnedMemoryEnvVar[] =
    "YRTPET_DISABLE_PINNED_MEMORY";

inline bool& usePinnedMemory()
{
	static bool s_usePinnedMemory = []()
	{
		const auto disablePinnedMemoryValue_opt =
		    util::getEnv(DisablePinnedMemoryEnvVar);
		if (disablePinnedMemoryValue_opt.has_value())
		{
			const std::string& disablePinnedMemoryValue =
			    disablePinnedMemoryValue_opt.value();

			if (disablePinnedMemoryValue == "1" ||
			    util::toLower(disablePinnedMemoryValue) == "yes" ||
			    util::toLower(disablePinnedMemoryValue) == "on" ||
			    util::toLower(disablePinnedMemoryValue) == "true")
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

// CUDA Constants
static constexpr size_t ThreadsPerBlockData = 256;
static constexpr size_t ThreadsPerBlockImg3d = 8;
static constexpr size_t ThreadsPerBlockImg2d = 32;

};  // namespace globals
}  // namespace yrt
