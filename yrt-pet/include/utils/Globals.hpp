/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "omp.h"
#include <cstddef>
#include <iostream>

class Globals
{
public:
	static int get_num_threads() { return num_threads; }
	static void set_num_threads(int t)
	{
		if (t <= 0)
		{
			num_threads = omp_get_max_threads();
		}
		else
		{
			num_threads = t;
		}
		std::cout << "Using " << num_threads << " threads." << std::endl;
		omp_set_num_threads(num_threads);
	}

private:
	static int num_threads;
};

class GlobalsCuda
{
public:
	static constexpr size_t MaxVRAMAllowed = 2ull << 30;  // bytes
	static constexpr size_t ThreadsPerBlockData = 256;
	static constexpr size_t ThreadsPerBlockImg3d = 8;
	static constexpr size_t ThreadsPerBlockImg2d = 32;
};
