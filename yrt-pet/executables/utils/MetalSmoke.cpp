/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/MetalSmoke.hpp"

#include <iostream>

int main()
{
	if (!yrt::backend::metal::isAvailable())
	{
		std::cerr << "Metal smoke kernel: FAIL (Metal device unavailable)\n";
		return 2;
	}

	if (!yrt::backend::metal::runSmokeKernel())
	{
		std::cerr << "Metal smoke kernel: FAIL\n";
		return 1;
	}

	std::cout << "Metal smoke kernel: PASS\n";
	return 0;
}
