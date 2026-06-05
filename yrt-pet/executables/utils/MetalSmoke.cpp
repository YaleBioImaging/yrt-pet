/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/MetalSmoke.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

bool envFlag(const char* name)
{
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

}  // namespace

int main(int argc, char** argv)
{
	bool runJosephAdjointAccumulationSmoke = envFlag(
	    "YRTPET_METAL_SMOKE_JOSEPH_ADJOINT_ACCUMULATION");
	bool runJosephAxisSpecializedSmoke = envFlag(
	    "YRTPET_METAL_SMOKE_JOSEPH_AXIS_SPECIALIZED");
	for (int i = 1; i < argc; ++i)
	{
		if (std::string(argv[i]) ==
		    "--joseph-adjoint-accumulation-smoke")
		{
			runJosephAdjointAccumulationSmoke = true;
		}
		else if (std::string(argv[i]) ==
		         "--joseph-axis-specialized-smoke")
		{
			runJosephAxisSpecializedSmoke = true;
		}
	}

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

	if (runJosephAdjointAccumulationSmoke)
	{
		if (!yrt::backend::metal::runJosephAdjointAccumulationSmoke())
		{
			std::cerr
			    << "Metal Joseph adjoint accumulation smoke: FAIL\n";
			return 1;
		}
		std::cout
		    << "Metal Joseph adjoint accumulation smoke: PASS\n";
	}
	if (runJosephAxisSpecializedSmoke)
	{
		if (!yrt::backend::metal::runJosephAxisSpecializedSmoke())
		{
			std::cerr
			    << "Metal Joseph axis-specialized smoke: FAIL\n";
			return 1;
		}
		std::cout
		    << "Metal Joseph axis-specialized smoke: PASS\n";
	}
	return 0;
}
