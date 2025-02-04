/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../../include/geometry/Line3D.hpp"
#include "../../include/operators/TimeOfFlight.hpp"

#include "../../../build-cuda-pybind11/external/Catch/single_include/catch2/catch.hpp"
#include <cmath>
#include <iostream>

TEST_CASE("TOF", "[tof]")
{
	int random_seed = time(0);
	srand(random_seed);
	std::string rseed_str = "random_seed=" + std::to_string(random_seed);
	INFO(rseed_str);

	SECTION("width_0")
	{
		auto tofHelper = TimeOfFlightHelper(0.f, 3);
		Line3D lor{Vector3D{rand() / (float)RAND_MAX, rand() / (float)RAND_MAX,
		                    rand() / (float)RAND_MAX},
		           Vector3D{rand() / (float)RAND_MAX, rand() / (float)RAND_MAX,
		                    rand() / (float)RAND_MAX}};
		double d_norm = (lor.point2 - lor.point1).getNorm();
		float tof_value_ps = (rand() / (float)RAND_MAX) * 0.5f - 0.5f;
		float amin, amax;
		tofHelper.getAlphaRange(amin, amax, d_norm, tof_value_ps);
		REQUIRE(amin == 0.);
		REQUIRE(amax == 1.);
	}

	SECTION("simple_geom")
	{
		float tof_width_ps = 95.f;
		auto tofHelper = TimeOfFlightHelper(tof_width_ps, 3);
		Line3D lor{Vector3D{-155.0, 118.75, 367.5},
		           Vector3D{155.0, 163.75, 373.5}};
		float d_norm = (lor.point2 - lor.point1).getNorm();
		float tof_value_ps = -408.f;
		float pix_pos_lo = 0.24f * d_norm;
		float pix_pos_hi = 0.25f * d_norm;
		float tof_weight =
		    tofHelper.getWeight(d_norm, tof_value_ps, pix_pos_lo, pix_pos_hi);
		REQUIRE(std::abs(tof_weight - 0.000543244) < 1e-4);
	}
}
