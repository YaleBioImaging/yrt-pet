/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../../../build-cuda-pybind11/external/Catch/single_include/catch2/catch.hpp"

#include "../../include/datastruct/projection/Histogram3D.hpp"
#include "../../include/datastruct/scanner/DetRegular.hpp"
#include "../../include/datastruct/scanner/Scanner.hpp"
#include "../../include/utils/Array.hpp"
#include "../test_utils.hpp"


TEST_CASE("scanner", "[createLUT]")
{
	auto scanner = TestUtils::makeScanner();

	auto histo3d = std::make_unique<Histogram3DOwned>(*scanner);

	Array2D<float> lut;
	scanner->createLUT(lut);

	SECTION("lut-size")
	{
		REQUIRE(lut.getSize(0) == scanner->getNumDets());
		REQUIRE(lut.getSize(1) == 6);
	}

	SECTION("lut-det_pos")
	{
		size_t bin_id = 100;
		Vector3D pos = scanner->getDetectorPos(bin_id);
		REQUIRE(lut[bin_id][0] == pos.x);
		REQUIRE(lut[bin_id][1] == pos.y);
		REQUIRE(lut[bin_id][2] == pos.z);
	}
}
