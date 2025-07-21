/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/scanner/DetRegular.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Array.hpp"


TEST_CASE("scanner", "[createLUT]")
{
	auto scanner = yrt::util::test::makeScanner();

	auto histo3d = std::make_unique<yrt::Histogram3DOwned>(*scanner);

	yrt::Array2D<float> lut;
	scanner->createLUT(lut);

	SECTION("lut-size")
	{
		REQUIRE(lut.getSize(0) == scanner->getNumDets());
		REQUIRE(lut.getSize(1) == 6);
	}

	SECTION("lut-det_pos")
	{
		size_t bin_id = 100;
		yrt::Vector3D pos = scanner->getDetectorPos(bin_id);
		REQUIRE(lut[bin_id][0] == pos.x);
		REQUIRE(lut[bin_id][1] == pos.y);
		REQUIRE(lut[bin_id][2] == pos.z);
	}
}
