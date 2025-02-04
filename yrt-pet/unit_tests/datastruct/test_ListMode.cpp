/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../../../build-cuda-pybind11/external/Catch/single_include/catch2/catch.hpp"

#include "../../include/datastruct/projection/ListModeLUT.hpp"
#include "../../include/utils/Array.hpp"
#include "../test_utils.hpp"

#include <cmath>
#include <cstdio>

std::unique_ptr<ListModeLUTOwned> getListMode(const Scanner& scanner)
{
	auto listMode = std::make_unique<ListModeLUTOwned>(scanner);
	listMode->allocate(15);
	listMode->setDetectorIdsOfEvent(0, 25, 36);
	listMode->setDetectorIdsOfEvent(13, 1, 60);
	return listMode;
}

TEST_CASE("listmode", "[list-mode]")
{
	const auto scanner = TestUtils::makeScanner();
	auto listMode = getListMode(*scanner);

	SECTION("listmode-data")
	{
		CHECK(listMode->getDetector1(0) == 25);
		CHECK(listMode->getDetector2(0) == 36);
		CHECK(listMode->getDetector1(13) == 1);
		CHECK(listMode->getDetector2(13) == 60);
	}

	SECTION("listmode-binding")
	{
		ListModeLUTAlias listMode_alias(*scanner);
		listMode_alias.bind(listMode.get());
		CHECK(listMode->getTimestamp(0) == listMode_alias.getTimestamp(0));
		CHECK(listMode->getTimestamp(13) == listMode_alias.getTimestamp(13));
		CHECK(listMode->getDetector1(0) == listMode_alias.getDetector1(0));
		CHECK(listMode->getDetector2(0) == listMode_alias.getDetector2(0));
		CHECK(listMode->getDetector1(13) == listMode_alias.getDetector1(13));
		CHECK(listMode->getDetector2(13) == listMode_alias.getDetector2(13));
	}

	SECTION("listmode-fileread")
	{
		listMode->writeToFile("listmode1");

		auto listMode2 = std::make_unique<ListModeLUTOwned>(*scanner);
		listMode2->readFromFile("listmode1");

		CHECK(listMode->getTimestamp(0) == listMode2->getTimestamp(0));
		CHECK(listMode->getTimestamp(13) == listMode2->getTimestamp(13));
		CHECK(listMode->getDetector1(0) == listMode2->getDetector1(0));
		CHECK(listMode->getDetector2(0) == listMode2->getDetector2(0));
		CHECK(listMode->getDetector1(13) == listMode2->getDetector1(13));
		CHECK(listMode->getDetector2(13) == listMode2->getDetector2(13));

		std::remove("listmode1");
	}

	SECTION("listmode-get-lor-id")
	{
		histo_bin_t histoBin = listMode->getHistogramBin(0);
		auto detPair = std::get<det_pair_t>(histoBin);
		CHECK(listMode->getDetector1(0) == detPair.d1);
		CHECK(listMode->getDetector2(0) == detPair.d2);
	}
}

void makeArraysDOI(Array1D<float>& ts, Array1D<float>& tof,
                   Array1D<det_id_t>& d1, Array1D<det_id_t>& d2,
                   Array1D<unsigned char>& z1, Array1D<unsigned char>& z2,
                   int numDOI, det_id_t d1_i, det_id_t d2_i)
{
	ts.allocate(numDOI);
	tof.allocate(numDOI);
	d1.allocate(numDOI);
	d2.allocate(numDOI);
	z1.allocate(numDOI);
	z2.allocate(numDOI);
	for (int li = 0; li < numDOI; li++)
	{
		ts[li] = 15.2f + (float)li;
		tof[li] = (rand() / (float)RAND_MAX) * 100.f - 50.f;
		d1[li] = d1_i;
		d2[li] = d2_i;
		z1[li] = li;
		z2[li] = li;
	}
}
