/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/utils/Array.hpp"

#include <cmath>
#include <cstdio>

namespace yrt
{
std::unique_ptr<ListModeLUTOwned> getListMode(const Scanner& scanner)
{
	auto listMode = std::make_unique<ListModeLUTOwned>(scanner);
	listMode->allocate(15);
	listMode->setDetectorIdsOfEvent(0, 25, 36);
	listMode->setDetectorIdsOfEvent(13, 1, 60);
	return listMode;
}
}  // namespace yrt

TEST_CASE("listmode", "[list-mode]")
{
	const auto scanner = yrt::util::test::makeScanner();
	auto listMode = yrt::getListMode(*scanner);

	SECTION("listmode-data")
	{
		CHECK(listMode->getDetector1(0) == 25);
		CHECK(listMode->getDetector2(0) == 36);
		CHECK(listMode->getDetector1(13) == 1);
		CHECK(listMode->getDetector2(13) == 60);
	}

	SECTION("listmode-binding")
	{
		yrt::ListModeLUTAlias listMode_alias(*scanner);
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

		auto listMode2 = std::make_unique<yrt::ListModeLUTOwned>(*scanner);
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
		yrt::histo_bin_t histoBin = listMode->getHistogramBin(0);
		auto detPair = std::get<yrt::det_pair_t>(histoBin);
		CHECK(listMode->getDetector1(0) == detPair.d1);
		CHECK(listMode->getDetector2(0) == detPair.d2);
	}
}

void makeArraysDOI(yrt::Array1D<float>& ts, yrt::Array1D<float>& tof,
                   yrt::Array1D<yrt::det_id_t>& d1,
                   yrt::Array1D<yrt::det_id_t>& d2,
                   yrt::Array1D<unsigned char>& z1,
                   yrt::Array1D<unsigned char>& z2, int numDOI,
                   yrt::det_id_t d1_i, yrt::det_id_t d2_i)
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
