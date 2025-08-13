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

using namespace yrt;

static void fillRandomListMode(
    ListModeLUTOwned& lm, size_t numEvents, bool withTOF, bool withRandoms,
    const std::shared_ptr<std::default_random_engine>& pp_engine)
{
	std::shared_ptr<std::default_random_engine> engine = pp_engine;
	if (engine == nullptr)
	{
		engine = std::make_shared<std::default_random_engine>();
	}

	std::uniform_int_distribution<timestamp_t> ts_dist(0, 1'000'000);
	std::uniform_int_distribution<det_id_t> det_dist(
	    0, lm.getScanner().getNumDets() - 1);
	std::uniform_real_distribution<float> tof_dist(-5000.0f, 5000.0f);
	std::uniform_real_distribution<float> rand_dist(0.0f, 1000.0f);

	for (size_t i = 0; i < numEvents; i++)
	{
		lm.setTimestampOfEvent(i, ts_dist(*engine));
		const det_id_t d1 = det_dist(*engine);
		const det_id_t d2 = det_dist(*engine);
		lm.setDetectorIdsOfEvent(i, d1, d2);
		if (withTOF)
		{
			lm.setTOFValueOfEvent(i, tof_dist(*engine));
		}
		if (withRandoms)
		{
			lm.setRandomsEstimateOfEvent(i, rand_dist(*engine));
		}
	}
}

static void compareListModes(const ListModeLUT& a, const ListModeLUT& b,
                             bool withTOF, bool withRandoms)
{
	REQUIRE(a.count() == b.count());
	for (size_t i = 0; i < a.count(); i++)
	{
		CHECK(a.getTimestamp(i) == b.getTimestamp(i));
		CHECK(a.getDetector1(i) == b.getDetector1(i));
		CHECK(a.getDetector2(i) == b.getDetector2(i));
		if (withTOF)
		{
			CHECK(a.getTOFValue(i) == Approx(b.getTOFValue(i)));
		}
		if (withRandoms)
		{
			CHECK(a.getRandomsEstimate(i) == Approx(b.getRandomsEstimate(i)));
		}
	}
}

TEST_CASE("listmodelut", "[list-mode]")
{
	const auto scanner = util::test::makeScanner();
	const auto seed = static_cast<unsigned int>(std::time(nullptr));
	const auto engine = std::make_shared<std::default_random_engine>(seed);

	constexpr size_t maxNumEvents = 100'000;
	const size_t numEvents =
	    std::uniform_int_distribution<size_t>(1, maxNumEvents)(*engine);

	auto testCase = [&](bool withTOF, bool withRandoms)
	{
		// Create & fill list mode
		ListModeLUTOwned lmOwned(*scanner, withTOF, withRandoms);
		lmOwned.allocate(numEvents);
		fillRandomListMode(lmOwned, numEvents, withTOF, withRandoms, engine);

		// Test alias binding
		ListModeLUTAlias lmAlias(*scanner, withTOF, withRandoms);
		lmAlias.bind(&lmOwned);
		compareListModes(lmOwned, lmAlias, withTOF, withRandoms);

		// Test array getters (pointers should match)
		CHECK(lmOwned.getTimestampArrayPtr()->getSize(0) == numEvents);
		CHECK(lmOwned.getDetector1ArrayPtr()->getSize(0) == numEvents);
		CHECK(lmOwned.getDetector2ArrayPtr()->getSize(0) == numEvents);
		if (withTOF)
		{
			CHECK(lmOwned.getTOFArrayPtr()->getSize(0) == numEvents);
		}
		if (withRandoms)
		{
			CHECK(lmOwned.getRandomsEstimatesArrayPtr()->getSize(0) ==
			      numEvents);
		}

		// Test write + read
		const std::string fname = "tmp_listmodelut.lmDat";
		lmOwned.writeToFile(fname);
		ListModeLUTOwned lmRead(*scanner, withTOF, withRandoms);
		lmRead.readFromFile(fname);
		compareListModes(lmOwned, lmRead, withTOF, withRandoms);
		std::remove(fname.c_str());

		// Test histogram bin
		for (bin_t eventId = 0; eventId < numEvents; eventId++)
		{
			histo_bin_t histoBin = lmOwned.getHistogramBin(eventId);
			auto detPair = std::get<det_pair_t>(histoBin);
			CHECK(lmOwned.getDetector1(eventId) == detPair.d1);
			CHECK(lmOwned.getDetector2(eventId) == detPair.d2);
		}
	};

	SECTION("no-tof-no-randoms")
	{
		testCase(false, false);
	}
	SECTION("with-tof-no-randoms")
	{
		testCase(true, false);
	}
	SECTION("no-tof-with-randoms")
	{
		testCase(false, true);
	}
	SECTION("with-tof-with-randoms")
	{
		testCase(true, true);
	}
}
