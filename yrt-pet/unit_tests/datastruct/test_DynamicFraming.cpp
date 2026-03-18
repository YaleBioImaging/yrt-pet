/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include <chrono>  // for timestamp in filename
#include <cstdio>  // std::remove
#include <fstream>
#include <random>  // for unique filename generation

#include "yrt-pet/datastruct/projection/DynamicFraming.hpp"

using namespace yrt;


TEST_CASE("dynamicframing-numframes-constructor", "[DynamicFraming]")
{
	SECTION("Valid number of frames")
	{
		size_t numFrames = 5;
		DynamicFraming df(numFrames);

		CHECK(df.getNumFrames() == numFrames);
		CHECK(df.getNumTimestamps() == numFrames + 1);

		// All timestamps should be zero-initialized
		for (size_t i = 0; i < df.getNumTimestamps(); ++i)
		{
			CHECK(df.getStartingTimestamp(i) == 0);
		}
	}

	SECTION("Invalid number of frames (0) throws")
	{
		REQUIRE_THROWS_AS(DynamicFraming(0), std::runtime_error);
	}

	SECTION("Invalid number of frames (1) throws")
	{
		REQUIRE_THROWS_AS(DynamicFraming(1), std::runtime_error);
	}
}

TEST_CASE("dynamicframing-vector-constructor", "[DynamicFraming]")
{
	SECTION("Valid vector")
	{
		std::vector<timestamp_t> timestamps = {100, 200, 350, 500};
		DynamicFraming df(timestamps);

		CHECK(df.getNumFrames() == timestamps.size() - 1);
		CHECK(df.getNumTimestamps() == timestamps.size());

		for (size_t i = 0; i < timestamps.size(); ++i)
		{
			CHECK(df.getStartingTimestamp(i) == timestamps[i]);
		}
	}

	SECTION("Empty vector throws")
	{
		std::vector<timestamp_t> empty;
		REQUIRE_THROWS_AS(DynamicFraming(empty), std::runtime_error);
	}

	SECTION("Single‑element vector throws")
	{
		std::vector<timestamp_t> one = {42};
		REQUIRE_THROWS_AS(DynamicFraming(one), std::runtime_error);
	}
}

TEST_CASE("dynamicframing-io", "[DynamicFraming]")
{
	std::string filename = "test.dyn";

	// Use a fixed seed for reproducible tests
	std::mt19937 rng(13);
	std::uniform_int_distribution<size_t> numFramesDist(2, 20);
	std::uniform_int_distribution<timestamp_t> durationDist(1, 100);

	const size_t numFrames = numFramesDist(rng);
	std::vector<timestamp_t> timestamps;
	timestamps.reserve(numFrames + 1);

	// Generate strictly increasing timestamps
	timestamp_t current = 0;
	timestamps.push_back(current);
	for (size_t i = 0; i < numFrames; ++i) {
		current += durationDist(rng);
		timestamps.push_back(current);
	}

	DynamicFraming original(timestamps);
	REQUIRE(original.isValid());

	// Write to file
	original.writeToFile(filename);

	// Read back
	DynamicFraming fromFile(filename);

	// Compare
	REQUIRE(fromFile.getNumTimestamps() == original.getNumTimestamps());
	for (size_t i = 0; i < original.getNumTimestamps(); ++i) {
		CHECK(fromFile.getStartingTimestamp(i) == original.getStartingTimestamp(i));
	}

	// Also verify raw file content
	{
		std::ifstream infile(filename);
		std::vector<timestamp_t> readBack;
		timestamp_t t;
		while (infile >> t) {
			readBack.push_back(t);
		}
		CHECK(readBack == timestamps);
	}

	std::remove(filename.c_str());
}

TEST_CASE("dynamicframing-methods", "[DynamicFraming]")
{
	std::vector<timestamp_t> timestamps = {10, 20, 35, 50, 70};
	DynamicFraming df(timestamps);
	REQUIRE(df.isValid());

	SECTION("getStartingTimestamp and getStoppingTimestamp")
	{
		CHECK(df.getStartingTimestamp(0) == 10);
		CHECK(df.getStoppingTimestamp(0) == 20);

		CHECK(df.getStartingTimestamp(2) == 35);
		CHECK(df.getStoppingTimestamp(2) == 50);

		// Last frame
		size_t lastFrame = df.getNumFrames() - 1;
		CHECK(df.getStartingTimestamp(lastFrame) == 50);
		CHECK(df.getStoppingTimestamp(lastFrame) == 70);
	}

	SECTION("getDuration")
	{
		CHECK(df.getDuration(0) == 10.0f);  // 20-10
		CHECK(df.getDuration(1) == 15.0f);  // 35-20
		CHECK(df.getDuration(2) == 15.0f);  // 50-35
		CHECK(df.getDuration(3) == 20.0f);  // 70-50

		// Out-of-range frame index throws
		REQUIRE_THROWS_AS(df.getDuration(4), std::runtime_error);
		REQUIRE_THROWS_AS(df.getDuration(100), std::runtime_error);
	}

	SECTION("getTotalDuration")
	{
		CHECK(df.getTotalDuration() == 60.0f);  // 70-10
	}

	SECTION("setStartingTimestamp and setLastTimestamp")
	{
		df.setStartingTimestamp(1, 25);
		CHECK(df.getStartingTimestamp(1) == 25);
		// The stop timestamp of frame 1 should now be 35 (unchanged)
		CHECK(df.getStoppingTimestamp(1) == 35);

		df.setLastTimestamp(80);
		CHECK(df.getStoppingTimestamp(df.getNumFrames() - 1) == 80);
		CHECK(df.getTotalDuration() == 70.0f);  // 80-10

		// Verify internal consistency
		CHECK(df.getNumTimestamps() == 5);
	}

	SECTION("isValid")
	{
		// Valid increasing sequence
		CHECK(df.isValid() == true);

		// Modify to create a non‑increasing sequence
		df.setStartingTimestamp(
		    2, 30);  // becomes 30, but previous was 35? Actually careful:
		// setStartingTimestamp(2,30) changes timestamp at index 2 (frame 2
		// start) to 30, while the next timestamp (index 3) is 50. That's still
		// increasing (30 < 50). To break it, make a timestamp not greater than
		// previous.
		df.setStartingTimestamp(2, 20);  // now 20, previous timestamp (index 1)
		                                 // is 20 -> not strictly greater
		CHECK(df.isValid() == false);

		// Reset to valid
		df.setStartingTimestamp(2, 35);
		CHECK(df.isValid() == true);

		// Make last timestamp equal to previous
		df.setLastTimestamp(50);
		CHECK(df.isValid() == false);  // 50 not > 50

		// Reset to valid again
		df.setLastTimestamp(70);
		CHECK(df.isValid() == true);
	}
}

TEST_CASE("dynamicframing-edge-cases", "[DynamicFraming]")
{
	SECTION("Two frames (minimum valid)")
	{
		DynamicFraming df(2);  // 2 frames -> 3 timestamps, all zero
		CHECK(df.getNumFrames() == 2);
		CHECK(df.getNumTimestamps() == 3);
		CHECK(df.getDuration(0) == 0.0f);
		CHECK(df.getDuration(1) == 0.0f);
		CHECK(df.getTotalDuration() == 0.0f);
		CHECK(df.isValid() ==
		      false);  // because timestamps are not strictly increasing (0,0,0)

		// Set increasing timestamps to make it valid
		df.setStartingTimestamp(0, 10);
		df.setStartingTimestamp(1, 20);
		df.setLastTimestamp(30);
		CHECK(df.isValid() == true);
		CHECK(df.getDuration(0) == 10.0f);
		CHECK(df.getDuration(1) == 10.0f);
		CHECK(df.getTotalDuration() == 20.0f);
	}

	SECTION("Large number of frames")
	{
		constexpr size_t big = 1000;
		DynamicFraming df(big);
		CHECK(df.getNumFrames() == big);
		CHECK(df.getNumTimestamps() == big + 1);
	}
}
