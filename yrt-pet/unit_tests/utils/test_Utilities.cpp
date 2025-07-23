/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"
#include <vector>

#include "yrt-pet/utils/RangeList.hpp"
#include "yrt-pet/utils/Utilities.hpp"

TEST_CASE("String", "[string]")
{
	SECTION("whitespace_test")
	{
		REQUIRE(yrt::util::beginWithNonWhitespace("string"));
		REQUIRE_FALSE(yrt::util::beginWithNonWhitespace(" string"));
	}
	SECTION("strip")
	{
		REQUIRE(yrt::util::stripWhitespaces(" ab cd  ") == "ab cd");
	}
	SECTION("case")
	{
		REQUIRE(yrt::util::toLower("AB Cd") == "ab cd");
		REQUIRE(yrt::util::toUpper("aB Cd") == "AB CD");
	}
	SECTION("split")
	{
		REQUIRE(yrt::util::split("ab/cd", "/") ==
		        std::vector<std::string>{"ab", "cd"});
		REQUIRE(yrt::util::split("ab", "/") == std::vector<std::string>{"ab"});
	}
	SECTION("ranges-insert")
	{
		yrt::util::RangeList ranges;
		REQUIRE(ranges.empty());
		// Test 1: Inserting into an empty vector
		ranges.readFromString("3-4");
		REQUIRE_FALSE(ranges.empty());
		REQUIRE(ranges.getSizeTotal() == 2);
		REQUIRE(ranges.get().size() == 1);
		REQUIRE(ranges.isIn(3));
		REQUIRE(ranges.isIn(4));
		REQUIRE_FALSE(ranges.isIn(5));

		ranges.insertSorted(8, 10);
		REQUIRE(ranges.get().size() == 2);
		REQUIRE(ranges.getSizeTotal() == 5);

		ranges.insertSorted(6, 7);
		REQUIRE(ranges.get().size() == 2);
		REQUIRE(ranges.getSizeTotal() == 7);

		ranges.insertSorted(6, 6);
		REQUIRE(ranges.get().size() == 2);
		REQUIRE(ranges.getSizeTotal() == 7);

		ranges.insertSorted(4, 9);
		REQUIRE(ranges.get().size() == 1);
		REQUIRE(ranges.getSizeTotal() == 8);
	}
	SECTION("ranges-parse")
	{
		yrt::util::RangeList ranges("1-3, 5-10, 15-15, 17-20");
		REQUIRE(ranges.get().size() == 4);
		REQUIRE(ranges.getSizeTotal() == 14);
	}
	SECTION("ranges-parse-red")
	{
		yrt::util::RangeList ranges("1-3, 5-16, 15-15, 17-20");
		REQUIRE(ranges.get().size() == 2);
		REQUIRE(ranges.getSizeTotal() == 19);
	}
	SECTION("ranges-step")
	{
		auto ranges = yrt::util::RangeList::makeRangeListStep(1, 9, 2);
		REQUIRE(ranges.getSizeTotal() == 5);
		REQUIRE(ranges.isIn(1));
		REQUIRE(!ranges.isIn(2));
		REQUIRE(ranges.get().size() == 5);
	}
}
