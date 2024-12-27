/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"
#include <cmath>
#include <vector>

#include "datastruct/projection/BinIterator.hpp"

bool test_iter(BinIterator* iter, size_t begin, size_t second, size_t end_t)
{
	if (iter->size() == 1)
	{
		return iter->begin() == begin && iter->get(0) == begin &&
		       iter->end() == begin;
	}
	return iter->begin() == begin && iter->get(1) == second &&
	       iter->end() == end_t;
}

TEST_CASE("biniterator_range", "[iterator]")
{
	SECTION("range-basic")
	{
		size_t begin = rand() % 100;
		size_t stride = 1 + rand() % 20;
		// Ensure at least two elements
		size_t end = begin + stride + (rand() % 20);
		size_t end_t = begin;
		while (end_t + stride <= end)
		{
			end_t += stride;
		}
		auto iter = BinIteratorRange(begin, end, stride);
		REQUIRE(test_iter(&iter, begin, begin + stride, end_t));
	}

	SECTION("range-singleton")
	{
		size_t begin = rand() % 100;
		size_t stride = 1 + rand() % 20;
		size_t end = begin + stride - 1;
		size_t end_t = begin;
		auto iter = BinIteratorRange(begin, end, stride);
		REQUIRE(test_iter(&iter, begin, begin + stride, end_t));
		REQUIRE(iter.size() == 1);
	}

	SECTION("range-large")
	{
		size_t begin = ((size_t)1 << 32) + rand() % 100;
		size_t stride = 1 + rand() % 20;
		size_t end = begin + (rand() % 20);
		size_t end_t = begin;
		while (end_t + stride <= end)
		{
			end_t += stride;
		}
		auto iter = BinIteratorRange(begin, end, stride);
		REQUIRE(test_iter(&iter, begin, begin + stride, end_t));
	}
}

TEST_CASE("biniterator_vector", "[iterator]")
{
	SECTION("vector-basic")
	{
		std::vector<size_t> vec{1, 2, 3, 4, 5};
		auto vec_ptr = std::make_unique<std::vector<size_t>>(vec);
		auto iter = BinIteratorVector(vec_ptr);
		REQUIRE(test_iter(&iter, 1, 2, 5));
		REQUIRE(iter.size() == 5);
	}
}

TEST_CASE("biniterator_chronological", "[iterator]")
{
	size_t numSubsets = 3;
	size_t numEvents = 13;
	SECTION("chronological-indxsubset = 0")
	{
		size_t idxSubset = 0;
		auto iter =
		    BinIteratorChronological(numSubsets, numEvents, idxSubset);
		REQUIRE(test_iter(&iter, 0, 1, 3));
		REQUIRE(iter.size() == 4);
	}
	SECTION("chronological-indxsubset = numsubset-1")
	{
		size_t idxSubset = numSubsets - 1;
		auto iter =
		    BinIteratorChronological(numSubsets, numEvents, idxSubset);
		REQUIRE(test_iter(&iter, 8, 9, 12));
		REQUIRE(iter.size() == 5);
	}
	SECTION("chronological-numsubset % idxSubset = 0")
	{
		size_t numSubsets = 3;
		size_t numEvents = 12;
		size_t idxSubset = numSubsets - 1;
		auto iter =
		    BinIteratorChronological(numSubsets, numEvents, idxSubset);
		REQUIRE(test_iter(&iter, 8, 9, 11));
		REQUIRE(iter.size() == 4);
	}
}

// TODO: Add a unit test for BinIteratorRange3D
