/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"
#include <cmath>
#include <vector>

#include "yrt-pet/datastruct/projection/BinIterator.hpp"

#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "test_utils.hpp"

bool test_iter(yrt::BinIterator* iter, size_t begin, size_t second,
               size_t end_t, size_t numEl)
{
	if (iter->size() == 1)
	{
		return iter->begin() == begin && iter->get(0) == begin &&
		       iter->end() == begin && begin < numEl;
	}
	return iter->begin() == begin && iter->get(1) == second &&
	       iter->end() == end_t && end_t < numEl;
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
		auto iter = yrt::BinIteratorRange(begin, end, stride);
		REQUIRE(test_iter(&iter, begin, begin + stride, end_t, end + 1));
	}

	SECTION("range-singleton")
	{
		size_t begin = rand() % 100;
		size_t stride = 1 + rand() % 20;
		size_t end = begin + stride - 1;
		size_t end_t = begin;
		auto iter = yrt::BinIteratorRange(begin, end, stride);
		REQUIRE(test_iter(&iter, begin, begin + stride, end_t, end_t + 1));
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
		auto iter = yrt::BinIteratorRange(begin, end, stride);
		REQUIRE(test_iter(&iter, begin, begin + stride, end_t, end + 1));
	}
}

TEST_CASE("biniterator_vector", "[iterator]")
{
	SECTION("vector-basic")
	{
		std::vector<size_t> vec{1, 2, 3, 4, 5};
		auto vec_ptr = std::make_unique<std::vector<size_t>>(vec);
		auto iter = yrt::BinIteratorVector(vec_ptr);
		REQUIRE(test_iter(&iter, 1, 2, 5, 6));
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
		    yrt::BinIteratorChronological(numSubsets, numEvents, idxSubset);
		REQUIRE(test_iter(&iter, 0, 1, 3, numEvents));
		REQUIRE(iter.size() == 4);
	}
	SECTION("chronological-indxsubset = numsubset-1")
	{
		size_t idxSubset = numSubsets - 1;
		auto iter =
		    yrt::BinIteratorChronological(numSubsets, numEvents, idxSubset);
		REQUIRE(test_iter(&iter, 8, 9, 12, numEvents));
		REQUIRE(iter.size() == 5);
	}
	SECTION("chronological-numsubset % idxSubset = 0")
	{
		size_t numSubsets = 3;
		size_t numEvents = 12;
		size_t idxSubset = numSubsets - 1;
		auto iter =
		    yrt::BinIteratorChronological(numSubsets, numEvents, idxSubset);
		REQUIRE(test_iter(&iter, 8, 9, 11, numEvents));
		REQUIRE(iter.size() == 4);
	}
}

TEST_CASE("biniterator_chronological_interleaved", "[iterator]")
{
	std::vector<size_t> endListRef{12, 10, 11};
	SECTION("chronological-interleaved")
	{
		constexpr size_t NumSubsets = 3;
		constexpr size_t NumEvents = 13;

		for (size_t idxSubset = 0; idxSubset < NumSubsets; idxSubset++)
		{
			auto iter = yrt::BinIteratorChronologicalInterleaved(
			    NumSubsets, NumEvents, idxSubset);
			REQUIRE(test_iter(&iter, idxSubset, idxSubset + NumSubsets,
			                  endListRef.at(idxSubset), NumEvents));
		}
	}

	SECTION("listmode-sum-of-events")
	{
		constexpr int NumSubsets = 9;
		constexpr size_t NumEvents = 15000;

		const auto scanner = yrt::util::test::makeFakeScanner();

		auto lm = yrt::ListModeLUTOwned(*scanner);
		lm.allocate(NumEvents);

		std::vector<std::unique_ptr<yrt::BinIterator>> binIterators;

		size_t totalSize = 0;

		for (int idxSubset = 0; idxSubset < NumSubsets; idxSubset++)
		{
			binIterators.push_back(lm.getBinIter(NumSubsets, idxSubset));

			// Ensure ListModes return a BinIteratorChronologicalInterleaved
			REQUIRE(typeid(*binIterators[binIterators.size() - 1]) ==
			        typeid(yrt::BinIteratorChronologicalInterleaved));

			totalSize += binIterators[binIterators.size() - 1]->size();
		}

		REQUIRE(totalSize == NumEvents);
	}
}

// TODO: Add a unit test for BinIteratorRange3D
