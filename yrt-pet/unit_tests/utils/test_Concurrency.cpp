/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "yrt-pet/utils/Concurrency.hpp"

#include <atomic>
#include <numeric>
#include <vector>

TEST_CASE("parallelForChunkedRandomized", "[concurrency]")
{
	constexpr size_t total = 1 << 20;  // Around 1 million
	constexpr size_t numThreads = 4;
	constexpr float probability = 0.05f;

	SECTION("probability-1-all")
	{
		std::vector<size_t> threadCounts(numThreads, 0);
		yrt::util::parallelForChunkedRandomized(
		    total, numThreads, 1.0f,
		    [&threadCounts](size_t, size_t, unsigned int tid)
		    { threadCounts[tid]++; });
		const size_t sum = std::accumulate(threadCounts.begin(),
		                                   threadCounts.end(), size_t{0});
		REQUIRE(sum == total);
	}

	SECTION("indices-in-bounds")
	{
		std::atomic<bool> outOfRange{false};
		yrt::util::parallelForChunkedRandomized(
		    total, numThreads, 0.5f,
		    [&outOfRange](size_t idx, size_t, unsigned int)
		    {
			    if (idx >= total)
				    outOfRange = true;
		    });
		REQUIRE_FALSE(outOfRange);
	}

	SECTION("no-duplicate-indices")
	{
		// Chunks are disjoint, so each index is owned by exactly one thread.
		// Within a chunk the algorithm only moves forward, so every visited
		//  index should be unique. Verify that the number of calls matches
		//  the number of distinct indices visited.
		std::vector<uint8_t> visited(total, 0);
		std::atomic<size_t> callCount{0};

		yrt::util::parallelForChunkedRandomized(
		    total, numThreads, 0.5f,
		    [&visited, &callCount](size_t idx, size_t, unsigned int)
		    {
			    visited[idx] = 1;  // Set to one
			    ++callCount;       // Add one
		    });

		const size_t visitedCount =
		    std::accumulate(visited.begin(), visited.end(), size_t{0});
		REQUIRE(visitedCount == callCount);
	}

	SECTION("all-threads-participate")
	{
		std::vector<std::atomic<bool>> threadUsed(numThreads);
		for (auto& t : threadUsed)
		{
			t = false;
		}

		yrt::util::parallelForChunkedRandomized(
		    total, numThreads, 0.5f,
		    [&threadUsed](size_t, size_t, unsigned int tid)
		    { threadUsed[tid].store(true, std::memory_order_relaxed); });

		for (size_t i = 0; i < numThreads; i++)
		{
			REQUIRE(threadUsed[i]);
		}
	}

	SECTION("probability-count-scale")
	{
		std::vector<size_t> threadCounts(numThreads, 0);

		yrt::util::parallelForChunked(total, numThreads,
		                              [&threadCounts](size_t, unsigned int tid)
		                              { threadCounts[tid]++; });
		const size_t fullSum = std::accumulate(threadCounts.begin(),
		                                       threadCounts.end(), size_t{0});
		REQUIRE(fullSum == total);

		std::fill(threadCounts.begin(), threadCounts.end(), size_t{0});
		yrt::util::parallelForChunkedRandomized(
		    total, numThreads, probability,
		    [&threadCounts](size_t, size_t, unsigned int tid)
		    { threadCounts[tid]++; });
		const size_t randomSum = std::accumulate(threadCounts.begin(),
		                                         threadCounts.end(), size_t{0});

		constexpr float expected = static_cast<float>(total) * probability;
		REQUIRE(static_cast<float>(randomSum) ==
		        Approx(expected).epsilon(0.1f));
	}

	SECTION("statistical-over-many-trials")
	{
		// Running many trials and averaging reduces variance,
		//  giving a much tighter check than the previous single-shot test.
		constexpr size_t trials = 50;
		std::vector<size_t> counts(trials);

		for (size_t t = 0; t < trials; ++t)
		{
			std::vector<size_t> threadCounts(numThreads, 0);
			yrt::util::parallelForChunkedRandomized(
			    total, numThreads, probability,
			    [&threadCounts](size_t, size_t, unsigned int tid)
			    { threadCounts[tid]++; });
			counts[t] = std::accumulate(threadCounts.begin(),
			                            threadCounts.end(), size_t{0});
		}

		const float mean = std::accumulate(counts.begin(), counts.end(), 0.0f) /
		                   static_cast<float>(trials);
		constexpr float expected = static_cast<float>(total) * probability;

		REQUIRE(mean == Approx(expected).epsilon(0.02f));
	}

	SECTION("counter-is-contiguous")
	{
		// Ensure that the "counter" we get in the lambda function is always
		//  increased by exactly one
		std::vector<std::vector<size_t>> counters(numThreads);
		yrt::util::parallelForChunkedRandomized(
		    total, numThreads, probability,
		    [&counters](size_t, size_t counter, unsigned int tid)
		    { counters[tid].push_back(counter); });

		for (size_t t = 0; t < numThreads; ++t)
		{
			REQUIRE_FALSE(counters[t].empty());
			for (size_t i = 1; i < counters[t].size(); ++i)
			{
				REQUIRE(counters[t][i] == counters[t][i - 1] + 1);
			}
		}
	}

	SECTION("edge-total-zero")
	{
		// If total is zero, the loop should do nothing
		std::atomic<size_t> count{0};
		yrt::util::parallelForChunkedRandomized(
		    0, numThreads, 0.5f,
		    [&count](size_t, size_t, unsigned int) { ++count; });
		REQUIRE(count == 0);
	}

	SECTION("edge-total-less-than-threads")
	{
		constexpr size_t smallTotal = 3;
		std::vector<size_t> threadCounts(numThreads, 0);
		yrt::util::parallelForChunkedRandomized(
		    smallTotal, numThreads, 1.0f,
		    [&threadCounts](size_t, size_t, unsigned int tid)
		    { threadCounts[tid]++; });

		const size_t sum = std::accumulate(threadCounts.begin(),
		                                   threadCounts.end(), size_t{0});
		REQUIRE(sum == smallTotal);
	}
}
