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
	constexpr size_t total = 1 << 16;
	constexpr size_t numThreads = 4;
	constexpr float probability = 0.05f;

	SECTION("probability-1-all")
	{
		std::vector<size_t> threadCounts(numThreads, 0);
		yrt::util::parallelForChunkedRandomized(
		    total, numThreads, 1.0f,
		    [&threadCounts](size_t, size_t tid) { threadCounts[tid]++; });
		const size_t sum = std::accumulate(threadCounts.begin(),
		                                   threadCounts.end(), size_t{0});
		REQUIRE(sum == total);
	}

	SECTION("indices-in-bounds")
	{
		std::atomic<bool> outOfRange{false};
		yrt::util::parallelForChunkedRandomized(
		    total, numThreads, 0.5f,
		    [&outOfRange](size_t idx, size_t)
		    {
			    if (idx >= total)
				    outOfRange = true;
		    });
		REQUIRE_FALSE(outOfRange);
	}

	SECTION("probability-count-scale")
	{
		std::vector<size_t> threadCounts(numThreads, 0);

		// Reference: full count via parallelForChunked
		yrt::util::parallelForChunked(total, numThreads,
		                              [&threadCounts](size_t, size_t tid)
		                              { threadCounts[tid]++; });
		const size_t fullSum = std::accumulate(threadCounts.begin(),
		                                       threadCounts.end(), size_t{0});
		REQUIRE(fullSum == total);

		// Reset and run with probability via parallelForChunkedRandomized
		std::fill(threadCounts.begin(), threadCounts.end(), size_t{0});
		yrt::util::parallelForChunkedRandomized(
		    total, numThreads, probability,
		    [&threadCounts](size_t, size_t tid) { threadCounts[tid]++; });
		const size_t randomSum = std::accumulate(threadCounts.begin(),
		                                         threadCounts.end(), size_t{0});

		// Expected: p * total
		constexpr float expected = static_cast<float>(total) * probability;
		REQUIRE(static_cast<float>(randomSum) == Approx(expected).epsilon(0.1));
	}
}
