/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <atomic>
#include <functional>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "yrt-pet/utils/Assert.hpp"

namespace yrt::util
{

template <typename F, typename... Args>
void parallelDo(size_t numThreads, F f, Args... args)
{
	std::vector<std::thread> workers;
	workers.reserve(numThreads);
	for (size_t i = 0; i < numThreads; i++)
		workers.emplace_back(f, args...);
	for (auto& worker : workers)
		worker.join();
}

template <typename F, typename... Args>
void parallelDoIndexed(size_t numThreads, F f, Args... args)
{
	std::vector<std::thread> workers;
	workers.reserve(numThreads);
	for (size_t i = 0; i < numThreads; i++)
		workers.emplace_back(f, i, args...);
	for (auto& worker : workers)
		worker.join();
}

// Function given as a parameter has two arguments: The position in the for loop
//  and the current thread's ID.
template <typename Func>
void parallelForChunkedRandomized(size_t total, size_t numThreads,
                                  float probability, Func fn)
{
	ASSERT_MSG(probability > 0.0f && probability <= 1.0f,
	           "Unsupported probability");

	using RNGSuite = std::tuple<std::mt19937_64,
	                            std::geometric_distribution<size_t>>;

	// One generator and distribution per thread (random_device is local only).
	std::vector<RNGSuite> rngs;
	rngs.reserve(numThreads);
	for (size_t threadId = 0; threadId < numThreads; ++threadId)
	{
		std::random_device rd;
		std::mt19937_64 gen{rd()};
		rngs.emplace_back(gen,
		                  std::geometric_distribution<size_t>{probability});
	}

	const size_t chunk = total / numThreads;
	std::vector<std::thread> threads;
	threads.reserve(numThreads);

	for (unsigned threadId = 0; threadId < numThreads; ++threadId)
	{
		const size_t start = threadId * chunk;
		const size_t end = (threadId + 1 == numThreads) ? total : start + chunk;
		threads.emplace_back(
		    [start, end, threadId, &rngs, &fn]()
		    {
			    auto& rng = rngs[threadId];
			    auto& gen = std::get<0>(rng);
			    auto& distribution = std::get<1>(rng);

			    size_t idx = start;
			    while (idx < end)
			    {
				    // Ask the generator how many elements to skip
				    const size_t skip = distribution(gen);

				    idx += skip;

				    // Don't overshoot the end of the data
				    if (idx < end)
				    {
					    fn(idx, threadId);

					    // Move to the next item after the selected one
					    idx++;
				    }
			    }
		    });
	}

	for (auto& th : threads)
	{
		th.join();
	}
}

// Function given as a parameter has two arguments: The position in the for loop
//  and the current thread's ID.
template <typename Func>
void parallelForChunked(size_t total, size_t numThreads, Func fn)
{
	const size_t chunk = total / numThreads;
	std::vector<std::thread> threads;
	threads.reserve(numThreads);

	for (unsigned threadId = 0; threadId < numThreads; ++threadId)
	{
		const size_t start = threadId * chunk;
		const size_t end = (threadId + 1 == numThreads) ? total : start + chunk;
		threads.emplace_back(
		    [start, end, threadId, &fn]()
		    {
			    for (size_t i = start; i < end; ++i)
			    {
				    fn(i, threadId);
			    }
		    });
	}

	for (auto& th : threads)
	{
		th.join();
	}
}

template <typename T, typename U>
T simpleReduceArray(const U* array, size_t length, std::function<T(T, T)> func,
                    T init, size_t numThreads = 1)
{
	std::vector<T> workspace;
	workspace.resize(numThreads);
	std::fill(workspace.begin(), workspace.end(), init);

	parallelForChunked(length, numThreads,
	                   [array, &workspace, func](size_t i, size_t tid)
	                   { workspace[tid] = func(workspace[tid], array[i]); });
	T output = init;
	for (size_t tid = 0; tid < numThreads; tid++)
	{
		output = func(output, workspace[tid]);
	}
	return output;
}

}  // namespace yrt::util
