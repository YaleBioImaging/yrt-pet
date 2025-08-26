/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <atomic>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

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
void parallelForChunked(size_t total, size_t numThreads, Func fn)
{
	const size_t chunk = total / numThreads;
	std::vector<std::thread> threads;
	threads.reserve(numThreads);

	for (unsigned t = 0; t < numThreads; ++t)
	{
		const size_t start = t * chunk;
		const size_t end = (t + 1 == numThreads) ? total : start + chunk;
		threads.emplace_back(
		    [=, &fn]()
		    {
			    for (size_t i = start; i < end; ++i)
			    {
				    fn(i, t);
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
