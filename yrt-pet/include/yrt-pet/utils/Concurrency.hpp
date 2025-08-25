/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <thread>
#include <vector>

namespace yrt::util
{

template <typename F, typename... Args>
void parallel_do(size_t numThreads, F f, Args... args)
{
	std::vector<std::thread> workers;
	workers.reserve(numThreads);
	for (size_t i = 0; i < numThreads; i++)
		workers.emplace_back(f, args...);
	for (auto& worker : workers)
		worker.join();
}

template <typename F, typename... Args>
void parallel_do_indexed(size_t numThreads, F f, Args... args)
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
void parallel_for_chunked(size_t total, size_t numThreads, Func fn)
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

}  // namespace yrt::util
