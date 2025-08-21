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
void parallel_do(size_t threadCnt, F f, Args... args)
{
	std::vector<std::thread> workers;
	workers.reserve(threadCnt);
	for (size_t i = 0; i < threadCnt; i++)
		workers.emplace_back(f, args...);
	for (auto& worker : workers)
		worker.join();
}

template <typename F, typename... Args>
void parallel_do_indexed(size_t threadCnt, F f, Args... args)
{
	std::vector<std::thread> workers;
	workers.reserve(threadCnt);
	for (size_t i = 0; i < threadCnt; i++)
		workers.emplace_back(f, i, args...);
	for (auto& worker : workers)
		worker.join();
}

// Function given as a parameter has two arguments: The position in the for loop
//  and the current thread's ID.
template <typename Func>
void parallel_for_chunked(size_t total, size_t threadCnt, Func fn)
{
	const size_t chunk = total / threadCnt;
	std::vector<std::thread> threads;
	threads.reserve(threadCnt);

	for (unsigned t = 0; t < threadCnt; ++t)
	{
		const size_t start = t * chunk;
		const size_t end = (t + 1 == threadCnt) ? total : start + chunk;
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
                    T init, size_t threadCnt = 1)
{
    std::vector<T> workspace;
	workspace.reserve(threadCnt);
	std::fill(workspace.begin(), workspace.end(), init);

    parallel_for_chunked(length, threadCnt,
                         [array, &workspace, func, init](size_t i, size_t tid)
                         {
                             workspace[tid] = func(workspace[tid], array[i]);
                         });
    T output = init;
    for (size_t tid = 0; tid < threadCnt; tid++)
    {
        output = func(output, workspace[tid]);
    }
    return output;
}

}  // namespace yrt::util
