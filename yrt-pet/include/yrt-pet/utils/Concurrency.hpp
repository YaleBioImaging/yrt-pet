/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <thread>
#include <vector>

namespace yrt::util {
  template<typename F, typename... Args>
  void parallel_do (size_t threadCnt, F f, Args... args) {
    std::vector<std::thread> workers;
    workers.reserve(threadCnt);
    for (size_t i = 0; i < threadCnt; i++) workers.emplace_back(f, args...);
    for (auto& worker : workers) worker.join();
  }

  template<typename F, typename... Args>
  void parallel_do_indexed (size_t threadCnt, F f, Args... args) {
    std::vector<std::thread> workers;
    workers.reserve(threadCnt);
    for (size_t i = 0; i < threadCnt; i++) workers.emplace_back(f, i, args...);
    for (auto& worker : workers) worker.join();
  }
}

