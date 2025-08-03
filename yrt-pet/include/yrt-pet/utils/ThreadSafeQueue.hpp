/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>

namespace yrt
{

template <typename T>
class ThreadSafeQueue
{
public:
	ThreadSafeQueue(size_t size) : sizeMax(size) {}

	void wait_and_push(const T& value)
	{
		std::unique_lock<std::mutex> lock(mtx);
		cv_push.wait(lock, [this] { return queue.size() < sizeMax; });
		queue.push(value);
		cv_pop.notify_one();  // Notify all waiting consumers
	}

	T& wait_and_pop()
	{
		std::unique_lock<std::mutex> lock(mtx);
		cv_pop.wait(lock, [this] { return !queue.empty(); });
		T& value = queue.front();
		queue.pop();
		cv_push.notify_one();  // Notify producer
		return value;
	}

	bool empty() const
	{
		std::lock_guard<std::mutex> lock(mtx);
		return queue.empty();
	}

	size_t size() const
	{
		std::lock_guard<std::mutex> lock(mtx);
		return queue.size();
	}

	size_t GetSizeMax() const { return sizeMax; }

private:
	mutable std::mutex mtx;
	std::condition_variable cv_push;
	std::condition_variable cv_pop;
	std::queue<T> queue;
	size_t sizeMax;
};

}  // namespace yrt
