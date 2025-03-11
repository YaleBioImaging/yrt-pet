#include <unordered_map>
#include <vector>
#include <set>
#include <functional>

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"

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

private:
	mutable std::mutex mtx;
	std::condition_variable cv_push;
	std::condition_variable cv_pop;
	std::queue<T> queue;
	size_t sizeMax;
};

using constraint_params = std::unordered_map<std::string, size_t>;
class Constraint
{
public:
	bool isValid(constraint_params info) const;
	virtual std::vector<std::string> getVariables() const = 0;
protected:
	std::function<bool(constraint_params)> mConstraintFcn;
};

class ConstraintAngleDiffIndex : public Constraint
{
public:
	ConstraintAngleDiffIndex(size_t pMinAngleDiffIdx);
	std::vector<std::string> getVariables() const override;
};
class ConstraintAngleDiffDeg : public Constraint
{
public:
	ConstraintAngleDiffDeg(size_t pMinAngleDiffDeg);
	std::vector<std::string> getVariables() const override;
};
class ConstraintDetectorMask : public Constraint
{
public:
	ConstraintDetectorMask(Scanner* scanner);
	std::vector<std::string> getVariables() const override;
};
class ConstraintHistoSubset : public Constraint
{
public:
	ConstraintHistoSubset(size_t p_numZBin, size_t p_numPhi,
	                      size_t p_numR, int p_numSubsets,
	                      int p_idxSubset);
	std::vector<std::string> getVariables() const override;
};

class BinIteratorConstrained
{
	BinIteratorConstrained(size_t pNumBins, int pQueueSizeMax);
	void addConstraint(Constraint& pConstraint);
	size_t count() const;
	void produce();
	const ProjectionProperties& get();

private:
	size_t mNumBins;
	std::vector<Constraint*> mConstraints;
	ThreadSafeQueue<ProjectionProperties> mQueue;
	size_t mCount;

	std::set<std::string> collectVariables() const;
	constraint_params collectInfo(size_t bin,
	                              std::set<std::string> variables) const;
	bool isValid(constraint_params info) const;
};

}  // namespace yrt
