#include <unordered_map>
#include <vector>
#include <set>
#include <functional>

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <atomic>

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"

namespace yrt
{

enum class ConstraintVariable
{
	Det1,
	Det2,
	AbsDeltaAngleDeg,
	AbsDeltaAngleIdx,
	AbsDeltaBlockIdx
};

using ConstraintParams = std::unordered_map<ConstraintVariable, size_t>;
class Constraint
{
public:
	bool isValid(ConstraintParams& info) const;
	virtual std::vector<ConstraintVariable> getVariables() const = 0;
protected:
	std::function<bool(ConstraintParams&)> mConstraintFcn;
};

class ConstraintAngleDiffIndex : public Constraint
{
public:
	ConstraintAngleDiffIndex(size_t pMinAngleDiffIdx);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintAngleDiffDeg : public Constraint
{
public:
	ConstraintAngleDiffDeg(size_t pMinAngleDiffDeg);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintBlockDiffIndex : public Constraint
{
public:
	ConstraintBlockDiffIndex(size_t pMinBlockDiffIdx);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintDetectorMask : public Constraint
{
public:
	ConstraintDetectorMask(Scanner* scanner);
	std::vector<ConstraintVariable> getVariables() const override;
};

class BinIteratorConstrained
{
public:
	BinIteratorConstrained(const ProjectionData* pProjData,
	                       const BinIterator* pBinIterBase);
	void addConstraint(Constraint& pConstraint);
	size_t count();

	std::set<ConstraintVariable> collectVariables() const;
	void collectInfo(bin_t bin, std::set<ConstraintVariable>& variables,
	                 ConstraintParams& info) const;
	bool isValid(ConstraintParams& info) const;

private:
	const ProjectionData* mProjData;
	const BinIterator* mBinIterBase;
	std::vector<Constraint*> mConstraints;
	size_t mCount;
	float mQueueFrac;


	// Loop variables
	std::set<ConstraintVariable> mVariables;
	std::atomic<size_t> mBinIdx;

};

}  // namespace yrt
