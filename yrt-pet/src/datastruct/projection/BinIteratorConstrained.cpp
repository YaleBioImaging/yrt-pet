#include "datastruct/projection/BinIteratorConstrained.hpp"
#include "datastruct/projection/ProjectionData.hpp"
#include <cstdlib>
#include <cmath>
#include "geometry/Constants.hpp"
#include "utils/Tools.hpp"

// Constraints

bool Constraint::isValid(ConstraintParams& info) const
{
	return mConstraintFcn(info);
}
std::vector<ConstraintVariable> Constraint::getVariables() const
{
	return std::vector<ConstraintVariable>();
}

// Minimum angle difference constraint (index)
ConstraintAngleDiffIndex::ConstraintAngleDiffIndex(size_t pMinAngleDiffIdx)
{
	mConstraintFcn = [pMinAngleDiffIdx](ConstraintParams& info)
	{
		return info[ConstraintVariable::AbsDeltaAngleIdx] >= pMinAngleDiffIdx;
	};
}
std::vector<ConstraintVariable> ConstraintAngleDiffIndex::getVariables() const
{
	return {ConstraintVariable::AbsDeltaAngleIdx};
}

// Minimum angle difference constraint (angle)
ConstraintAngleDiffDeg::ConstraintAngleDiffDeg(size_t pMinAngleDiffDeg)
{
	mConstraintFcn = [pMinAngleDiffDeg](ConstraintParams& info)
	{
		return info[ConstraintVariable::AbsDeltaAngleDeg] >= pMinAngleDiffDeg;
	};
}
std::vector<ConstraintVariable> ConstraintAngleDiffDeg::getVariables() const
{
	return {ConstraintVariable::AbsDeltaAngleDeg};
}

// Minimum angle difference constraint (index)
ConstraintBlockDiffIndex::ConstraintBlockDiffIndex(size_t pMinBlockDiffIdx)
{
	mConstraintFcn = [pMinBlockDiffIdx](ConstraintParams& info)
	{
		return info[ConstraintVariable::AbsDeltaBlockIdx] >= pMinBlockDiffIdx;
	};
}
std::vector<ConstraintVariable> ConstraintBlockDiffIndex::getVariables() const
{
	return {ConstraintVariable::AbsDeltaBlockIdx};
}

// Detector mask
ConstraintDetectorMask::ConstraintDetectorMask(Scanner* scanner)
{
	mConstraintFcn = [scanner](ConstraintParams& info)
	{
		return (scanner->isDetectorAllowed(info[ConstraintVariable::Det1]) &&
		        scanner->isDetectorAllowed(info[ConstraintVariable::Det2]));
	};
}
std::vector<ConstraintVariable> ConstraintDetectorMask::getVariables() const
{
	return {ConstraintVariable::Det1, ConstraintVariable::Det2};
}


// Constrained bin iterator
BinIteratorConstrained::BinIteratorConstrained(const ProjectionData* pProjData,
                                               const BinIterator* pBinIterBase,
                                               int pQueueSizeMax,
                                               float pQueueFrac)
    : mProjData(pProjData),
      mBinIterBase(pBinIterBase),
      mQueue(pQueueSizeMax),
      mCount(0),
      mQueueFrac(pQueueFrac)
{
}


void BinIteratorConstrained::addConstraint(Constraint& pConstraint)
{
	mConstraints.push_back(&pConstraint);
	mCount = 0;
}

std::set<ConstraintVariable> BinIteratorConstrained::collectVariables() const
{
	// List variables required by constraints
	std::set<ConstraintVariable> variables;
	for (auto constraint : mConstraints)
	{
		for (auto variable : constraint->getVariables())
		{
			variables.insert(variable);
		}
	}
	return variables;
}

void BinIteratorConstrained::collectInfo(
    bin_t bin, std::set<ConstraintVariable>& variables,
    ConstraintParams& info) const
{
	auto [d1, d2] = mProjData->getDetectorPair(bin);
	info[ConstraintVariable::Det1] = d1;
	info[ConstraintVariable::Det2] = d2;

	Line3D lor = mProjData->getLOR(bin);
	const Scanner* scanner = &mProjData->getScanner();

	if (variables.find(ConstraintVariable::AbsDeltaAngleDeg) != variables.end())
	{
		// In-plane angle
		float a1 = std::atan2(lor.point1.y, lor.point1.x);
		float a2 = std::atan2(lor.point2.y, lor.point2.x);
		info[ConstraintVariable::AbsDeltaAngleDeg] =
		    Util::periodicDiff(a1, a2, (float)(2.f * PI));
	}

	bool needsPlaneIdx =
	    variables.find(ConstraintVariable::AbsDeltaAngleIdx) !=
	        variables.end() ||
	    variables.find(ConstraintVariable::AbsDeltaBlockIdx) != variables.end();
	size_t d1xyi;
	size_t d2xyi;
	if (needsPlaneIdx)
	{
		d1xyi = d1 % scanner->detsPerRing;
		d2xyi = d2 % scanner->detsPerRing;
	}
	if (variables.find(ConstraintVariable::AbsDeltaAngleIdx) != variables.end())
	{
		info[ConstraintVariable::AbsDeltaAngleIdx] =
		    Util::periodicDiff(d1xyi, d2xyi, scanner->detsPerRing);
	}

	bool needsPlaneBlock =
	    variables.find(ConstraintVariable::AbsDeltaBlockIdx) != variables.end();
	size_t d1bi;
	size_t d2bi;
	if (needsPlaneBlock)
	{
		d1bi = d1xyi / scanner->detsPerBlock;
		d2bi = d2xyi / scanner->detsPerBlock;
	}
	if (variables.find(ConstraintVariable::AbsDeltaBlockIdx) != variables.end())
	{
		info[ConstraintVariable::AbsDeltaBlockIdx] =
		    Util::periodicDiff(d1bi, d2bi, scanner->detsPerBlock);
	}
}

bool BinIteratorConstrained::isValid(ConstraintParams& info) const
{
	for (auto constraint : mConstraints)
	{
		if (!constraint->isValid(info))
		{
			return false;
		}
	}
	return true;
}


size_t BinIteratorConstrained::count()
{
	if (mCount != 0)
	{
		return mCount;
	}
	else
	{
		auto variables = collectVariables();
		size_t count = 0;
#pragma omp parallel for reduction (+ : count)
		for (size_t binIdx = 0; binIdx < mBinIterBase->size(); binIdx++)
		{
			ConstraintParams info;
			bin_t bin = mBinIterBase->get(binIdx);
			collectInfo(bin, variables, info);
			if (isValid(info))
			{
				count++;
			}
		}
		return count;
	}
}

void BinIteratorConstrained::prepare()
{
	mVariables = collectVariables();
	count();
	mBinIdx = 0;
}

void BinIteratorConstrained::produceNext(ConstraintParams& info)
{
	if (done())
	{
		return;
	}
	bin_t bin = mBinIterBase->get(mBinIdx);
	collectInfo(bin, mVariables, info);
	if (isValid(info))
	{
		// TODO get properties
		const Line3D lor = mProjData->getLOR(bin);
		float tofValue = 0.f;
		if (mProjData->hasTOF())
		{
			tofValue = mProjData->getTOFValue(bin);
		}
		const Vector3D det1Orient =
		    mProjData->getScanner().getDetectorOrient(info[ConstraintVariable::Det1]);
		const Vector3D det2Orient =
		    mProjData->getScanner().getDetectorOrient(info[ConstraintVariable::Det2]);
		// TODO check size
		mQueue.wait_and_push(
		    ProjectionProperties({bin, lor, tofValue, det1Orient, det2Orient}));
	}
	mBinIdx++;
}

const ProjectionProperties& BinIteratorConstrained::get()
{
	return mQueue.wait_and_pop();
}

bool BinIteratorConstrained::nextTaskProduce() const
{
	return mQueue.size() < mQueueFrac * mQueue.GetSizeMax();
}

bool BinIteratorConstrained::done() const
{
	return mBinIdx >= mBinIterBase->size();
}
