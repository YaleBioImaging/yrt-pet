/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/BinIteratorConstrained.hpp"

#include <cmath>
#include <cstdlib>

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/utils/Tools.hpp"

namespace yrt
{

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
	{ return info[ConstraintVariable::AbsDeltaAngleIdx] >= pMinAngleDiffIdx; };
}
std::vector<ConstraintVariable> ConstraintAngleDiffIndex::getVariables() const
{
	return {ConstraintVariable::AbsDeltaAngleIdx};
}

// Minimum angle difference constraint (angle)
ConstraintAngleDiffDeg::ConstraintAngleDiffDeg(size_t pMinAngleDiffDeg)
{
	mConstraintFcn = [pMinAngleDiffDeg](ConstraintParams& info)
	{ return info[ConstraintVariable::AbsDeltaAngleDeg] >= pMinAngleDiffDeg; };
}
std::vector<ConstraintVariable> ConstraintAngleDiffDeg::getVariables() const
{
	return {ConstraintVariable::AbsDeltaAngleDeg};
}

// Minimum angle difference constraint (index)
ConstraintBlockDiffIndex::ConstraintBlockDiffIndex(size_t pMinBlockDiffIdx)
{
	mConstraintFcn = [pMinBlockDiffIdx](ConstraintParams& info)
	{ return info[ConstraintVariable::AbsDeltaBlockIdx] >= pMinBlockDiffIdx; };
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
                                               const BinIterator* pBinIterBase)
    : mProjData(pProjData), mBinIterBase(pBinIterBase), mCount(0)
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
    bin_t bin, std::set<ConstraintVariable>& consVariables,
    std::set<ProjectionPropertiesVariable>& projVariables,
    ProjectionProperties& projProps,
    ConstraintParams& consInfo) const
{
	auto [d1, d2] = mProjData->getDetectorPair(bin);
	consInfo[ConstraintVariable::Det1] = d1;
	consInfo[ConstraintVariable::Det2] = d2;

	bool needsLOR =
		projVariables.find(ProjectionPropertiesVariable::LOR) != projVariables.end() ||
	    consVariables.find(ConstraintVariable::AbsDeltaAngleDeg) != consVariables.end();
	Line3D lor;
	if (needsLOR)
	{
		lor = mProjData->getLOR(bin);
		projProps.lor = lor;
	}
	const Scanner* scanner = &mProjData->getScanner();

	if (consVariables.find(ConstraintVariable::AbsDeltaAngleDeg) != consVariables.end())
	{
		// In-plane angle
		float a1 = std::atan2(lor.point1.y, lor.point1.x);
		float a2 = std::atan2(lor.point2.y, lor.point2.x);
		consInfo[ConstraintVariable::AbsDeltaAngleDeg] =
		    util::periodicDiff(a1, a2, (float)(2.f * PI));
	}

	bool needsPlaneIdx =
	    consVariables.find(ConstraintVariable::AbsDeltaAngleIdx) !=
	        consVariables.end() ||
	    consVariables.find(ConstraintVariable::AbsDeltaBlockIdx) !=
	        consVariables.end();
	size_t d1xyi;
	size_t d2xyi;
	if (needsPlaneIdx)
	{
		d1xyi = d1 % scanner->detsPerRing;
		d2xyi = d2 % scanner->detsPerRing;
	}
	if (consVariables.find(ConstraintVariable::AbsDeltaAngleIdx) != consVariables.end())
	{
		consInfo[ConstraintVariable::AbsDeltaAngleIdx] =
		    util::periodicDiff(d1xyi, d2xyi, scanner->detsPerRing);
	}

	bool needsPlaneBlock =
	    consVariables.find(ConstraintVariable::AbsDeltaBlockIdx) != consVariables.end();
	size_t d1bi;
	size_t d2bi;
	if (needsPlaneBlock)
	{
		d1bi = d1xyi / scanner->detsPerBlock;
		d2bi = d2xyi / scanner->detsPerBlock;
	}
	if (consVariables.find(ConstraintVariable::AbsDeltaBlockIdx) != consVariables.end())
	{
		consInfo[ConstraintVariable::AbsDeltaBlockIdx] =
		    util::periodicDiff(d1bi, d2bi, scanner->detsPerBlock);
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
#pragma omp parallel for reduction(+ : count)
		for (size_t binIdx = 0; binIdx < mBinIterBase->size(); binIdx++)
		{
			ConstraintParams info;
			bin_t bin = mBinIterBase->get(binIdx);
			// FIXME: collectInfo(bin, variables, info);
			if (isValid(info))
			{
				count++;
			}
		}
		return count;
	}
}

}  // namespace yrt
