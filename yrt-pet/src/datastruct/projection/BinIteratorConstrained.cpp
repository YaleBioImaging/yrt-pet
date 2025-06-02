#include "datastruct/projection/BinIteratorConstrained.hpp"
#include "datastruct/projection/ProjectionData.hpp"
#include <cstdlib>
#include <cmath>
#include "geometry/Constants.hpp"
#include "utils/Tools.hpp"

// Constraints

bool Constraint::isValid(constraint_params info) const
{
	return mConstraintFcn(info);
}
std::vector<std::string> Constraint::getVariables() const
{
	return std::vector<std::string>();
}

// Minimum angle difference constraint (index)
ConstraintAngleDiffIndex::ConstraintAngleDiffIndex(size_t pMinAngleDiffIdx)
{
	mConstraintFcn = [pMinAngleDiffIdx](constraint_params info)
	{
		return info["abs_delta_angle_idx"] >= pMinAngleDiffIdx;
	};
}
std::vector<std::string> ConstraintAngleDiffIndex::getVariables() const
{
	return {"abs_delta_angle_idx"};
}

// Minimum angle difference constraint (angle)
ConstraintAngleDiffDeg::ConstraintAngleDiffDeg(size_t pMinAngleDiffDeg)
{
	mConstraintFcn = [pMinAngleDiffDeg](constraint_params info)
	{
		return info["abs_delta_angle_deg"] >= pMinAngleDiffDeg;
	};
}
std::vector<std::string> ConstraintAngleDiffDeg::getVariables() const
{
	return {"abs_delta_angle_deg"};
}

// Minimum angle difference constraint (index)
ConstraintBlockDiffIndex::ConstraintBlockDiffIndex(size_t pMinBlockDiffIdx)
{
	mConstraintFcn = [pMinBlockDiffIdx](constraint_params info)
	{
		return info["abs_delta_block_idx"] >= pMinBlockDiffIdx;
	};
}
std::vector<std::string> ConstraintBlockDiffIndex::getVariables() const
{
	return {"abs_delta_block_idx"};
}

// Detector mask
ConstraintDetectorMask::ConstraintDetectorMask(Scanner* scanner)
{
	mConstraintFcn = [scanner](constraint_params info)
	{
		return (scanner->isDetectorAllowed(info["det1"]) &&
		        scanner->isDetectorAllowed(info["det2"]));
	};
}
std::vector<std::string> ConstraintDetectorMask::getVariables() const
{
	return {"det1", "det2"};
}


// Constrained bin iterator
BinIteratorConstrained::BinIteratorConstrained(ProjectionData* pProjData,
                                               BinIterator* pBinIterBase,
                                               int pQueueSizeMax)
	: mProjData(pProjData), mBinIterBase(pBinIterBase), mQueue(pQueueSizeMax)
{
}


void BinIteratorConstrained::addConstraint(Constraint& pConstraint)
{
	mConstraints.push_back(&pConstraint);
	mCount = 0;
}

std::set<std::string> BinIteratorConstrained::collectVariables() const
{
	// List variables required by constraints
	std::set<std::string> variables;
	for (auto constraint : mConstraints)
	{
		for (auto variable : constraint->getVariables())
		{
			variables.insert(variable);
		}
	}
	return variables;
}

constraint_params
    BinIteratorConstrained::collectInfo(
	    bin_t bin, std::set<std::string> variables) const
{
	constraint_params info;

	auto [d1, d2] = mProjData->getDetectorPair(bin);
	info["det1"] = d1;
	info["det2"] = d2;

	bool needsLOR = variables.find("abs_delta_angle_deg") != variables.end();
	Line3D lor;
	if (needsLOR)
	{
		lor = mProjData->getLOR(bin);
	}

	const Scanner* scanner = nullptr;
	bool needsScanner = (
		variables.find("abs_delta_block_idx") != variables.end() ||
		variables.find("abs_delta_angle_idx") != variables.end());
	if (needsScanner)
	{
		scanner = &mProjData->getScanner();
	}

	if (variables.find("abs_delta_angle_deg") != variables.end())
	{
		// In-plane angle
		float a1 = std::atan2(lor.point1.y, lor.point1.x);
		float a2 = std::atan2(lor.point2.y, lor.point2.x);
		info["abs_delta_angle_deg"] =
		    Util::periodicDiff(a1, a2, (float)(2.f * PI));
	}

	bool needsPlaneIdx =
	    variables.find("abs_delta_angle_idx") != variables.end() ||
		variables.find("abs_delta_block_idx") != variables.end();
	size_t d1xyi;
	size_t d2xyi;
	if (needsPlaneIdx)
	{
		d1xyi = d1 % scanner->detsPerRing;
		d2xyi = d2 % scanner->detsPerRing;
	}
	if (variables.find("abs_delta_angle_idx") != variables.end())
	{
		info["abs_delta_angle_idx"] = Util::periodicDiff(
			d1xyi, d2xyi, scanner->detsPerRing);
	}

	bool needsPlaneBlock =
	    variables.find("abs_delta_block_idx") != variables.end();
	size_t d1bi;
	size_t d2bi;
	if (needsPlaneBlock)
	{
		d1bi = d1xyi / scanner->detsPerBlock;
		d2bi = d2xyi / scanner->detsPerBlock;
	}
	if (variables.find("abs_delta_block_idx") != variables.end())
	{
		info["abs_delta_block_idx"] = Util::periodicDiff(
			d1bi, d2bi, scanner->detsPerBlock);
	}

	return info;
}

bool BinIteratorConstrained::isValid(constraint_params info) const
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
		for (size_t binIdx = 0; binIdx < mBinIterBase->size(); binIdx++)
		{
			bin_t bin = mBinIterBase->get(binIdx);
			auto info = collectInfo(bin, variables);
			if (isValid(info))
			{
				count++;
			}
		}
		return count;
	}
}

void BinIteratorConstrained::produce()
{
	auto variables = collectVariables();
	for (size_t binIdx = 0; binIdx < mBinIterBase->size(); binIdx++)
	{
		bin_t bin = mBinIterBase->get(binIdx);
		auto info = collectInfo(bin, variables);
		if (isValid(info))
		{
			// TODO get properties
			const Line3D lor = mProjData->getLOR(bin);
			float tofValue = mProjData->getTOFValue(bin);;
			const Vector3D det1Orient = mProjData->getScanner().getDetectorOrient(info["det1"]);
			const Vector3D det2Orient = mProjData->getScanner().getDetectorOrient(info["det2"]);
			// TODO check size
			mQueue.wait_and_push(
			    ProjectionProperties({lor, tofValue, det1Orient, det2Orient}));
		}
	}
}

const ProjectionProperties& BinIteratorConstrained::get()
{
	return mQueue.wait_and_pop();
}
