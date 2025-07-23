#include "datastruct/projection/BinIteratorConstrained.hpp"
#include <cstdlib>

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

// Detector mask
ConstraintDetectorMask::ConstraintDetectorMask(Scanner* scanner)
{
	mConstraintFcn = [scanner](constraint_params info)
	{
		return scanner->isDetectorAllowed(info["det1"]);
	};
}
std::vector<std::string> ConstraintDetectorMask::getVariables() const
{
	return {"det1"};
}

// Constrained bin iterator
BinIteratorConstrained::BinIteratorConstrained(size_t pNumBins,
                                               int pQueueSizeMax)
    : mNumBins(pNumBins), mQueue(pQueueSizeMax)
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
        size_t bin, std::set<std::string> variables) const
{
	constraint_params info;
	// TODO
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


size_t BinIteratorConstrained::count() const
{
	if (mCount != 0)
	{
		return mCount;
	}
	else
	{
		auto variables = collectVariables();
		size_t count = 0;
		for (size_t bin = 0; bin < mNumBins; bin++)
		{
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
	for (size_t bin = 0; bin < mNumBins; bin++)
	{
		auto info = collectInfo(bin, variables);
		if (isValid(info))
		{
			// TODO get properties
			const Line3D lor{};
			float tofValue;
			const Vector3D det1Orient{};
			const Vector3D det2Orient{};
			// TODO check size
			mQueue.wait_and_push(
			    ProjectionProperties({lor, tofValue, det1Orient, det2Orient}));
		}
	}
}
