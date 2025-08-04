/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/BinIteratorConstrained.hpp"

#include <cmath>
#include <cstdlib>
#include <memory>

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/utils/Tools.hpp"
#include "yrt-pet/utils/Types.hpp"

namespace yrt
{

// Constraints

bool Constraint::isValid(ConstraintParams& info) const
{
	return m_constraintFcn(info);
}
std::vector<ConstraintVariable> Constraint::getVariables() const
{
	return std::vector<ConstraintVariable>();
}

// Minimum angle difference constraint (index)
ConstraintAngleDiffIndex::ConstraintAngleDiffIndex(
    const ConstraintManager& p_manager, int p_minAngleDiffIdx)
{
	m_constraintFcn = [p_manager, p_minAngleDiffIdx](ConstraintParams& info)
	{
		int absDeltaAngleIdx = p_manager.getDataValue<int>(
		    info, 0, ConstraintVariable::AbsDeltaAngleIdx);
		return absDeltaAngleIdx >= p_minAngleDiffIdx;
	};
}
std::vector<ConstraintVariable> ConstraintAngleDiffIndex::getVariables() const
{
	return {ConstraintVariable::AbsDeltaAngleIdx};
}

// Minimum angle difference constraint (angle)
ConstraintAngleDiffDeg::ConstraintAngleDiffDeg(
    const ConstraintManager& p_manager, float p_minAngleDiffDeg)
{
	m_constraintFcn = [p_manager, p_minAngleDiffDeg](ConstraintParams& info)
	{
		float absDeltaAngleDeg = p_manager.getDataValue<float>(
			info, 0, ConstraintVariable::AbsDeltaAngleDeg);
		return absDeltaAngleDeg >= p_minAngleDiffDeg;
	};
}
std::vector<ConstraintVariable> ConstraintAngleDiffDeg::getVariables() const
{
	return {ConstraintVariable::AbsDeltaAngleDeg};
}

// Minimum angle difference constraint (index)
ConstraintBlockDiffIndex::ConstraintBlockDiffIndex(
    const ConstraintManager& p_manager, int p_minBlockDiffIdx)
{
	m_constraintFcn = [p_manager, p_minBlockDiffIdx](ConstraintParams& info)
	{
		int absDeltaBlockIdx = p_manager.getDataValue<int>(
			info, 0, ConstraintVariable::AbsDeltaBlockIdx);
		return absDeltaBlockIdx >= p_minBlockDiffIdx;
	};
}
std::vector<ConstraintVariable> ConstraintBlockDiffIndex::getVariables() const
{
	return {ConstraintVariable::AbsDeltaBlockIdx};
}

// Detector mask
ConstraintDetectorMask::ConstraintDetectorMask(
    const ConstraintManager& p_manager, const Scanner* scanner)
{
	m_constraintFcn = [p_manager, scanner](ConstraintParams& info)
	{
		det_id_t d1 =
		    p_manager.getDataValue<det_id_t>(info, 0, ConstraintVariable::Det1);
		det_id_t d2 =
		    p_manager.getDataValue<det_id_t>(info, 0, ConstraintVariable::Det2);
		return (scanner->isDetectorAllowed(d1) &&
		        scanner->isDetectorAllowed(d2));
	};
}
std::vector<ConstraintVariable> ConstraintDetectorMask::getVariables() const
{
	return {ConstraintVariable::Det1, ConstraintVariable::Det2};
}


// Constrained bin iterator
BinIteratorConstrained::BinIteratorConstrained(
    const ProjectionData* p_projData, std::vector<const Constraint*> p_constraints)
	: m_projData(p_projData), m_constraints(p_constraints)
{
	auto variables = collectVariables();
	m_constraintManager = std::make_unique<ConstraintManager>(variables);
}

std::set<ConstraintVariable> BinIteratorConstrained::collectVariables() const
{
	// List variables required by constraints
	std::set<ConstraintVariable> variables;
	for (auto constraint : m_constraints)
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
    std::set<ProjectionPropertyType>& projVariables,
    char*& projProps,
    ConstraintParams& consInfo) const
{
	auto [d1, d2] = m_projData->getDetectorPair(bin);
	m_constraintManager->setDataValue(consInfo, 0, ConstraintVariable::Det1,
	                                  d1);
	m_constraintManager->setDataValue(consInfo, 0, ConstraintVariable::Det2,
	                                  d2);

	bool needsLOR =
		projVariables.find(ProjectionPropertyType::LOR) != projVariables.end() ||
	    consVariables.find(ConstraintVariable::AbsDeltaAngleDeg) != consVariables.end();
	Line3D lor;
	if (needsLOR)
	{
		lor = m_projData->getLOR(bin);
		m_propManager->setDataValue(projProps, 0, ProjectionPropertyType::LOR, lor);
	}
	const Scanner* scanner = &m_projData->getScanner();

	if (consVariables.find(ConstraintVariable::AbsDeltaAngleDeg) != consVariables.end())
	{
		// In-plane angle
		float a1 = std::atan2(lor.point1.y, lor.point1.x);
		float a2 = std::atan2(lor.point2.y, lor.point2.x);
		float diff = util::periodicDiff(a1, a2, (float)(2.f * PI));
		m_constraintManager->setDataValue(
		    consInfo, 0, ConstraintVariable::AbsDeltaAngleDeg, diff);
	}

	bool needsPlaneIdx =
	    consVariables.find(ConstraintVariable::AbsDeltaAngleIdx) !=
	        consVariables.end() ||
	    consVariables.find(ConstraintVariable::AbsDeltaBlockIdx) !=
	        consVariables.end();
	int d1xyi;
	int d2xyi;
	if (needsPlaneIdx)
	{
		d1xyi = d1 % scanner->detsPerRing;
		d2xyi = d2 % scanner->detsPerRing;
	}
	if (consVariables.find(ConstraintVariable::AbsDeltaAngleIdx) != consVariables.end())
	{
		int diff = util::periodicDiff(d1xyi, d2xyi,
		                              static_cast<int>(scanner->detsPerRing));
		m_constraintManager->setDataValue(
		    consInfo, 0, ConstraintVariable::AbsDeltaAngleIdx, diff);
	}

	bool needsPlaneBlock =
	    consVariables.find(ConstraintVariable::AbsDeltaBlockIdx) != consVariables.end();
	int d1bi;
	int d2bi;
	if (needsPlaneBlock)
	{
		d1bi = d1xyi / scanner->detsPerBlock;
		d2bi = d2xyi / scanner->detsPerBlock;
	}
	if (consVariables.find(ConstraintVariable::AbsDeltaBlockIdx) != consVariables.end())
	{
		int diff = util::periodicDiff(d1bi, d2bi,
		                              static_cast<int>(scanner->detsPerBlock));
		m_constraintManager->setDataValue(
		    consInfo, 0, ConstraintVariable::AbsDeltaBlockIdx, diff);
	}
}

bool BinIteratorConstrained::isValid(ConstraintParams& info) const
{
	for (auto constraint : m_constraints)
	{
		if (!constraint->isValid(info))
		{
			return false;
		}
	}
	return true;
}

}  // namespace yrt
