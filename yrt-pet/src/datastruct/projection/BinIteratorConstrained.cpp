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

bool Constraint::isValid(const ConstraintManager& manager, ConstraintParams& info) const
{
	return m_constraintFcn(manager, info);
}
std::vector<ConstraintVariable> Constraint::getVariables() const
{
	return std::vector<ConstraintVariable>();
}

// Minimum angle difference constraint (index)
ConstraintAngleDiffIndex::ConstraintAngleDiffIndex(int p_minAngleDiffIdx)
{
	m_constraintFcn = [p_minAngleDiffIdx](
		const ConstraintManager& manager, ConstraintParams& info)
	{
		int absDeltaAngleIdx = manager.getDataValue<int>(
		    info, 0, ConstraintVariable::AbsDeltaAngleIdx);
		return absDeltaAngleIdx >= p_minAngleDiffIdx;
	};
}
std::vector<ConstraintVariable> ConstraintAngleDiffIndex::getVariables() const
{
	return {ConstraintVariable::AbsDeltaAngleIdx};
}

// Minimum angle difference constraint (angle)
ConstraintAngleDiffDeg::ConstraintAngleDiffDeg(float p_minAngleDiffDeg)
{
	m_constraintFcn = [p_minAngleDiffDeg](const ConstraintManager& manager,
	                                      ConstraintParams& info)
	{
		float absDeltaAngleDeg = manager.getDataValue<float>(
		    info, 0, ConstraintVariable::AbsDeltaAngleDeg);
		return absDeltaAngleDeg >= p_minAngleDiffDeg;
	};
}
std::vector<ConstraintVariable> ConstraintAngleDiffDeg::getVariables() const
{
	return {ConstraintVariable::AbsDeltaAngleDeg};
}

// Minimum angle difference constraint (index)
ConstraintBlockDiffIndex::ConstraintBlockDiffIndex(int p_minBlockDiffIdx)
{
	m_constraintFcn = [p_minBlockDiffIdx](const ConstraintManager& manager,
	                                      ConstraintParams& info)
	{
		int absDeltaBlockIdx = manager.getDataValue<int>(
		    info, 0, ConstraintVariable::AbsDeltaBlockIdx);
		return absDeltaBlockIdx >= p_minBlockDiffIdx;
	};
}
std::vector<ConstraintVariable> ConstraintBlockDiffIndex::getVariables() const
{
	return {ConstraintVariable::AbsDeltaBlockIdx};
}

// Detector mask
ConstraintDetectorMask::ConstraintDetectorMask(const Scanner* scanner)
{
	m_constraintFcn = [scanner](const ConstraintManager& manager,
	                            ConstraintParams& info)
	{
		det_id_t d1 =
		    manager.getDataValue<det_id_t>(info, 0, ConstraintVariable::Det1);
		det_id_t d2 =
		    manager.getDataValue<det_id_t>(info, 0, ConstraintVariable::Det2);
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
    const std::vector<Constraint*>& constraints,
    const std::set<ProjectionPropertyType>& projVariables)
	: m_constraints(constraints),
	  m_projVariables(projVariables)
{
	setupManagers();
	collectConstraintVariables();
}

void BinIteratorConstrained::clearConstraints()
{
	m_constraints.clear();
	m_consVariables.clear();
	m_constraintManager = nullptr;
}

void BinIteratorConstrained::setupManagers()
{
	if (m_constraints.size() > 0)
	{
		collectConstraintVariables();
		m_constraintManager =
		    std::make_unique<ConstraintManager>(m_consVariables);
	}
	m_propManager =
	    std::make_unique<ProjectionPropertyManager>(m_projVariables);
}

void BinIteratorConstrained::collectConstraintVariables()
{
	// List variables required by constraints
	for (auto& constraint : m_constraints)
	{
		for (auto variable : constraint->getVariables())
		{
			m_consVariables.insert(variable);
		}
	}
}


const ConstraintManager& BinIteratorConstrained::getConstraintManager() const
{
	return *m_constraintManager.get();
}

const ProjectionPropertyManager&
    BinIteratorConstrained::getPropertyManager() const
{
	return *m_propManager.get();
}

void BinIteratorConstrained::collectInfo(
    bin_t bin, int tid, const ProjectionData& projData,
    ProjectionProperties& projProps, ConstraintParams& consInfo) const
{
	auto [d1, d2] = projData.getDetectorPair(bin);
	m_constraintManager->setDataValue(consInfo, tid, ConstraintVariable::Det1,
	                                  d1);
	m_constraintManager->setDataValue(consInfo, tid, ConstraintVariable::Det2,
	                                  d2);

	bool needsLOR =
		m_projVariables.find(ProjectionPropertyType::LOR) !=
	        m_projVariables.end() ||
	    m_consVariables.find(ConstraintVariable::AbsDeltaAngleDeg) !=
	        m_consVariables.end();
	Line3D lor;
	if (needsLOR)
	{
		lor = projData.getLOR(bin);
		m_propManager->setDataValue(projProps, tid,
		                            ProjectionPropertyType::LOR, lor);
	}
	const Scanner* scanner = &projData.getScanner();

	if (m_consVariables.find(ConstraintVariable::AbsDeltaAngleDeg) !=
	    m_consVariables.end())
	{
		// In-plane angle
		float a1 = std::atan2(lor.point1.y, lor.point1.x);
		float a2 = std::atan2(lor.point2.y, lor.point2.x);
		float diff = util::periodicDiff(a1, a2, (float)(2.f * PI));
		m_constraintManager->setDataValue(
		    consInfo, tid, ConstraintVariable::AbsDeltaAngleDeg, diff);
	}

	bool needsPlaneIdx =
	    m_consVariables.find(ConstraintVariable::AbsDeltaAngleIdx) !=
	        m_consVariables.end() ||
	    m_consVariables.find(ConstraintVariable::AbsDeltaBlockIdx) !=
	        m_consVariables.end();
	int d1xyi;
	int d2xyi;
	if (needsPlaneIdx)
	{
		d1xyi = d1 % scanner->detsPerRing;
		d2xyi = d2 % scanner->detsPerRing;
	}
	if (m_consVariables.find(ConstraintVariable::AbsDeltaAngleIdx) !=
	    m_consVariables.end())
	{
		int diff = util::periodicDiff(d1xyi, d2xyi,
		                              static_cast<int>(scanner->detsPerRing));
		m_constraintManager->setDataValue(
		    consInfo, tid, ConstraintVariable::AbsDeltaAngleIdx, diff);
	}

	bool needsPlaneBlock =
	    m_consVariables.find(ConstraintVariable::AbsDeltaBlockIdx) !=
	    m_consVariables.end();
	int d1bi;
	int d2bi;
	if (needsPlaneBlock)
	{
		d1bi = d1xyi / scanner->detsPerBlock;
		d2bi = d2xyi / scanner->detsPerBlock;
	}
	if (m_consVariables.find(ConstraintVariable::AbsDeltaBlockIdx) !=
	    m_consVariables.end())
	{
		int diff = util::periodicDiff(d1bi, d2bi,
		                              static_cast<int>(scanner->detsPerBlock));
		m_constraintManager->setDataValue(
		    consInfo, tid, ConstraintVariable::AbsDeltaBlockIdx, diff);
	}
}

bool BinIteratorConstrained::isValid(const ConstraintManager& manager,
                                     ConstraintParams& info) const
{
	for (auto& constraint : m_constraints)
	{
		if (!constraint->isValid(manager, info))
		{
			return false;
		}
	}
	return true;
}

}  // namespace yrt
