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

// Constrained bin iterator
BinIteratorConstrained::BinIteratorConstrained(
    const std::vector<Constraint*>& constraints,
    const std::set<ProjectionPropertyType>& projVariables)
    : m_constraints(constraints), m_projVariables(projVariables)
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

const std::set<ProjectionPropertyType>&
    BinIteratorConstrained::getProjPropertyTypes() const
{
	return m_projVariables;
}

void BinIteratorConstrained::setupManagers()
{
	if (m_constraints.size() > 0)
	{
		collectConstraintVariables();
	}
	m_constraintManager = std::make_unique<ConstraintManager>(m_consVariables);
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

void BinIteratorConstrained::collectInfo(bin_t bin, size_t projIdx, int consIdx,
                                         const ProjectionData& projData,
                                         ProjectionProperties& projProps,
                                         ConstraintParams& consInfo) const
{
	det_pair_t detPair;

	bool needsPlaneIdx =
	    m_consVariables.find(ConstraintVariable::ABS_DELTA_ANGLE_IDX) !=
	        m_consVariables.end() ||
	    m_consVariables.find(ConstraintVariable::ABS_DELTA_BLOCK_IDX) !=
	        m_consVariables.end();
	bool needsPlaneBlock =
	    m_consVariables.find(ConstraintVariable::ABS_DELTA_BLOCK_IDX) !=
	    m_consVariables.end();
	bool needsDetPair = m_projVariables.find(ProjectionPropertyType::DET_ID) !=
	                        m_projVariables.end() ||
	                    needsPlaneIdx || needsPlaneBlock;

	if (needsDetPair)
	{
		detPair = projData.getDetectorPair(bin);
	}
	if (m_projVariables.find(ProjectionPropertyType::DET_ID) !=
	    m_projVariables.end())
	{
		m_propManager->setDataValue(projProps, projIdx,
		                            ProjectionPropertyType::DET_ID, detPair);
	}

	bool needsLOR =
	    m_projVariables.find(ProjectionPropertyType::LOR) !=
	        m_projVariables.end() ||
	    m_consVariables.find(ConstraintVariable::ABS_DELTA_ANGLE_DEG) !=
	        m_consVariables.end();
	Line3D lor;
	if (needsLOR)
	{
		lor = projData.getLOR(bin);
		m_propManager->setDataValue(projProps, projIdx,
		                            ProjectionPropertyType::LOR, lor);
	}
	const Scanner* scanner = &projData.getScanner();

	if (m_consVariables.find(ConstraintVariable::ABS_DELTA_ANGLE_DEG) !=
	    m_consVariables.end())
	{
		// In-plane angle
		float a1 = std::atan2(lor.point1.y, lor.point1.x);
		float a2 = std::atan2(lor.point2.y, lor.point2.x);
		float diff = util::periodicDiff(a1, a2, (float)(2.f * PI));
		m_constraintManager->setDataValue(
		    consInfo, consIdx, ConstraintVariable::ABS_DELTA_ANGLE_DEG, diff);
	}

	int d1xyi;
	int d2xyi;
	if (needsPlaneIdx)
	{
		d1xyi = detPair.d1 % scanner->detsPerRing;
		d2xyi = detPair.d2 % scanner->detsPerRing;
	}
	if (m_consVariables.find(ConstraintVariable::ABS_DELTA_ANGLE_IDX) !=
	    m_consVariables.end())
	{
		int diff = util::periodicDiff(d1xyi, d2xyi,
		                              static_cast<int>(scanner->detsPerRing));
		m_constraintManager->setDataValue(
		    consInfo, consIdx, ConstraintVariable::ABS_DELTA_ANGLE_IDX, diff);
	}

	int d1bi;
	int d2bi;
	if (needsPlaneBlock)
	{
		d1bi = d1xyi / scanner->detsPerBlock;
		d2bi = d2xyi / scanner->detsPerBlock;
	}
	if (m_consVariables.find(ConstraintVariable::ABS_DELTA_BLOCK_IDX) !=
	    m_consVariables.end())
	{
		int diff = util::periodicDiff(d1bi, d2bi,
		                              static_cast<int>(scanner->detsPerBlock));
		m_constraintManager->setDataValue(
		    consInfo, consIdx, ConstraintVariable::ABS_DELTA_BLOCK_IDX, diff);
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
