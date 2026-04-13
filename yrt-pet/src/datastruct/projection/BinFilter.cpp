/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/BinFilter.hpp"

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/utils/Tools.hpp"
#include "yrt-pet/utils/Types.hpp"

#include <cmath>
#include <memory>

#if BUILD_PYBIND11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_binfilter(py::module& m)
{
	auto c = py::class_<BinFilter>(m, "BinFilter");
	c.def("clearConstraints", &BinFilter::clearConstraints);
	c.def("collectConstraintVariables", &BinFilter::collectConstraintVariables);

	c.def("getConstraintManager", &BinFilter::getConstraintManager);
	c.def("getPropertyManager", &BinFilter::getPropertyManager);
	c.def("getProjPropertyTypes", &BinFilter::getProjPropertyTypes);
}
}  // namespace yrt

#endif

namespace yrt
{

// Bin filter/extractor
BinFilter::BinFilter(const std::vector<Constraint*>& constraints,
                     const std::set<ProjectionPropertyType>& projVariables)
    : m_constraints(constraints),
      m_projVariables(projVariables),
      mp_constraintManagerPtr(nullptr),
      mp_propManagerPtr(nullptr)
{
	if (!m_constraints.empty())
	{
		collectConstraintVariables();
	}
}

void BinFilter::clearConstraints()
{
	m_constraints.clear();
	m_consVariables.clear();
}

void BinFilter::collectConstraintVariables()
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

void BinFilter::collectFlags(CollectInfoFlags& collectFlags) const
{
	collectFlags[static_cast<size_t>(CollectInfoFlag::PlaneIdx)] =
	    m_consVariables.find(ConstraintVariableType::ABS_DELTA_ANGLE_IDX) !=
	        m_consVariables.end() ||
	    m_consVariables.find(ConstraintVariableType::ABS_DELTA_BLOCK_IDX) !=
	        m_consVariables.end();
	collectFlags[static_cast<size_t>(CollectInfoFlag::PlaneBlock)] =
	    m_consVariables.find(ConstraintVariableType::ABS_DELTA_BLOCK_IDX) !=
	    m_consVariables.end();
	collectFlags[static_cast<size_t>(CollectInfoFlag::DetID)] =
	    m_projVariables.find(ProjectionPropertyType::DET_ID) !=
	    m_projVariables.end();
	collectFlags[static_cast<size_t>(CollectInfoFlag::Det1)] =
	    m_consVariables.find(ConstraintVariableType::DET1) !=
	    m_consVariables.end();
	collectFlags[static_cast<size_t>(CollectInfoFlag::Det2)] =
	    m_consVariables.find(ConstraintVariableType::DET2) != m_consVariables.end();
	collectFlags[static_cast<size_t>(CollectInfoFlag::DetPair)] =
	    collectFlags[static_cast<size_t>(CollectInfoFlag::DetID)] ||
	    collectFlags[static_cast<size_t>(CollectInfoFlag::Det1)] ||
	    collectFlags[static_cast<size_t>(CollectInfoFlag::Det2)] ||
	    collectFlags[static_cast<size_t>(CollectInfoFlag::PlaneIdx)] ||
	    collectFlags[static_cast<size_t>(CollectInfoFlag::PlaneBlock)];
	collectFlags[static_cast<size_t>(CollectInfoFlag::LOR)] =
	    m_projVariables.find(ProjectionPropertyType::LOR) !=
	        m_projVariables.end() ||
	    m_consVariables.find(ConstraintVariableType::ABS_DELTA_ANGLE_DEG) !=
	        m_consVariables.end();
	collectFlags[static_cast<size_t>(CollectInfoFlag::AbsDeltaAngleDeg)] =
	    m_consVariables.find(ConstraintVariableType::ABS_DELTA_ANGLE_DEG) !=
	    m_consVariables.end();
	collectFlags[static_cast<size_t>(CollectInfoFlag::AbsDeltaAngleIdx)] =
	    m_consVariables.find(ConstraintVariableType::ABS_DELTA_ANGLE_IDX) !=
	    m_consVariables.end();
	collectFlags[static_cast<size_t>(CollectInfoFlag::AbsDeltaBlockIdx)] =
	    m_consVariables.find(ConstraintVariableType::ABS_DELTA_BLOCK_IDX) !=
	    m_consVariables.end();
}

void BinFilter::collectInfo(bin_t bin, size_t projIdx, int consIdx,
                            const ProjectionData& projData,
                            const CollectInfoFlags& collectFlags,
                            PropertyUnit* projProps,
                            PropertyUnit* consInfo) const
{
	det_pair_t detPair;
	if (collectFlags[static_cast<size_t>(CollectInfoFlag::DetPair)])
	{
		detPair = projData.getDetectorPair(bin);
	}
	if (collectFlags[static_cast<size_t>(CollectInfoFlag::DetID)])
	{
		mp_propManagerPtr->setDataValue(
		    projProps, projIdx, ProjectionPropertyType::DET_ID, detPair);
	}
	if (collectFlags[static_cast<size_t>(CollectInfoFlag::Det1)])
	{
		mp_constraintManagerPtr->setDataValue(
		    consInfo, consIdx, ConstraintVariableType::DET1, detPair.d1);
	}
	if (collectFlags[static_cast<size_t>(CollectInfoFlag::Det2)])
	{
		mp_constraintManagerPtr->setDataValue(
		    consInfo, consIdx, ConstraintVariableType::DET2, detPair.d2);
	}

	Line3D lor;
	if (collectFlags[static_cast<size_t>(CollectInfoFlag::LOR)])
	{
		lor = projData.getLOR(bin);
		mp_propManagerPtr->setDataValue(projProps, projIdx,
		                                ProjectionPropertyType::LOR, lor);
	}

	const Scanner* scanner = &projData.getScanner();

	if (collectFlags[static_cast<size_t>(CollectInfoFlag::AbsDeltaAngleDeg)])
	{
		// In-plane angle
		float a1 = std::atan2(lor.point1.y, lor.point1.x);
		float a2 = std::atan2(lor.point2.y, lor.point2.x);
		float diff = util::periodicDiff(a1, a2, (float)(2.f * PI)) / PI * 180.f;
		mp_constraintManagerPtr->setDataValue(
		    consInfo, consIdx, ConstraintVariableType::ABS_DELTA_ANGLE_DEG,
		    diff);
	}

	int d1xyi;
	int d2xyi;
	if (collectFlags[static_cast<size_t>(CollectInfoFlag::PlaneIdx)])
	{
		d1xyi = detPair.d1 % scanner->detsPerRing;
		d2xyi = detPair.d2 % scanner->detsPerRing;
	}
	if (collectFlags[static_cast<size_t>(CollectInfoFlag::AbsDeltaAngleIdx)])
	{
		int diff = std::abs(util::periodicDiff(
		    d1xyi, d2xyi, static_cast<int>(scanner->detsPerRing)));
		mp_constraintManagerPtr->setDataValue(
		    consInfo, consIdx, ConstraintVariableType::ABS_DELTA_ANGLE_IDX,
		    diff);
	}

	int d1bi;
	int d2bi;
	if (collectFlags[static_cast<size_t>(CollectInfoFlag::PlaneBlock)])
	{
		d1bi = d1xyi / scanner->detsPerBlock;
		d2bi = d2xyi / scanner->detsPerBlock;
	}
	if (collectFlags[static_cast<size_t>(CollectInfoFlag::AbsDeltaBlockIdx)])
	{
		size_t numBlocks = scanner->detsPerRing / scanner->detsPerBlock;
		int diff = std::abs(
		    util::periodicDiff(d1bi, d2bi, static_cast<int>(numBlocks)));
		mp_constraintManagerPtr->setDataValue(
		    consInfo, consIdx, ConstraintVariableType::ABS_DELTA_BLOCK_IDX,
		    diff);
	}
}

bool BinFilter::isValid(const ConstraintManager& manager,
                        const PropertyUnit* consInfo, size_t pos) const
{
	for (auto& constraint : m_constraints)
	{
		if (!constraint->isValid(manager, consInfo, pos))
		{
			return false;
		}
	}
	return true;
}

const ConstraintManager& BinFilter::getConstraintManager() const
{
	return *mp_constraintManagerPtr;
}

const ProjectionPropertyManager& BinFilter::getPropertyManager() const
{
	return *mp_propManagerPtr;
}

const std::set<ProjectionPropertyType>& BinFilter::getProjPropertyTypes() const
{
	return m_projVariables;
}

size_t BinFilter::getProjectionPropertiesElementSize() const
{
	return mp_propManagerPtr->getElementSize();
}

BinFilterOwned::BinFilterOwned(
    const std::vector<Constraint*>& constraints,
    const std::set<ProjectionPropertyType>& projProperties)
    : BinFilter(constraints, projProperties)
{
	mp_constraintManager = std::make_unique<ConstraintManager>(m_consVariables);
	mp_propManager =
	    std::make_unique<ProjectionPropertyManager>(m_projVariables);
	mp_constraintManagerPtr = mp_constraintManager.get();
	mp_propManagerPtr = mp_propManager.get();
}

}  // namespace yrt
