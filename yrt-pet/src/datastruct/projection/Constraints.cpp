/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/Constraints.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace yrt
{

void py_setup_constraints(py::module& m)
{
	auto c = py::class_<Constraint>(m, "Constraint");
	auto c_AngleDiffIndex = py::class_<ConstraintAngleDiffIndex, Constraint>(
	    m, "ConstraintAngleDiffIndex");
	c_AngleDiffIndex.def(py::init<int>(), py::arg("minAngleDiffIdx"));
	auto c_AngleDiffDeg = py::class_<ConstraintAngleDiffDeg, Constraint>(
	    m, "ConstraintAngleDiffDeg");
	c_AngleDiffDeg.def(py::init<float>(), py::arg("minAngleDiffDeg"));
	auto c_BlockDiffIndex = py::class_<ConstraintBlockDiffIndex, Constraint>(
	    m, "ConstraintBlockDiffIndex");
	c_BlockDiffIndex.def(py::init<int>(), py::arg("minBlockDiffIdx"));
	auto c_DetectorMask = py::class_<ConstraintDetectorMask, Constraint>(
	    m, "ConstraintDetectorMask");
	c_DetectorMask.def(py::init<const Scanner*>(), py::arg("scanner"));
}
}  // namespace yrt

#endif

namespace yrt
{

// Constraints

bool Constraint::isValid(const ConstraintManager& manager,
                         const PropertyUnit* info, size_t pos) const
{
	return m_constraintFcn(manager, info, pos);
}
std::vector<ConstraintVariableType> Constraint::getVariables() const
{
	return std::vector<ConstraintVariableType>();
}

// Minimum angle difference constraint (index)
ConstraintAngleDiffIndex::ConstraintAngleDiffIndex(int p_minAngleDiffIdx)
{
	m_constraintFcn = [p_minAngleDiffIdx](const ConstraintManager& manager,
	                                      const PropertyUnit* info,
	                                      size_t pos)
	{
		const int absDeltaAngleIdx = manager.getDataValue<int>(
		    info, pos, ConstraintVariableType::ABS_DELTA_ANGLE_IDX);
		return absDeltaAngleIdx >= p_minAngleDiffIdx;
	};
}
std::vector<ConstraintVariableType>
    ConstraintAngleDiffIndex::getVariables() const
{
	return {ConstraintVariableType::ABS_DELTA_ANGLE_IDX};
}

// Minimum angle difference constraint (angle)
ConstraintAngleDiffDeg::ConstraintAngleDiffDeg(float p_minAngleDiffDeg)
{
	m_constraintFcn = [p_minAngleDiffDeg](const ConstraintManager& manager,
	                                      const PropertyUnit* info,
	                                      size_t pos)
	{
		const float absDeltaAngleDeg = manager.getDataValue<float>(
		    info, pos, ConstraintVariableType::ABS_DELTA_ANGLE_DEG);
		return absDeltaAngleDeg >= p_minAngleDiffDeg;
	};
}
std::vector<ConstraintVariableType> ConstraintAngleDiffDeg::getVariables() const
{
	return {ConstraintVariableType::ABS_DELTA_ANGLE_DEG};
}

// Minimum angle difference constraint (index)
ConstraintBlockDiffIndex::ConstraintBlockDiffIndex(int p_minBlockDiffIdx)
{
	m_constraintFcn = [p_minBlockDiffIdx](const ConstraintManager& manager,
	                                      const PropertyUnit* info,
	                                      size_t pos)
	{
		const int absDeltaBlockIdx = manager.getDataValue<int>(
		    info, pos, ConstraintVariableType::ABS_DELTA_BLOCK_IDX);
		return absDeltaBlockIdx >= p_minBlockDiffIdx;
	};
}
std::vector<ConstraintVariableType>
    ConstraintBlockDiffIndex::getVariables() const
{
	return {ConstraintVariableType::ABS_DELTA_BLOCK_IDX};
}

// Detector mask
ConstraintDetectorMask::ConstraintDetectorMask(const Scanner* scanner)
{
	m_constraintFcn = [scanner](const ConstraintManager& manager,
	                            const PropertyUnit* info, size_t pos)
	{
		const det_id_t d1 = manager.getDataValue<det_id_t>(
		    info, pos, ConstraintVariableType::DET1);
		const det_id_t d2 = manager.getDataValue<det_id_t>(
		    info, pos, ConstraintVariableType::DET2);
		return (scanner->isDetectorAllowed(d1) &&
		        scanner->isDetectorAllowed(d2));
	};
}
std::vector<ConstraintVariableType> ConstraintDetectorMask::getVariables() const
{
	return {ConstraintVariableType::DET1, ConstraintVariableType::DET2};
}

}  // namespace yrt
