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
                         ConstraintParams& info) const
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
	m_constraintFcn = [p_minAngleDiffIdx](const ConstraintManager& manager,
	                                      ConstraintParams& info)
	{
		int absDeltaAngleIdx = manager.getDataValue<int>(
		    info, 0, ConstraintVariable::ABSDELTAANGLEIDX);
		return absDeltaAngleIdx >= p_minAngleDiffIdx;
	};
}
std::vector<ConstraintVariable> ConstraintAngleDiffIndex::getVariables() const
{
	return {ConstraintVariable::ABSDELTAANGLEIDX};
}

// Minimum angle difference constraint (angle)
ConstraintAngleDiffDeg::ConstraintAngleDiffDeg(float p_minAngleDiffDeg)
{
	m_constraintFcn = [p_minAngleDiffDeg](const ConstraintManager& manager,
	                                      ConstraintParams& info)
	{
		float absDeltaAngleDeg = manager.getDataValue<float>(
		    info, 0, ConstraintVariable::ABSDELTAANGLEDEG);
		return absDeltaAngleDeg >= p_minAngleDiffDeg;
	};
}
std::vector<ConstraintVariable> ConstraintAngleDiffDeg::getVariables() const
{
	return {ConstraintVariable::ABSDELTAANGLEDEG};
}

// Minimum angle difference constraint (index)
ConstraintBlockDiffIndex::ConstraintBlockDiffIndex(int p_minBlockDiffIdx)
{
	m_constraintFcn = [p_minBlockDiffIdx](const ConstraintManager& manager,
	                                      ConstraintParams& info)
	{
		int absDeltaBlockIdx = manager.getDataValue<int>(
		    info, 0, ConstraintVariable::ABSDELTABLOCKIDX);
		return absDeltaBlockIdx >= p_minBlockDiffIdx;
	};
}
std::vector<ConstraintVariable> ConstraintBlockDiffIndex::getVariables() const
{
	return {ConstraintVariable::ABSDELTABLOCKIDX};
}

// Detector mask
ConstraintDetectorMask::ConstraintDetectorMask(const Scanner* scanner)
{
	m_constraintFcn =
	    [scanner](const ConstraintManager& manager, ConstraintParams& info)
	{
		det_id_t d1 =
		    manager.getDataValue<det_id_t>(info, 0, ConstraintVariable::DET1);
		det_id_t d2 =
		    manager.getDataValue<det_id_t>(info, 0, ConstraintVariable::DET2);
		return (scanner->isDetectorAllowed(d1) &&
		        scanner->isDetectorAllowed(d2));
	};
}
std::vector<ConstraintVariable> ConstraintDetectorMask::getVariables() const
{
	return {ConstraintVariable::DET1, ConstraintVariable::DET2};
}

}  // namespace yrt
