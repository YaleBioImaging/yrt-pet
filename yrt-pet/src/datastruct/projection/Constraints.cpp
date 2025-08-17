/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/Constraints.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"

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

}  // namespace yrt
