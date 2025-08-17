/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include <functional>

namespace yrt
{

class Scanner;

using ConstraintParams = char*;
using ConstraintManager = PropStructManager<ConstraintVariable>;

class Constraint
{
public:
	bool isValid(const ConstraintManager& manager,
	             ConstraintParams& info) const;
	virtual std::vector<ConstraintVariable> getVariables() const = 0;
protected:
	std::function<bool(const ConstraintManager&, ConstraintParams&)>
	    m_constraintFcn;
};

class ConstraintAngleDiffIndex : public Constraint
{
public:
	ConstraintAngleDiffIndex(int p_minAngleDiffIdx);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintAngleDiffDeg : public Constraint
{
public:
	ConstraintAngleDiffDeg(float p_minAngleDiffDeg);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintBlockDiffIndex : public Constraint
{
public:
	ConstraintBlockDiffIndex(int p_minBlockDiffIdx);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintDetectorMask : public Constraint
{
public:
	ConstraintDetectorMask(const Scanner* scanner);
	std::vector<ConstraintVariable> getVariables() const override;
};

}  // namespace yrt
