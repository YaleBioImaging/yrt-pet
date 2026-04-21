/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/datastruct/scanner/DetectorMask.hpp"

#include <functional>

namespace yrt
{

class Scanner;

using ConstraintManager = PropStructManager<ConstraintVariableType>;

class Constraint
{
public:
	bool isValid(const ConstraintManager& manager,
	             const PropertyUnit* info, size_t pos) const;
	virtual std::vector<ConstraintVariableType> getVariables() const = 0;
	virtual ~Constraint() = default;

protected:
	std::function<bool(const ConstraintManager&, const PropertyUnit*,
	                   size_t)>
	    m_constraintFcn;
};

class ConstraintAngleDiffIndex : public Constraint
{
public:
	explicit ConstraintAngleDiffIndex(int p_minAngleDiffIdx);
	std::vector<ConstraintVariableType> getVariables() const override;
};
class ConstraintAngleDiffDeg : public Constraint
{
public:
	explicit ConstraintAngleDiffDeg(float p_minAngleDiffDeg);
	std::vector<ConstraintVariableType> getVariables() const override;
};
class ConstraintBlockDiffIndex : public Constraint
{
public:
	explicit ConstraintBlockDiffIndex(int p_minBlockDiffIdx);
	std::vector<ConstraintVariableType> getVariables() const override;
};
class ConstraintDetectorMask : public Constraint
{
public:
	explicit ConstraintDetectorMask(const Scanner* scanner);
	explicit ConstraintDetectorMask(const DetectorMask* mask);
	std::vector<ConstraintVariableType> getVariables() const override;
};

}  // namespace yrt
