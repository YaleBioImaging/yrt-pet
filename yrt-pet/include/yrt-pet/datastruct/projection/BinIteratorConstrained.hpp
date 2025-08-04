/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <unordered_map>
#include <vector>
#include <set>
#include <functional>

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/utils/Types.hpp"

namespace yrt
{

using ConstraintParams = char*;
using ConstraintManager = PropStructManager<ConstraintVariable>;

class Constraint
{
public:
	bool isValid(ConstraintParams& info) const;
	virtual std::vector<ConstraintVariable> getVariables() const = 0;
protected:
	std::function<bool(ConstraintParams&)> m_constraintFcn;
};

class ConstraintAngleDiffIndex : public Constraint
{
public:
	ConstraintAngleDiffIndex(const ConstraintManager& p_manager,
	                         int p_minAngleDiffIdx);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintAngleDiffDeg : public Constraint
{
public:
	ConstraintAngleDiffDeg(const ConstraintManager& p_manager,
	                       float p_minAngleDiffDeg);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintBlockDiffIndex : public Constraint
{
public:
	ConstraintBlockDiffIndex(const ConstraintManager& p_manager,
	                         int p_minBlockDiffIdx);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintDetectorMask : public Constraint
{
public:
	ConstraintDetectorMask(const ConstraintManager& p_manager,
	                       const Scanner* scanner);
	std::vector<ConstraintVariable> getVariables() const override;
};

class BinIteratorConstrained
{
public:
	BinIteratorConstrained(const ProjectionData* p_projData,
	                       std::vector<const Constraint*> p_constraints);

	std::set<ConstraintVariable> collectVariables() const;
	void collectInfo(
	    bin_t bin, std::set<ConstraintVariable>& consVariables,
	    std::set<ProjectionPropertyType>& projVariables,
	    char*& projProps,
	    ConstraintParams& consInfo) const;
	bool isValid(ConstraintParams& info) const;

private:
	const ProjectionData* m_projData;
	const BinIterator* m_binIterBase;
	std::vector<const Constraint*> m_constraints;

	// Loop variables
	std::set<ConstraintVariable> m_variables;
	std::unique_ptr<ConstraintManager> m_constraintManager;
	std::unique_ptr<ProjectionPropertyManager> m_propManager;
};

}  // namespace yrt
