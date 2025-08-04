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

enum class ConstraintVariable
{
	Det1,
	Det2,
	AbsDeltaAngleDeg,
	AbsDeltaAngleIdx,
	AbsDeltaBlockIdx
};

inline std::map<ConstraintVariable, std::pair<std::string, int>>
    ConstraintVariableInfo{
        {ConstraintVariable::Det1, {"Det1", sizeof(det_id_t)}},
        {ConstraintVariable::Det2, {"Det2", sizeof(det_id_t)}},
        {ConstraintVariable::AbsDeltaAngleDeg,
         {"AbsDeltaAngleDeg", sizeof(float)}},
        {ConstraintVariable::AbsDeltaAngleIdx,
         {"AbsDeltaAngleIdx", sizeof(int)}},
        {ConstraintVariable::AbsDeltaBlockIdx,
         {"AbsDeltaBlockIdx", sizeof(int)}}};

using ConstraintParams = char*;

class Constraint
{
public:
	bool isValid(ConstraintParams& info) const;
	virtual std::vector<ConstraintVariable> getVariables() const = 0;
protected:
	std::function<bool(ConstraintParams&)> mConstraintFcn;
};

class ConstraintAngleDiffIndex : public Constraint
{
public:
	ConstraintAngleDiffIndex(size_t pMinAngleDiffIdx);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintAngleDiffDeg : public Constraint
{
public:
	ConstraintAngleDiffDeg(size_t pMinAngleDiffDeg);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintBlockDiffIndex : public Constraint
{
public:
	ConstraintBlockDiffIndex(size_t pMinBlockDiffIdx);
	std::vector<ConstraintVariable> getVariables() const override;
};
class ConstraintDetectorMask : public Constraint
{
public:
	ConstraintDetectorMask(const Scanner* scanner);
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
	    ProjectionProperties& projProps,
	    ConstraintParams& consInfo) const;
	bool isValid(ConstraintParams& info) const;

private:
	const ProjectionData* m_projData;
	const BinIterator* m_binIterBase;
	std::vector<const Constraint*> m_constraints;

	// Loop variables
	std::set<ConstraintVariable> m_variables;

};

}  // namespace yrt
