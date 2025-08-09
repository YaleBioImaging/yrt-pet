/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <functional>
#include <set>
#include <unordered_map>
#include <vector>

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
	BinIteratorConstrained();

	template <typename C, typename... Args>
	void addConstraint(Args... args);
	void clearConstraints();

	void addProjVariable(ProjectionPropertyType prop);
	void addProjVariableSens(ProjectionPropertyType prop);
	void addProjVariableRecon(ProjectionPropertyType prop);
	void setupManagers();

	std::set<ConstraintVariable> collectConstraintVariables();
	void collectInfoSens(bin_t bin, int tid, const ProjectionData& projData,
	                     ProjectionProperties& projProps,
	                     ConstraintParams& consInfo) const;
	void collectInfoRecon(bin_t bin, int tid,
	                      const ProjectionData& projData,
	                      ProjectionProperties& projProps,
	                      ConstraintParams& consInfo) const;
	bool isValid(ConstraintParams& info) const;

	const ConstraintManager& getConstraintManager() const;
	const ProjectionPropertyManager& getPropertyManagerSens() const;
	const ProjectionPropertyManager& getPropertyManagerRecon() const;

private:
	const BinIterator* m_binIterBase;
	std::vector<std::unique_ptr<Constraint>> m_constraints;

	// Variables for constraints, sensitivity image, reconstruction
	std::set<ConstraintVariable> m_consVariables;
	std::unique_ptr<ConstraintManager> m_constraintManager = nullptr;

	std::set<ProjectionPropertyType> m_projVariablesSens;
	std::unique_ptr<ProjectionPropertyManager> m_propManagerSens = nullptr;

	std::set<ProjectionPropertyType> m_projVariablesRecon;
	std::unique_ptr<ProjectionPropertyManager> m_propManagerRecon = nullptr;

	void collectInfo(bin_t bin, int tid, const ProjectionData& projData,
	                 ProjectionPropertyManager& projPropManager,
	                 const std::set<ProjectionPropertyType>& projVariables,
	                 ProjectionProperties& projProps,
	                 ConstraintParams& consInfo) const;
};

}  // namespace yrt
