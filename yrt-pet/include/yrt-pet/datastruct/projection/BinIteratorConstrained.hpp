/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <functional>
#include <set>
#include <unordered_map>
#include <vector>

#include "yrt-pet/datastruct/projection/Constraints.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/utils/Types.hpp"

namespace yrt
{

class BinIteratorConstrained
{
public:
	BinIteratorConstrained(
	    const std::vector<Constraint*>& constraints,
	    const std::set<ProjectionPropertyType>& projProperties);

	void clearConstraints();

	void setupManagers();

	void collectConstraintVariables();
	void collectInfo(bin_t bin, size_t projIdx, int consIdx,
	                 const ProjectionData& projData,
	                 ProjectionProperties& projProps,
	                 ConstraintParams& consInfo) const;
	bool isValid(const ConstraintManager& manager,
	             ConstraintParams& info) const;

	const ConstraintManager& getConstraintManager() const;
	const ProjectionPropertyManager& getPropertyManager() const;
	const std::set<ProjectionPropertyType>& getProjPropertyTypes() const;

private:
	std::vector<Constraint*> m_constraints;

	// Variables for constraints, sensitivity image, reconstruction
	std::set<ConstraintVariable> m_consVariables;
	std::unique_ptr<ConstraintManager> m_constraintManager = nullptr;

	std::set<ProjectionPropertyType> m_projVariables;
	std::unique_ptr<ProjectionPropertyManager> m_propManager = nullptr;
};

}  // namespace yrt
