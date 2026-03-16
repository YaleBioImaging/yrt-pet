/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <bitset>
#include <set>
#include <vector>

#include "yrt-pet/datastruct/projection/Constraints.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/utils/Types.hpp"

namespace yrt
{

class BinFilter
{
public:
	enum class CollectInfoFlag
	{
		PlaneIdx = 0,
		PlaneBlock,
		DetPair,
		DetID,
		Det1,
		Det2,
		LOR,
		AbsDeltaAngleDeg,
		AbsDeltaAngleIdx,
		AbsDeltaBlockIdx,
		COUNT
	};
	using CollectInfoFlags =
	    std::bitset<static_cast<size_t>(CollectInfoFlag::COUNT)>;

	virtual ~BinFilter() = default;

	virtual void clearConstraints();

	void collectConstraintVariables();
	void collectFlags(CollectInfoFlags& collectFlags) const;
	void collectInfo(bin_t bin, size_t projIdx, int consIdx,
	                 const ProjectionData& projData,
	                 const CollectInfoFlags& collectFlags,
	                 PropertyUnit* projProps, PropertyUnit* consInfo) const;
	bool isValid(const ConstraintManager& manager, const PropertyUnit* consInfo,
	             size_t pos) const;

	const ConstraintManager& getConstraintManager() const;
	const ProjectionPropertyManager& getPropertyManager() const;
	const std::set<ProjectionPropertyType>& getProjPropertyTypes() const;
	size_t getProjectionPropertiesElementSize() const;

protected:
	BinFilter(const std::vector<Constraint*>& constraints,
	          const std::set<ProjectionPropertyType>& projProperties);

	std::vector<Constraint*> m_constraints;

	// Variables for constraints and pr
	std::set<ConstraintVariableType> m_consVariables;
	std::set<ProjectionPropertyType> m_projVariables;

	const ConstraintManager* mp_constraintManagerPtr;
	const ProjectionPropertyManager* mp_propManagerPtr;
};

class BinFilterOwned : public BinFilter
{
public:
	BinFilterOwned(const std::vector<Constraint*>& constraints,
	               const std::set<ProjectionPropertyType>& projProperties);

protected:
	std::unique_ptr<ConstraintManager> mp_constraintManager;
	std::unique_ptr<ProjectionPropertyManager> mp_propManager;
};

}  // namespace yrt
