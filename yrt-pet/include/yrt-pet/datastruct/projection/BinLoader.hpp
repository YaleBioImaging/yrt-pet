/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/BinFilter.hpp"

namespace yrt
{

class BinLoader : public BinFilter
{
public:
	BinLoader(const std::vector<Constraint*>& constraints,
	          const std::set<ProjectionPropertyType>& projProperties);

	// Allocate for a given number of projection-space properties
	void allocate(size_t numElementsProperties);
	bool isAllocated() const;

	void clearConstraints() override;

	// This will iterate on all the bins (using the bin iterator) and gather the
	//  projection properties
	void collectFromBins(const ProjectionData& projData,
	                     const BinIterator& binIter, bool memsetOnInvalid);
	template <bool PrintProgress = false>
	void parallelDoOnBins(
	    const ProjectionData& projData, const BinIterator& binIter,
	    const std::function<void(const ProjectionPropertyManager& propManager,
	                             PropertyUnit* propStruct, size_t pos,
	                             bin_t bin)>& funcIfValid);

	bool verifyConstraints(size_t pos) const;

	PropertyUnit* getProjectionPropertiesRawPointer() const;
	PropertyUnit* getConstraintVariablesRawPointer() const;

	void collectInfo(bin_t bin, size_t projIdx, int consIdx,
	                 const ProjectionData& projData,
	                 const CollectInfoFlags& collectFlags) const;

	// Return the number of elements (or "rows") in the projection properties
	//  structure
	size_t getProjectionPropertiesSize() const;
	PropStruct<ConstraintVariableType>* getConstraintPropStruct() const;
	PropStruct<ProjectionPropertyType>* getProjectionPropStruct() const;

private:
	void setupStructs();

	std::unique_ptr<PropStruct<ConstraintVariableType>> mp_constraintStruct;
	std::unique_ptr<PropStruct<ProjectionPropertyType>> mp_propStruct;
};

}  // namespace yrt
