/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Types.hpp"

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>

namespace yrt
{

enum class ProjectionPropertyType
{
	DetID = 0,
	LOR,
	DetOrient,
	TOF,
	EventFrame,
	COUNT
};

enum class ConstraintVariable
{
	Det1,
	Det2,
	AbsDeltaAngleDeg,
	AbsDeltaAngleIdx,
	AbsDeltaBlockIdx,
	COUNT
};

inline std::map<ConstraintVariable, std::pair<std::string, int>>
    constraintVariableInfo{
        {ConstraintVariable::Det1, {"Det1", sizeof(det_id_t)}},
        {ConstraintVariable::Det2, {"Det2", sizeof(det_id_t)}},
        {ConstraintVariable::AbsDeltaAngleDeg,
         {"AbsDeltaAngleDeg", sizeof(float)}},
        {ConstraintVariable::AbsDeltaAngleIdx,
         {"AbsDeltaAngleIdx", sizeof(int)}},
        {ConstraintVariable::AbsDeltaBlockIdx,
         {"AbsDeltaBlockIdx", sizeof(int)}}};


template <typename Enum>
class PropStructManager
{
public:
	PropStructManager(std::set<Enum>& props);

	// Helper functions
	std::unique_ptr<char[]> createDataArray(size_t numElements) const;
	template <typename T>
	T* getDataPtr(char* data, int idx, Enum prop) const;
	template <typename T>
	void setDataValue(char* data, int idx, Enum prop, T& value) const;
	template <typename T>
	T& getDataValue(char* data, int idx, Enum prop) const;

	// Accessors
	unsigned int getElementSize() const;
	int getTypeID() const;
	unsigned int getOffset(Enum prop) const;
	bool has(Enum prop) const;

	// Offset table
	std::map<Enum, std::pair<std::string, int>> getInfo() const;

private:
	// Data information
	// ----------------

	// Bit mask with flag for each allowed variable
	int type;
	// Total element size
	unsigned int elementSize;
	// Offset in raw pointer for each included prop
	//std::array<int, static_cast<size_t>(Enum::COUNT)> offsetMap;
	std::vector<int> offsetMap;
};

using ProjectionProperties = char*;
using ProjectionPropertyManager = PropStructManager<ProjectionPropertyType>;

template <typename Enum>
std::ostream& operator<<(std::ostream& oss, const PropStructManager<Enum>& t);

}  // namespace yrt
