/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/GPUUtils.cuh"
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
	Det1 = 0,
	Det2,
	AbsDeltaAngleDeg,
	AbsDeltaAngleIdx,
	AbsDeltaBlockIdx,
	COUNT
};

template <typename Enum>
class PropStructManager
{
public:
	PropStructManager(std::set<Enum>& props);

	// Helper functions
	std::unique_ptr<char[]> createDataArray(size_t numElements) const;
	template <typename T>
	HOST_DEVICE_CALLABLE inline T* getDataPtr(char* data, size_t idx,
	                                          Enum prop) const
	{
		return reinterpret_cast<T*>(data + elementSize * idx +
		                            offsetMap[static_cast<int>(prop)]);
	}
	template <typename T>
	HOST_DEVICE_CALLABLE inline const T* getDataPtr(const char* data,
	                                                size_t idx, Enum prop) const
	{
		return reinterpret_cast<const T*>(data + elementSize * idx +
		                                  offsetMap[static_cast<int>(prop)]);
	}

	template <typename T>
	HOST_DEVICE_CALLABLE inline void
	    setDataValue(char* data, size_t idx, Enum prop, const T& value) const
	{
		T* ptr = getDataPtr<T>(data, idx, prop);
		*ptr = value;
	}
	template <typename T>
	HOST_DEVICE_CALLABLE inline const T&
	    getDataValue(const char* data, size_t idx, Enum prop) const
	{
		const T* ptr = getDataPtr<T>(data, idx, prop);
		return *ptr;
	}

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
	int offsetMap[static_cast<size_t>(Enum::COUNT)];
};

using ProjectionProperties = char*;
using ProjectionPropertyManager = PropStructManager<ProjectionPropertyType>;

template <typename Enum>
std::ostream& operator<<(std::ostream& oss, const PropStructManager<Enum>& t)
{
	const auto& info = t.getInfo();
	oss << "[";
	for (int i = 0; i < static_cast<int>(Enum::COUNT); i++)
	{
		if (t.getTypeID() & (1 << i))
		{
			auto var = static_cast<Enum>(i);
			oss << info.at(var).first << ": " << info.at(var).second
			    << " (offset: " << t.getOffset(var) << "), ";
		}
	}
	oss << "]";
	return oss;
}

}  // namespace yrt
