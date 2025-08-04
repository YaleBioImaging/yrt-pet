/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/utils/Types.hpp"

#include <map>
#include <memory>
#include <ostream>
#include <string>

namespace yrt
{

enum class ProjectionPropertyType
{
	DET_ID = 0,
	LOR,
	ORIENT,
	TOF,
	ADD_CORR,
	MULT_CORR,
	FRAME,
	COUNT
};

inline std::map<ProjectionPropertyType, std::pair<std::string, int>>
    ProjectionPropertiesInfo{
        {ProjectionPropertyType::DET_ID, {"DET_ID", sizeof(det_pair_t)}},
        {ProjectionPropertyType::LOR, {"LOR", sizeof(Line3D)}},
        {ProjectionPropertyType::ORIENT, {"ORIENT", sizeof(det_orient_t)}},
        {ProjectionPropertyType::TOF, {"TOF", sizeof(float)}},
        {ProjectionPropertyType::ADD_CORR, {"ADD_CORR", sizeof(float)}},
        {ProjectionPropertyType::MULT_CORR, {"MULT_CORR", sizeof(float)}},
        {ProjectionPropertyType::FRAME, {"FRAME", sizeof(frame_t)}}};

class ProjectionPropertiesManager
{
public:
	ProjectionPropertiesManager(std::set<ProjectionPropertyType>& props);

	// Helper functions
	std::unique_ptr<char[]> createDataArray(size_t numElements) const;
	template <typename T>
	T* getDataPtr(char* data, int idx, ProjectionPropertyType prop) const;
	template <typename T>
	void setDataValue(char* data, int idx, ProjectionPropertyType prop,
	                  T& value) const;
	template <typename T>
	T& getDataValue(char* data, int idx, ProjectionPropertyType prop) const;

	// Accessors
	unsigned int getElementSize() const;
	int getTypeID() const;
	unsigned int getOffset(ProjectionPropertyType prop) const;

private:
	// Data information
	// ----------------

	// Bit mask with flag for each allowed variable
	int type;
	// Total element size
	unsigned int elementSize;
	// Offset in raw pointer for each included prop
	std::unordered_map<ProjectionPropertyType, unsigned int> offsetMap;
};

std::ostream& operator<<(std::ostream& oss,
                         const ProjectionPropertiesManager& t);

}  // namespace yrt
