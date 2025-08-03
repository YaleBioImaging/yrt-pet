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

enum class ProjectionPropertiesList
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

inline std::map<ProjectionPropertiesList, std::pair<std::string, int>>
    ProjectionPropertiesInfo{
        {ProjectionPropertiesList::DET_ID, {"DET_ID", sizeof(det_pair_t)}},
        {ProjectionPropertiesList::LOR, {"LOR", sizeof(Line3D)}},
        {ProjectionPropertiesList::ORIENT, {"ORIENT", sizeof(det_orient_t)}},
        {ProjectionPropertiesList::TOF, {"TOF", sizeof(float)}},
        {ProjectionPropertiesList::ADD_CORR, {"ADD_CORR", sizeof(float)}},
        {ProjectionPropertiesList::MULT_CORR, {"MULT_CORR", sizeof(float)}},
        {ProjectionPropertiesList::FRAME, {"FRAME", sizeof(frame_t)}}};

class ProjectionPropertiesManager
{
public:
	ProjectionPropertiesManager(std::set<ProjectionPropertiesList>& props);

	// Helper functions
	std::unique_ptr<char[]> createDataArray(size_t numElements) const;
	template <typename T>
	T* getDataPtr(char* data, int idx, ProjectionPropertiesList prop) const;
	template <typename T>
	void setDataValue(char* data, int idx, ProjectionPropertiesList prop,
	                  T& value) const;
	template <typename T>
	T& getDataValue(char* data, int idx, ProjectionPropertiesList prop) const;

	// Accessors
	unsigned int getElementSize() const;
	int getTypeID() const;
	unsigned int getOffset(ProjectionPropertiesList prop) const;

private:
	// Data information
	// ----------------

	// Bit mask with flag for each allowed variable
	int type;
	// Total element size
	unsigned int elementSize;
	// Offset in raw pointer for each included prop
	std::unordered_map<ProjectionPropertiesList, unsigned int> offsetMap;
};

std::ostream& operator<<(std::ostream& oss,
                         const ProjectionPropertiesManager& t);

}  // namespace yrt
