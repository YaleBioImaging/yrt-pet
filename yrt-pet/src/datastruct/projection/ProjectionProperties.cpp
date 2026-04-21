/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/geometry/Line3D.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace yrt
{

void py_setup_projectionpropertytype(pybind11::module& m)
{
	pybind11::enum_<ProjectionPropertyType>(m, "ProjectionPropertyType")
	    .value("DET_ID", ProjectionPropertyType::DET_ID)
	    .value("TIMESTAMP", ProjectionPropertyType::TIMESTAMP)
	    .value("LOR", ProjectionPropertyType::LOR)
	    .value("DET_ORIENT", ProjectionPropertyType::DET_ORIENT)
	    .value("MEASUREMENT", ProjectionPropertyType::MEASUREMENT)
	    .value("TOF", ProjectionPropertyType::TOF)
	    .value("DYNAMIC_FRAME", ProjectionPropertyType::DYNAMIC_FRAME)
	    .value("SENSITIVITY", ProjectionPropertyType::SENSITIVITY)
	    .value("ATTENUATION", ProjectionPropertyType::ATTENUATION)
	    .value("ATTENUATION_PRECORRECTION",
	           ProjectionPropertyType::ATTENUATION_PRECORRECTION)
	    .value("SCATTER_ESTIMATE", ProjectionPropertyType::SCATTER_ESTIMATE)
	    .value("RANDOMS_ESTIMATE", ProjectionPropertyType::RANDOMS_ESTIMATE)
	    .export_values();
}
}  // namespace yrt

#endif

namespace yrt
{

template <typename Enum>
PropStructManager<Enum>::PropStructManager(const std::set<Enum>& props)
{
	const auto& info = getInfo();
	type = 0;
	elementSize = 0;
	std::fill(offsetMap, offsetMap + static_cast<int>(Enum::COUNT), -1);
	for (int i = 0; i < static_cast<int>(Enum::COUNT); i++)
	{
		auto var = static_cast<Enum>(i);
		if (props.contains(var))
		{
			type |= (1 << i);
			offsetMap[i] = elementSize;
			elementSize += info.at(var).second;
		}
	}
}

template <>
std::map<ProjectionPropertyType, std::pair<std::string, int>>
    PropStructManager<ProjectionPropertyType>::getInfo() const
{
	return std::map<ProjectionPropertyType, std::pair<std::string, int>>{
	    {ProjectionPropertyType::DET_ID, {"DET_ID", sizeof(det_pair_t)}},
	    {ProjectionPropertyType::TIMESTAMP, {"TIMESTAMP", sizeof(timestamp_t)}},
	    {ProjectionPropertyType::LOR, {"LOR", sizeof(Line3D)}},
	    {ProjectionPropertyType::DET_ORIENT,
	     {"DET_ORIENT", sizeof(det_orient_t)}},
	    {ProjectionPropertyType::MEASUREMENT, {"MEASUREMENT", sizeof(float)}},
	    {ProjectionPropertyType::TOF, {"TOF", sizeof(float)}},
	    {ProjectionPropertyType::DYNAMIC_FRAME,
	     {"DYNAMIC_FRAME", sizeof(frame_t)}},
	    {ProjectionPropertyType::SENSITIVITY, {"SENSITIVITY", sizeof(float)}},
	    {ProjectionPropertyType::ATTENUATION, {"ATTENUATION", sizeof(float)}},
	    {ProjectionPropertyType::ATTENUATION_PRECORRECTION,
	     {"ATTENUATION_PRECORRECTION", sizeof(float)}},
	    {ProjectionPropertyType::SCATTER_ESTIMATE,
	     {"SCATTER_ESTIMATE", sizeof(float)}},
	    {ProjectionPropertyType::RANDOMS_ESTIMATE,
	     {"RANDOMS_ESTIMATE", sizeof(float)}}};
}
template <>
std::map<ConstraintVariableType, std::pair<std::string, int>>
    PropStructManager<ConstraintVariableType>::getInfo() const
{
	return std::map<ConstraintVariableType, std::pair<std::string, int>>{
	    {ConstraintVariableType::DET1, {"DET1", sizeof(det_id_t)}},
	    {ConstraintVariableType::DET2, {"DET2", sizeof(det_id_t)}},
	    {ConstraintVariableType::ABS_DELTA_ANGLE_DEG,
	     {"ABS_DELTA_ANGLE_DEG", sizeof(float)}},
	    {ConstraintVariableType::ABS_DELTA_ANGLE_IDX,
	     {"ABS_DELTA_ANGLE_IDX", sizeof(int)}},
	    {ConstraintVariableType::ABS_DELTA_BLOCK_IDX,
	     {"ABS_DELTA_BLOCK_IDX", sizeof(int)}}};
}

template <typename Enum>
std::unique_ptr<PropertyUnit[]>
    PropStructManager<Enum>::createDataArray(size_t numElements) const
{
	return std::make_unique<PropertyUnit[]>(numElements *
	                                        static_cast<size_t>(elementSize));
}

template <typename Enum>
unsigned int PropStructManager<Enum>::getElementSize() const
{
	return elementSize;
}

template <typename Enum>
int PropStructManager<Enum>::getTypeID() const
{
	return type;
}

template <typename Enum>
int PropStructManager<Enum>::getOffset(Enum prop) const
{
	return offsetMap[static_cast<int>(prop)];
}

template class PropStructManager<ProjectionPropertyType>;
template class PropStructManager<ConstraintVariableType>;

}  // namespace yrt
