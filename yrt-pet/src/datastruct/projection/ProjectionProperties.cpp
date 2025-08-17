/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include <stdexcept>

namespace yrt
{

template <typename Enum>
PropStructManager<Enum>::PropStructManager(std::set<Enum>& props)
{
	const auto& info = getInfo();
	type = 0;
	elementSize = 0;
	for (int i = 0; i < static_cast<int>(Enum::COUNT); i++)
	{
		auto var = static_cast<Enum>(i);
		if (props.find(var) != props.end())
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
	    {ProjectionPropertyType::DetID, {"DET_ID", sizeof(det_pair_t)}},
	    {ProjectionPropertyType::LOR, {"LOR", sizeof(Line3D)}},
	    {ProjectionPropertyType::DetOrient, {"ORIENT", sizeof(det_orient_t)}},
	    {ProjectionPropertyType::TOF, {"TOF", sizeof(float)}},
	    {ProjectionPropertyType::EventFrame, {"FRAME", sizeof(frame_t)}}};
}
template <>
std::map<ConstraintVariable, std::pair<std::string, int>>
    PropStructManager<ConstraintVariable>::getInfo() const
{
	return std::map<ConstraintVariable, std::pair<std::string, int>>{
	    {ConstraintVariable::Det1, {"Det1", sizeof(det_id_t)}},
	    {ConstraintVariable::Det2, {"Det2", sizeof(det_id_t)}},
	    {ConstraintVariable::AbsDeltaAngleDeg,
	     {"AbsDeltaAngleDeg", sizeof(float)}},
	    {ConstraintVariable::AbsDeltaAngleIdx,
	     {"AbsDeltaAngleIdx", sizeof(int)}},
	    {ConstraintVariable::AbsDeltaBlockIdx,
	     {"AbsDeltaBlockIdx", sizeof(int)}}};
}

template <typename Enum>
std::unique_ptr<char>
    PropStructManager<Enum>::createDataArray(size_t numElements) const
{
	return std::make_unique<char>(numElements *
	                              static_cast<size_t>(elementSize));
}

template <typename Enum>
template <typename T>
T* PropStructManager<Enum>::getDataPtr(char* data, int idx, Enum prop) const
{
	return reinterpret_cast<T*>(data + elementSize * idx +
	                            offsetMap[static_cast<int>(prop)]);
}

template <typename Enum>
template <typename T>
void PropStructManager<Enum>::setDataValue(char* data, int idx, Enum prop,
                                           T& value) const
{
	T* ptr = getDataPtr<T>(data, idx, prop);
	*ptr = value;
}

template <typename Enum>
template <typename T>
T& PropStructManager<Enum>::getDataValue(char* data, int idx, Enum prop) const
{
	T* ptr = getDataPtr<T>(data, idx, prop);
	return *ptr;
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
unsigned int PropStructManager<Enum>::getOffset(Enum prop) const
{
	return offsetMap[static_cast<int>(prop)];
}

template <typename Enum>
bool PropStructManager<Enum>::has(Enum prop) const
{
	return type & (1 << static_cast<int>(prop));
}

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
			oss << info.at(var).first << ": "
			    << info.at(var).second
			    << " (offset: " << t.getOffset(var) << "), ";
		}
	}
	oss << "]";
	return oss;
}

template class PropStructManager<ProjectionPropertyType>;
template class PropStructManager<ConstraintVariable>;

template float* PropStructManager<ProjectionPropertyType>::getDataPtr(
    char* data, int idx, ProjectionPropertyType prop) const;
template Line3D* PropStructManager<ProjectionPropertyType>::getDataPtr(
    char* data, int idx, ProjectionPropertyType prop) const;
template det_orient_t* PropStructManager<ProjectionPropertyType>::getDataPtr(
    char* data, int idx, ProjectionPropertyType prop) const;
template det_pair_t* PropStructManager<ProjectionPropertyType>::getDataPtr(
    char* data, int idx, ProjectionPropertyType prop) const;
template frame_t* PropStructManager<ProjectionPropertyType>::getDataPtr(
    char* data, int idx, ProjectionPropertyType prop) const;

template void PropStructManager<ProjectionPropertyType>::setDataValue(
    char* data, int idx, ProjectionPropertyType prop, float&) const;
template void PropStructManager<ProjectionPropertyType>::setDataValue(
    char* data, int idx, ProjectionPropertyType prop, Line3D&) const;
template void PropStructManager<ProjectionPropertyType>::setDataValue(
    char* data, int idx, ProjectionPropertyType prop, det_orient_t&) const;
template void PropStructManager<ProjectionPropertyType>::setDataValue(
    char* data, int idx, ProjectionPropertyType prop, det_pair_t&) const;
template void PropStructManager<ProjectionPropertyType>::setDataValue(
    char* data, int idx, ProjectionPropertyType prop, frame_t&) const;

template float& PropStructManager<ProjectionPropertyType>::getDataValue(
    char* data, int idx, ProjectionPropertyType prop) const;
template Line3D& PropStructManager<ProjectionPropertyType>::getDataValue(
    char* data, int idx, ProjectionPropertyType prop) const;
template det_orient_t& PropStructManager<ProjectionPropertyType>::getDataValue(
    char* data, int idx, ProjectionPropertyType prop) const;
template det_pair_t& PropStructManager<ProjectionPropertyType>::getDataValue(
    char* data, int idx, ProjectionPropertyType prop) const;
template frame_t& PropStructManager<ProjectionPropertyType>::getDataValue(
    char* data, int idx, ProjectionPropertyType prop) const;

template float* PropStructManager<ConstraintVariable>::getDataPtr(
    char* data, int idx, ConstraintVariable prop) const;
template Line3D* PropStructManager<ConstraintVariable>::getDataPtr(
    char* data, int idx, ConstraintVariable prop) const;
template det_orient_t* PropStructManager<ConstraintVariable>::getDataPtr(
    char* data, int idx, ConstraintVariable prop) const;
template det_pair_t* PropStructManager<ConstraintVariable>::getDataPtr(
    char* data, int idx, ConstraintVariable prop) const;
template frame_t* PropStructManager<ConstraintVariable>::getDataPtr(
    char* data, int idx, ConstraintVariable prop) const;

template void PropStructManager<ConstraintVariable>::setDataValue(
    char* data, int idx, ConstraintVariable prop, float&) const;
template void PropStructManager<ConstraintVariable>::setDataValue(
    char* data, int idx, ConstraintVariable prop, Line3D&) const;
template void PropStructManager<ConstraintVariable>::setDataValue(
    char* data, int idx, ConstraintVariable prop, det_orient_t&) const;
template void PropStructManager<ConstraintVariable>::setDataValue(
    char* data, int idx, ConstraintVariable prop, det_pair_t&) const;
template void PropStructManager<ConstraintVariable>::setDataValue(
    char* data, int idx, ConstraintVariable prop, frame_t&) const;

template float& PropStructManager<ConstraintVariable>::getDataValue(
    char* data, int idx, ConstraintVariable prop) const;
template Line3D& PropStructManager<ConstraintVariable>::getDataValue(
    char* data, int idx, ConstraintVariable prop) const;
template det_orient_t& PropStructManager<ConstraintVariable>::getDataValue(
    char* data, int idx, ConstraintVariable prop) const;
template det_pair_t& PropStructManager<ConstraintVariable>::getDataValue(
    char* data, int idx, ConstraintVariable prop) const;
template frame_t& PropStructManager<ConstraintVariable>::getDataValue(
    char* data, int idx, ConstraintVariable prop) const;

}  // namespace yrt
