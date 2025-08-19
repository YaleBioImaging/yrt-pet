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
std::unique_ptr<char[]>
    PropStructManager<Enum>::createDataArray(size_t numElements) const
{
	return std::make_unique<char[]>(numElements *
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
unsigned int PropStructManager<Enum>::getOffset(Enum prop) const
{
	return offsetMap[static_cast<int>(prop)];
}

template <typename Enum>
bool PropStructManager<Enum>::has(Enum prop) const
{
	return type & (1 << static_cast<int>(prop));
}

template class PropStructManager<ProjectionPropertyType>;
template class PropStructManager<ConstraintVariable>;

}  // namespace yrt
