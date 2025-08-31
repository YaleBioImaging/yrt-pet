/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/utils/Types.hpp"
#include <bitset>

TEST_CASE("proj_props", "[projProps]")
{
	SECTION("props-simple-elementSize")
	{
		std::set<yrt::ProjectionPropertyType> variables{
		    yrt::ProjectionPropertyType::LOR, yrt::ProjectionPropertyType::TOF};
		yrt::PropStructManager<yrt::ProjectionPropertyType> propManager(
		    variables);
		REQUIRE(propManager.getElementSize() ==
		        sizeof(yrt::Line3D) + sizeof(float));
	}

	SECTION("props-simple-elementSize")
	{
		std::set<yrt::ProjectionPropertyType> variables;
		variables.insert(yrt::ProjectionPropertyType::DET_ID);
		variables.insert(yrt::ProjectionPropertyType::DYNAMIC_FRAME);
		yrt::PropStructManager<yrt::ProjectionPropertyType> propManager(
		    variables);

		// Create data list
		unsigned int numElements = 10;
		auto data = propManager.createDataArray(numElements);
		for (unsigned int i = 0; i < numElements; i++)
		{
			yrt::det_pair_t d;
			d.d1 = i;
			d.d2 = i + 1;
			propManager.setDataValue(data.get(), i,
			                         yrt::ProjectionPropertyType::DET_ID, d);
			yrt::frame_t frame = 12 + i;
			propManager.setDataValue(
			    data.get(), i, yrt::ProjectionPropertyType::DYNAMIC_FRAME, frame);
		}
		// Get data
		for (unsigned int i = 0; i < numElements; i++)
		{
			const yrt::det_pair_t& det_pair =
			    propManager.getDataValue<yrt::det_pair_t>(
			        data.get(), i, yrt::ProjectionPropertyType::DET_ID);
			const yrt::frame_t frame = propManager.getDataValue<int>(
			    data.get(), i, yrt::ProjectionPropertyType::DYNAMIC_FRAME);
			REQUIRE(det_pair.d1 == i);
			REQUIRE(det_pair.d2 == i + 1);
			REQUIRE(frame == static_cast<yrt::frame_t>(12 + i));
		}
	}
}
