/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"
#include <stdio.h>

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"


template <typename T>
float getTOF(T& obj)
{
	return obj.TOFValue.tofValue;
}

TEST_CASE("proj_props", "[projProps]")
{
	yrt::ProjProps::ACFInVivo a;
	a.acfInVivo = 12.f;

	using TestType =
		yrt::ProjProps::GetProductType<yrt::ProjProps::Dets, yrt::ProjProps::LOR,
		                               yrt::ProjProps::TOFValue>::type;
	TestType props;
	getTOF(props);
	//std::cout << "Test " << yrt::ProjProps::TestHasDet<TestType>::value << std::endl;

}
