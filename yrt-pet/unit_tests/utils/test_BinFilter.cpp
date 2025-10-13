/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/projection/BinFilter.hpp"
#include "yrt-pet/datastruct/projection/Constraints.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/geometry/Vector3D.hpp"
#include "yrt-pet/utils/Types.hpp"

#include <cstdint>
#include <memory>

using TestLOR = std::vector<std::pair<yrt::det_pair_t, bool>>;

void testIndexDiffHelper(yrt::Scanner& scanner, yrt::Constraint& constraint,
                         TestLOR& testLORs)
{
	std::vector<yrt::Constraint*> constraints;
	constraints.emplace_back(&constraint);
	std::set<yrt::ProjectionPropertyType> projProperties{
	    yrt::ProjectionPropertyType::LOR};

	yrt::BinFilter binFilter(constraints, projProperties);
	yrt::BinFilter::CollectInfoFlags collectFlags;
	binFilter.collectFlags(collectFlags);

	auto& projPropManager = binFilter.getPropertyManager();
	auto projectionProperties = projPropManager.createDataArray(1);
	auto projectionPropertiesPtr = projectionProperties.get();

	auto& consManager = binFilter.getConstraintManager();
	auto constraintParams = consManager.createDataArray(1);
	auto constraintParamsPtr = constraintParams.get();

	yrt::ListModeLUTOwned projData(scanner);
	projData.allocate(testLORs.size());
	for (size_t i = 0; i < testLORs.size(); i++)
	{
		projData.setDetectorIdsOfEvent(i, testLORs[i].first.d1,
		                               testLORs[i].first.d2);
	}

	for (size_t i = 0; i < testLORs.size(); i++)
	{
		binFilter.collectInfo(i, 0, 0, projData, collectFlags,
		                      projectionPropertiesPtr, constraintParamsPtr);
		INFO(std::format("Line {} (d1={} d2={} valid? {})", i,
		                 testLORs[i].first.d1, testLORs[i].first.d2,
		                 testLORs[i].second));
		REQUIRE(binFilter.isValid(consManager, constraintParamsPtr) ==
		        testLORs[i].second);
	}
}

double angleBetweenVectors(const yrt::Vector3D& a, const yrt::Vector3D& b)
{
	double dp = a.x * b.x + a.y * b.y;
	double mag_a = std::sqrt(a.x * a.x + a.y * a.y);
	double mag_b = std::sqrt(b.x * b.x + b.y * b.y);
	if (mag_a == 0.0 || mag_b == 0.0)
	{
		return 0.0;
	}
	double cos_theta = dp / (mag_a * mag_b);
	cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
	return std::acos(cos_theta);
}

TEST_CASE("binfilter", "[binfilter]")
{
	auto scanner = yrt::util::test::makeScanner();

	SECTION("block_idx-diff")
	{
		auto constraint = std::make_unique<yrt::ConstraintBlockDiffIndex>(2);
		TestLOR lors{
		    {{0, static_cast<uint32_t>(scanner->detsPerBlock)}, false},
		    {{0, static_cast<uint32_t>(2 * scanner->detsPerBlock)}, true},
		    {{static_cast<uint32_t>(scanner->detsPerBlock - 1),
		      static_cast<uint32_t>(2 * scanner->detsPerBlock)},
		     true},
		    {{
		         static_cast<uint32_t>(scanner->detsPerBlock - 1),
		         static_cast<uint32_t>(scanner->detsPerRing - 1),
		     },
		     false},
		    {{
		         static_cast<uint32_t>(scanner->detsPerBlock),
		         static_cast<uint32_t>(scanner->detsPerRing - 1),
		     },
		     true}};
		testIndexDiffHelper(*scanner.get(), *constraint.get(), lors);
	}

	SECTION("angle_idx-diff")
	{
		auto constraint = std::make_unique<yrt::ConstraintAngleDiffIndex>(5);
		TestLOR lors{
		    {{0, 4}, false},
		    {{100, 105}, true},
		    {{static_cast<uint32_t>(scanner->detsPerRing - 1), 3}, false},
		    {{static_cast<uint32_t>(scanner->detsPerRing - 1), 4}, true}};
		testIndexDiffHelper(*scanner.get(), *constraint.get(), lors);
	}

	SECTION("angle_deg-diff")
	{
		float angleMin = 40.0f;
		auto constraint =
		    std::make_unique<yrt::ConstraintAngleDiffDeg>(angleMin);
		yrt::det_id_t d1 = scanner->detsPerRing - 10;
		yrt::Vector3D p1 = scanner->getDetectorPos(d1);
		yrt::det_id_t numDets = 12;
		TestLOR lors{{{0, 0}, false}};
		for (size_t idx = 0; idx < numDets; idx++)
		{
			yrt::det_id_t d2 = (d1 + idx) % scanner->detsPerRing;
			yrt::Vector3D p2 = scanner->getDetectorPos(d2);
			float angle = angleBetweenVectors(p1, p2) / yrt::PI * 180.f;
			lors.push_back({{d1, d2}, angle >= angleMin});
		}
		testIndexDiffHelper(*scanner.get(), *constraint.get(), lors);
	}
}
