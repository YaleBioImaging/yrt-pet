/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/geometry/Constants.hpp"

namespace yrt
{
// Compare two transform_t
bool compareTransforms(const transform_t& a, const transform_t& b,
                       float epsilon = 1e-5)
{
	const auto fa = reinterpret_cast<const float*>(&a);
	const auto fb = reinterpret_cast<const float*>(&b);
	for (size_t i = 0; i < sizeof(transform_t) / sizeof(float); ++i)
	{
		const float diff = std::fabs(fa[i] - fb[i]);
		if (diff > epsilon)
		{
			return false;
		}
	}
	return true;
}
}  // namespace yrt

TEST_CASE("lor-motion", "[motion]")
{
	const std::string tempFilename = "tmp_lormotion.vc";

	// Create sample data
	std::vector<yrt::LORMotion::Record> originalRecords = {
	    {1000,
	     {1.0f, 0.0f, 0.0f, 10.0f, 0.0f, 1.0f, 0.0f, 20.0f, 0.0f, 0.0f, 1.0f,
	      30.0f},
	     1.0f},
	    {2000,
	     {0.707f, 0.707f, 0.0f, 5.0f, -0.707f, 0.707f, 0.0f, 15.0f, 0.0f, 0.0f,
	      1.0f, 25.0f},
	     1.0f}};

	// Populate LORMotion
	const auto lorMot =
	    std::make_unique<yrt::LORMotion>(originalRecords.size());
	for (size_t frame_i = 0; frame_i < originalRecords.size(); frame_i++)
	{
		const yrt::LORMotion::Record record = originalRecords[frame_i];
		lorMot->setStartingTimestamp(frame_i, record.timestamp);
		lorMot->setTransform(frame_i, record.transform);
		lorMot->setError(frame_i, record.error);
	}

	// Write to file
	lorMot->writeToFile(tempFilename);

	// Read the file just written
	auto lorMot2 = std::make_unique<yrt::LORMotion>(tempFilename);

	// Compare
	REQUIRE(lorMot2->getNumFrames() == originalRecords.size());
	for (size_t frame_i = 0; frame_i < originalRecords.size(); frame_i++)
	{
		const yrt::LORMotion::Record originalRecord = originalRecords[frame_i];
		CHECK(compareTransforms(originalRecord.transform,
		                        lorMot2->getTransform(frame_i),
		                        yrt::SMALL_FLT));
		CHECK(originalRecord.timestamp ==
		      lorMot2->getStartingTimestamp(frame_i));
		CHECK(std::fabs(originalRecord.error - lorMot2->getError(frame_i) <
		                yrt::SMALL_FLT));
	}

	// Clean up
	std::remove(tempFilename.c_str());
}
