/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/fancyarray/trivial_struct_of_arrays.hpp"
#include "utils/Types.hpp"

class LORMotion
{
public:
	explicit LORMotion(const std::string& filename);
	explicit LORMotion(size_t numFrames);

	transform_t getTransform(frame_t frame) const;
	timestamp_t getStartingTimestamp(frame_t frame) const;
	void setTransform(frame_t frame, const transform_t& transform);
	void setStartingTimestamp(frame_t frame, timestamp_t timestamp);
	float getDuration(frame_t frame) const;  // In ms
	size_t getNumFrames() const;
	void writeToFile(const std::string& filename) const;
	float getTotalDuration() const;

private:
	// Setup internal pointers for the transforms and the timestamps
	void setupPointers();

	using LORMotionStructure =
	    fancyarray::trivial_struct_of_arrays<timestamp_t, transform_t>;
	LORMotionStructure m_structure;

	transform_t* mp_transforms;
	timestamp_t* mp_startingTimestamps;
};
