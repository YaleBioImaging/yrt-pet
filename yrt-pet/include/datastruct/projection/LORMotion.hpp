/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/fancyarray/trivial_struct_of_arrays.hpp"
#include "utils/Types.hpp"

#include <vector>

class LORMotion
{
public:
	struct Record
	{
		timestamp_t timestamp;
		transform_t transform;
		float error;
	};

	explicit LORMotion(const std::string& filename);
	explicit LORMotion(size_t numFrames);

	transform_t getTransform(frame_t frame) const;
	timestamp_t getStartingTimestamp(frame_t frame) const;
	void setTransform(frame_t frame, const transform_t& transform);
	void setStartingTimestamp(frame_t frame, timestamp_t timestamp);
	void setError(frame_t frame, float error);
	float getDuration(frame_t frame) const;  // In ms
	float getError(frame_t frame) const;

	size_t getNumFrames() const;
	// Get the total duration of the motion data (in ms) based on the timestamps
	float getTotalDuration() const;
	void readFromFile(const std::string& filename);
	void writeToFile(const std::string& filename) const;

	// Safe getters and setters (for python)
	transform_t getTransformSafe(frame_t frame) const;
	timestamp_t getStartingTimestampSafe(frame_t frame) const;
	void setTransformSafe(frame_t frame, const transform_t& transform);
	void setStartingTimestampSafe(frame_t frame, timestamp_t timestamp);
	void setErrorSafe(frame_t frame, float error);
	float getErrorSafe(frame_t frame) const;

protected:
	void resize(size_t newSize);

private:
	// Member data
	std::vector<Record> m_records;
};
