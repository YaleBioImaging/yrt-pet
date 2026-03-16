/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Types.hpp"
#include <vector>

namespace yrt
{

class DynamicFraming
{
public:
	// Create a virgin dynamic framing
	//  (Would have to later be defined with "setStartingTimestamp")
	explicit DynamicFraming(size_t numFrames);
	// Create from an existing std::vector
	explicit DynamicFraming(
	    const std::vector<timestamp_t>& dynamicFramingVector);

	// Return the number of frames
	size_t getNumFrames() const;
	// Return the timestamp of the frame "frame" minus the subsequent timestamp
	float getDuration(frame_t frame) const;
	// Return the last timestamp minus the first timestamp
	float getTotalDuration() const;
	// Different from "getNumFrames" as this includes the last timestamp
	size_t getNumTimestamps() const;
	// Set the timestamp at which the frame "frame" starts
	void setStartingTimestamp(frame_t frame, timestamp_t timestamp);
	// Set the timestamp at which the entire dynamic framing ends
	void setLastTimestamp(timestamp_t timestamp);
	// Validate the chronological order of the timestamps
	bool isValid() const;
	// Get the timestamp at which a frame starts
	timestamp_t getStartingTimestamp(frame_t frame) const;
	// Get the timestamp at which a frame ends
	timestamp_t getStoppingTimestamp(frame_t frame) const;

private:
	// Vector of n + 1 elements (where n = number of frames),
	//  with starting timestamps of each frame, and with the
	//  last element being the last timestamp of the last frame
	std::vector<timestamp_t> m_frameTimestamps;
};

}  // namespace yrt
