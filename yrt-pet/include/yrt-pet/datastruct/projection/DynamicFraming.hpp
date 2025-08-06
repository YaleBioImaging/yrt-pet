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
	explicit DynamicFraming(const std::vector<timestamp_t>& dynamicFramingVector);
	size_t getNumFrames() const;
	float getDuration(frame_t frame) const;
	float getTotalDuration() const;
	size_t getNumTimestamps() const;
	void setStartingTimestamp(frame_t frame, timestamp_t timestamp);
	timestamp_t getStartingTimestamp(frame_t frame) const;
	timestamp_t getStoppingTimestamp(frame_t frame) const;

	
private:
	/*
	 * vector of n + 1 elements (where n = number of frames),
	 * with starting timestamps of each frame, and with the
	 * last element being the last timestamp of the last frame
	 */
	std::vector<timestamp_t> m_frameTimestamps;
};

}

