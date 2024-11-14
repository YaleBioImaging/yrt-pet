/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/LORMotion.hpp"
#include "utils/Assert.hpp"

LORMotion::LORMotion(const std::string& filename)
    : m_structure{filename, 1 << 20}
{
	ASSERT_MSG(getNumFrames() > 0, "There has to be at least one frame");
	mp_startingTimestamps = m_structure.get_pointer<0>();
	mp_transforms = m_structure.get_pointer<1>();
}

transform_t LORMotion::getTransform(frame_t frame) const
{
	return mp_transforms[frame];
}

float LORMotion::getDuration(frame_t frame) const
{
	const size_t numFrames = getNumFrames();

	if (frame < numFrames - 1)
	{
		return mp_startingTimestamps[frame + 1] - mp_startingTimestamps[frame];
	}

	// Last frame, take duration of second-to-last frame
	return mp_startingTimestamps[numFrames - 1] -
	       mp_startingTimestamps[numFrames - 2];
}

timestamp_t LORMotion::getStartingTimestamp(frame_t frame) const
{
	return mp_startingTimestamps[frame];
}

size_t LORMotion::getNumFrames() const
{
	return m_structure.get_num_columns();
}
