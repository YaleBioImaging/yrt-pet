/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/Vector3D.hpp"
#include <cstdint>
#include <variant>

namespace yrt
{

typedef uint32_t det_id_t;  // detector ids
typedef uint64_t bin_t;     // histogram bins or listmode event ids
typedef uint64_t size_t;
typedef uint32_t timestamp_t;  // timestamps in milliseconds
typedef int32_t frame_t;       // motion correction frame

// Defining a pair of detectors
struct det_pair_t
{
	bool operator==(const det_pair_t& other) const
	{
		return d1 == other.d1 && d2 == other.d2;
	}
	det_id_t d1, d2;
};

// Defining a pair of detectors
struct det_pair_tof_t
{
	bool operator==(const det_pair_tof_t& other) const
	{
		return d1 == other.d1 && d2 == other.d2 && tof_ps == other.tof_ps;
	}
	det_id_t d1, d2;
	float tof_ps;
};

struct det_orient_t
{
	Vector3D d1, d2;
};
inline std::ostream& operator<<(std::ostream& oss, const det_orient_t& d)
{
	oss << "[" << d.d1 << ", " << d.d2 << "]";
	return oss;
}

// Defining an LOR
using histo_bin_t = std::variant<det_pair_t, det_pair_tof_t, bin_t>;

// For defining a rotation & translation
struct transform_t
{
	float r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz;
};

}  // namespace yrt
