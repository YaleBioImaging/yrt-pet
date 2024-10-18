/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cstdint>
#include <variant>

typedef uint32_t det_id_t; // detector ids
typedef uint64_t bin_t; // histogram bins or listmode event ids
typedef uint64_t size_t;
typedef uint32_t timestamp_t; // timestamps in milliseconds
typedef int32_t frame_t; // motion correction frame

// Defining a pair of detectors
struct det_pair_t
{
	det_id_t d1, d2;
};

// Defining an LOR
using histo_bin_t = std::variant<det_pair_t, bin_t>;

// For defining a rotation & translation
struct transform_t
{
	float r00, r01, r02, r10, r11, r12, r20, r21, r22;
	float tx, ty, tz;
};
