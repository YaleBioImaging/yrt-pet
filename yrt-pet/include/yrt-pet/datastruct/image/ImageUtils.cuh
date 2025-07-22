/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/GPUUtils.cuh"

namespace yrt::util
{

inline HOST_DEVICE_CALLABLE void
    trilinearInterpolate(float pos_x, float pos_y, float pos_z, int nx, int ny,
                         int nz, float length_x, float length_y, float length_z,
                         float off_x, float off_y, float off_z, int indices[8],
                         float weights[8])
{
	// Get voxel size
	const float vx = length_x / nx;
	const float vy = length_y / ny;
	const float vz = length_z / nz;

	// Convert physical position to grid coordinates
	const float cx = (pos_x - off_x + length_x * 0.5f - vx * 0.5f) / vx;
	const float cy = (pos_y - off_y + length_y * 0.5f - vy * 0.5f) / vy;
	const float cz = (pos_z - off_z + length_z * 0.5f - vz * 0.5f) / vz;

	// Integer coordinates of base voxel
	const int ix0 = static_cast<int>(floorf(cx));
	const int iy0 = static_cast<int>(floorf(cy));
	const int iz0 = static_cast<int>(floorf(cz));

	// Fractional components
	const float dx = cx - ix0;
	const float dy = cy - iy0;
	const float dz = cz - iz0;

	// Calculate weights for the 8 neighboring voxels
	weights[0] = (1 - dx) * (1 - dy) * (1 - dz);
	weights[1] = dx * (1 - dy) * (1 - dz);
	weights[2] = (1 - dx) * dy * (1 - dz);
	weights[3] = dx * dy * (1 - dz);
	weights[4] = (1 - dx) * (1 - dy) * dz;
	weights[5] = dx * (1 - dy) * dz;
	weights[6] = (1 - dx) * dy * dz;
	weights[7] = dx * dy * dz;

	// Calculate indices for the 8 neighbors
	constexpr int x_offsets[8] = {0, 1, 0, 1, 0, 1, 0, 1};
	constexpr int y_offsets[8] = {0, 0, 1, 1, 0, 0, 1, 1};
	constexpr int z_offsets[8] = {0, 0, 0, 0, 1, 1, 1, 1};
	const int slice_size = nx * ny;

	for (int i = 0; i < 8; i++)
	{
		const int x = ix0 + x_offsets[i];
		const int y = iy0 + y_offsets[i];
		const int z = iz0 + z_offsets[i];

		// Check bounds and compute index
		if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz)
		{
			indices[i] = x + y * nx + z * slice_size;
		}
		else
		{
			// Out-of-bounds: set weight to zero and use safe index
			weights[i] = 0.0f;
			indices[i] = 0;
		}
	}
}

inline HOST_DEVICE_CALLABLE float indexToPosition(int index, float voxelSize,
                                                  float length, float offset)
{
	return static_cast<float>(index) * voxelSize - 0.5f * length + offset +
	       0.5f * voxelSize;
}

}  // namespace yrt
