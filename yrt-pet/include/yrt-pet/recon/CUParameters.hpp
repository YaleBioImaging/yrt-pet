/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

namespace yrt
{

struct CUScannerParams
{
	float crystalSize_trans;
	float crystalSize_z;
	size_t numDets;
};

struct CUImageParams
{
	int nx,ny,nz;
	float vx, vy, vz;
	float length_x, length_y, length_z;
	float off_x, off_y, off_z;
	float fovRadius;
};

struct CUImage
{
	CUImageParams params;
	float* devicePointer = nullptr;
};

}  // namespace yrt
