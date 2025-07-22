/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

namespace yrt
{
typedef struct CUScannerParams
{
	float crystalSize_trans;
	float crystalSize_z;
	size_t numDets;
} CUScannerParams;

typedef struct CUImageParams
{
	int voxelNumber[3];
	float voxelSize[3];
	float imgLength[3];
	float offset[3];
	float fovRadius;
} CUImageParams;
}  // namespace yrt
