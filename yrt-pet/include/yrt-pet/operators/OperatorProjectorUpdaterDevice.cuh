/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/OperatorProjectorUpdaterBase.hpp"
#include "yrt-pet/utils/Types.hpp"
#include <cuda_runtime.h>

namespace yrt
{
class OperatorProjectorUpdaterDevice : public OperatorProjectorUpdaterBase
{
public:
	OperatorProjectorUpdaterDevice() = default;
	virtual ~OperatorProjectorUpdaterDevice() = default;

	__device__ virtual float forwardUpdate(
		float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame = 0,
		size_t numVoxelPerFrame = 0
		) const = 0;
	__device__ virtual void backUpdate(
		float value, float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame = 0,
		size_t numVoxelPerFrame = 0
		) = 0;
};


class OperatorProjectorUpdaterDeviceDefault3D : public OperatorProjectorUpdaterDevice
{
public:
	OperatorProjectorUpdaterDeviceDefault3D() = default;

	__device__ float forwardUpdate(
		float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame = 0,
		size_t numVoxelPerFrame = 0
		) const override
	{
		(void) dynamicFrame;
		(void) numVoxelPerFrame;
		return weight * cur_img_ptr[offset];
	}

	__device__ void backUpdate(
		float value, float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame = 0,
		size_t numVoxelPerFrame = 0
		) override
	{
		(void) dynamicFrame;
		(void) numVoxelPerFrame;
		float output = value * weight;
		atomicAdd(cur_img_ptr + offset, output);
	}
};

class OperatorProjectorUpdaterDeviceDefault4D : public OperatorProjectorUpdaterDevice
{
public:
	OperatorProjectorUpdaterDeviceDefault4D() = default;

	__device__ float forwardUpdate(
		float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame = 0,
		size_t numVoxelPerFrame = 0
		) const override
	{
		return weight * cur_img_ptr[dynamicFrame * numVoxelPerFrame + offset];
	}

	__device__ void backUpdate(
		float value, float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame = 0,
		size_t numVoxelPerFrame = 0
		) override
	{
		float output = value * weight;
		atomicAdd(cur_img_ptr + dynamicFrame * numVoxelPerFrame + offset, output);
	}
};


}
