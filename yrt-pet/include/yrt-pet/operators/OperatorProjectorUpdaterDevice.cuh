/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

//#include "yrt-pet/operators/OperatorProjectorUpdaterBase.hpp"
#include "yrt-pet/utils/Types.hpp"
#include <cuda_runtime.h>

namespace yrt
{
class OperatorProjectorUpdaterDevice //: public OperatorProjectorUpdaterBase
{
public:
	__device__ OperatorProjectorUpdaterDevice() { debug_id = 0; }
	__device__ virtual ~OperatorProjectorUpdaterDevice() {}

	__device__ virtual float debug_float() const = 0;
	__device__ virtual void debug_void(float* cur_img_ptr) = 0;

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

	int debug_id;
};


class OperatorProjectorUpdaterDeviceDefault3D : public OperatorProjectorUpdaterDevice
{
public:
	__device__ OperatorProjectorUpdaterDeviceDefault3D() { debug_id = 1; }

	__device__ virtual float debug_float() const {return 1.0f;}
	__device__ virtual void debug_void(float* cur_img_ptr)
	{
		auto output = 0.02f;
		cur_img_ptr[1] += output;
	}

	__device__ float forwardUpdate(
		float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame = 0,
		size_t numVoxelPerFrame = 0
		) const override
	{
		// (void) dynamicFrame;
		// (void) numVoxelPerFrame;
		// printf("\n%.1f", weight);
		// return weight * cur_img_ptr[offset];
		return weight;
	}

	__device__ void backUpdate(
		float value, float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame = 0,
		size_t numVoxelPerFrame = 0
		) override
	{
		// (void) dynamicFrame;
		// (void) numVoxelPerFrame;
		// float output = value * weight;
		// atomicAdd(cur_img_ptr + offset, output);
		float output = value * weight;
		float* ptr = &cur_img_ptr[offset];
		atomicAdd(ptr, output);
	}
};

class OperatorProjectorUpdaterDeviceDefault4D : public OperatorProjectorUpdaterDevice
{
public:
	__device__ OperatorProjectorUpdaterDeviceDefault4D() { debug_id = 2; }

	__device__ virtual float debug_float() const {return 2.0f;}
	__device__ virtual void debug_void(float* cur_img_ptr)
	{
		auto output = 0.02f;
		cur_img_ptr[1] += output;
	}

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
