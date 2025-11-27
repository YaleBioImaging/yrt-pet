/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/utils/Types.hpp"
#include <cuda_runtime.h>

namespace yrt
{

class OperatorProjectorUpdaterDevice
{
public:
	__device__ OperatorProjectorUpdaterDevice() {}
	__device__ virtual ~OperatorProjectorUpdaterDevice() {}

	__device__ virtual float forwardUpdate(float weight, float* cur_img_ptr,
	                                       size_t offset, frame_t dynamicFrame,
	                                       size_t numVoxelPerFrame) const = 0;
	__device__ virtual void backUpdate(float value, float weight,
	                                   float* cur_img_ptr, size_t offset,
	                                   frame_t dynamicFrame,
	                                   size_t numVoxelPerFrame) = 0;
};


class OperatorProjectorUpdaterDeviceDefault3D
    : public OperatorProjectorUpdaterDevice
{
public:
	__device__ OperatorProjectorUpdaterDeviceDefault3D(){}

	__device__ float forwardUpdate(float weight, float* cur_img_ptr,
	                               size_t offset, frame_t dynamicFrame,
	                               size_t numVoxelPerFrame) const override

	{
		// printf("\n%.1f", weight);
		return weight * cur_img_ptr[offset];
	}

	__device__ void backUpdate(float value, float weight, float* cur_img_ptr,
	                           size_t offset, frame_t dynamicFrame,
	                           size_t numVoxelPerFrame) override
	{
		float output = value * weight;
		float* ptr = &cur_img_ptr[offset];
		atomicAdd(ptr, output);
	}
};

class OperatorProjectorUpdaterDeviceDefault4D
    : public OperatorProjectorUpdaterDevice
{
public:
	__device__ OperatorProjectorUpdaterDeviceDefault4D() {}

	__device__ float forwardUpdate(float weight, float* cur_img_ptr,
	                               size_t offset, frame_t dynamicFrame,
	                               size_t numVoxelPerFrame) const override
	{
		return weight * cur_img_ptr[dynamicFrame * numVoxelPerFrame + offset];
	}

	__device__ void backUpdate(float value, float weight, float* cur_img_ptr,
	                           size_t offset, frame_t dynamicFrame,
	                           size_t numVoxelPerFrame) override
	{
		float output = value * weight;
		atomicAdd(cur_img_ptr + dynamicFrame * numVoxelPerFrame + offset,
		          output);
	}
};

using UpdaterPointer = OperatorProjectorUpdaterDevice**;

inline __global__ void constructUpdaterOnDevice(
    UpdaterPointer ppd_updater,
    const OperatorProjectorParams::ProjectorUpdaterType p_updaterType)
{
	// It is necessary to create object representing a function
	// directly in global memory of the GPU device for virtual
	// functions to work correctly, i.e. virtual function table
	// HAS to be on GPU as well.
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		if (p_updaterType ==
		    OperatorProjectorParams::ProjectorUpdaterType::DEFAULT3D)
		{
			*ppd_updater = new OperatorProjectorUpdaterDeviceDefault3D();
		}
		else if (p_updaterType ==
		         OperatorProjectorParams::ProjectorUpdaterType::DEFAULT4D)
		{
			*ppd_updater = new OperatorProjectorUpdaterDeviceDefault4D();
		}
		else
		{
			// TODO : Have the LR Updater instantiated here once the class
			//  exists (Also requires to add an argument to have the H
			//  matrix there)
		}
	}
}

inline __global__ void destroyUpdaterOnDevice(UpdaterPointer ppd_updater)
{
	delete *ppd_updater;
}

class OperatorProjectorUpdaterDeviceWrapper
{
public:
	OperatorProjectorUpdaterDeviceWrapper() : mpd_updater(nullptr) {}

	~OperatorProjectorUpdaterDeviceWrapper()
	{
		if (mpd_updater != nullptr)
		{
			destroyUpdaterOnDevice<<<1, 1>>>(mpd_updater);
		}
	}

	void initUpdater(
	    const OperatorProjectorParams::ProjectorUpdaterType updaterType)
	{
		cudaMalloc(&mpd_updater, sizeof(UpdaterPointer));
		constructUpdaterOnDevice<<<1, 1>>>(mpd_updater, updaterType);
		cudaDeviceSynchronize();
	}

	UpdaterPointer getUpdaterDevicePointer()
	{
		if (mpd_updater == nullptr)
		{
			return nullptr;
		}
		return mpd_updater;
	}

private:
	UpdaterPointer mpd_updater;
};

}  // namespace yrt
