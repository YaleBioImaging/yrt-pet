/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/ProjectorUtils.hpp"
#include "yrt-pet/operators/ProjectionPsfManager.hpp"
#include "yrt-pet/utils/DeviceArray.cuh"

#include <cuda_runtime_api.h>

namespace yrt
{
struct ProjectionPsfProperties
{
	float sStep;
	float kSpacing;
	int numKernels;
	int kernelSize;
	float halfWidth;
};

class ProjectionPsfManagerDevice : public ProjectionPsfManager
{
public:
	explicit
	    ProjectionPsfManagerDevice(const std::string& psfFilename,
	                               const cudaStream_t* pp_stream = nullptr);
	void readFromFile(const std::string& psfFilename) override;
	void readFromFile(const std::string& psfFilename,
	                  GPULaunchConfig launchConfig);
	const float* getKernelsDevicePointer() const;
	const float* getFlippedKernelsDevicePointer() const;

	ProjectionPsfProperties getProjectionPsfProperties() const;

protected:
	std::unique_ptr<DeviceArray<float>> mpd_kernels;
	std::unique_ptr<DeviceArray<float>> mpd_kernelsFlipped;

private:
	void copyKernelsToDevice(GPULaunchConfig launchConfig);
	void readFromFileInternal(const std::string& psfFilename,
	                          GPULaunchConfig launchConfig);
};
}  // namespace yrt
