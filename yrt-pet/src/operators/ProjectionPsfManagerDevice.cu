/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/ProjectionPsfManagerDevice.cuh"

#include <cuda_runtime.h>

namespace yrt
{
ProjectionPsfManagerDevice::ProjectionPsfManagerDevice(
    const std::string& psfFilename, const cudaStream_t* pp_stream)
    : ProjectionPsfManager{}
{
	readFromFileInternal(psfFilename, {pp_stream, true});
}

void ProjectionPsfManagerDevice::readFromFile(const std::string& psfFilename)
{
	readFromFileInternal(psfFilename, {nullptr, true});
}

void ProjectionPsfManagerDevice::readFromFile(const std::string& psfFilename,
                                              GPULaunchConfig launchConfig)
{
	readFromFileInternal(psfFilename, launchConfig);
}

ProjectionPsfProperties
    ProjectionPsfManagerDevice::getProjectionPsfProperties() const
{
	return {m_sStep, m_kSpacing, static_cast<int>(m_kernels.getSize(0)),
	        static_cast<int>(m_kernels.getSize(1)), getHalfWidth_mm()};
}

const float* ProjectionPsfManagerDevice::getKernelsDevicePointer() const
{
	return mpd_kernels->getDevicePointer();
}

const float* ProjectionPsfManagerDevice::getFlippedKernelsDevicePointer() const
{
	return mpd_kernelsFlipped->getDevicePointer();
}

void ProjectionPsfManagerDevice::readFromFileInternal(
    const std::string& psfFilename, GPULaunchConfig launchConfig)
{
	ProjectionPsfManager::readFromFile(psfFilename);
	copyKernelsToDevice(launchConfig);
}

void ProjectionPsfManagerDevice::copyKernelsToDevice(
    GPULaunchConfig launchConfig)
{
	// TODO: Copy kernels to device
	const size_t kernelSize = m_kernels.getSizeTotal();
	ASSERT(kernelSize == m_kernelsFlipped.getSizeTotal());

	if (mpd_kernels == nullptr)
	{
		mpd_kernels = std::make_unique<DeviceArray<float>>(kernelSize,
		                                                   launchConfig.stream);
	}
	if (mpd_kernelsFlipped == nullptr)
	{
		mpd_kernelsFlipped = std::make_unique<DeviceArray<float>>(
		    kernelSize, launchConfig.stream);
	}

	mpd_kernels->copyFromHost(m_kernels.getRawPointer(), kernelSize,
	                          {launchConfig.stream, false});
	mpd_kernelsFlipped->copyFromHost(m_kernelsFlipped.getRawPointer(),
	                                 kernelSize, launchConfig);
}
}  // namespace yrt
