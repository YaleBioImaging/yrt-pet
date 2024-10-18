/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/ProjectionPsfManagerDevice.cuh"

#include <cuda_runtime.h>

ProjectionPsfManagerDevice::ProjectionPsfManagerDevice(
    const std::string& psfFilename, const cudaStream_t* pp_stream)
    : ProjectionPsfManager{}
{
	readFromFileInternal(psfFilename, pp_stream);
}

void ProjectionPsfManagerDevice::readFromFile(const std::string& psfFilename)
{
	readFromFileInternal(psfFilename, nullptr);
}

void ProjectionPsfManagerDevice::readFromFile(const std::string& psfFilename,
                                              const cudaStream_t* pp_stream)
{
	readFromFileInternal(psfFilename, pp_stream);
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
    const std::string& psfFilename, const cudaStream_t* pp_stream)
{
	ProjectionPsfManager::readFromFile(psfFilename);
	copyKernelsToDevice(pp_stream);
}

void ProjectionPsfManagerDevice::copyKernelsToDevice(
    const cudaStream_t* pp_stream)
{
	// TODO: Copy kernels to device
	const size_t kernelSize = m_kernels.getSizeTotal();
	ASSERT(kernelSize == m_kernelsFlipped.getSizeTotal());

	if (mpd_kernels == nullptr)
	{
		mpd_kernels =
		    std::make_unique<DeviceArray<float>>(kernelSize, pp_stream);
	}
	if (mpd_kernelsFlipped == nullptr)
	{
		mpd_kernelsFlipped =
		    std::make_unique<DeviceArray<float>>(kernelSize, pp_stream);
	}

	mpd_kernels->copyFromHost(m_kernels.getRawPointer(), kernelSize, pp_stream);
	mpd_kernelsFlipped->copyFromHost(m_kernelsFlipped.getRawPointer(),
	                                 kernelSize, pp_stream);
	if (pp_stream != nullptr)
	{
		cudaStreamSynchronize(*pp_stream);
	}
}
