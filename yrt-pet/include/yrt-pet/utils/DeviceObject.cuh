/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"

namespace yrt
{

template <typename T>
class DeviceObject
{
public:
	template <typename... Args>
	DeviceObject(Args&&... args) : m_hostSideObject(std::forward<Args>(args)...)
	{
		// Allocate on device
		cudaMalloc(&mpd_deviceSideObject, sizeof(T));
		// Copy to device
		cudaMemcpy(mpd_deviceSideObject, &m_hostSideObject, sizeof(T),
		           cudaMemcpyHostToDevice);
		// Synchronize
		cudaDeviceSynchronize();
		ASSERT(cudaCheckError());
	}

	~DeviceObject() { cudaFree(mpd_deviceSideObject); }

	void syncFromHostToDevice(const cudaStream_t* pp_stream = nullptr)
	{
		if (pp_stream != nullptr)
		{
			cudaMemcpyAsync(mpd_deviceSideObject, &m_hostSideObject, sizeof(T),
			                cudaMemcpyHostToDevice, *pp_stream);
		}
		else
		{
			cudaMemcpy(mpd_deviceSideObject, &m_hostSideObject, sizeof(T),
			           cudaMemcpyHostToDevice);
		}
		ASSERT(cudaCheckError());
	}

	void syncFromDeviceToHost(const cudaStream_t* pp_stream = nullptr)
	{
		if (pp_stream != nullptr)
		{
			cudaMemcpyAsync(&m_hostSideObject, mpd_deviceSideObject, sizeof(T),
			                cudaMemcpyDeviceToHost, *pp_stream);
		}
		else
		{
			cudaMemcpy(&m_hostSideObject, mpd_deviceSideObject, sizeof(T),
			           cudaMemcpyDeviceToHost);
		}
		ASSERT(cudaCheckError());
	}

	T& getHostObject() { return m_hostSideObject; }
	const T& getHostObject() const { return m_hostSideObject; }
	const T* getDevicePointer() const { return mpd_deviceSideObject; }
	T* getDevicePointer() { return mpd_deviceSideObject; }

private:
	T m_hostSideObject;
	T* mpd_deviceSideObject;
};

}  // namespace yrt
