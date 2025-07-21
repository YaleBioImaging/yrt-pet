/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/GPUMemory.cuh"

namespace yrt
{

// A 1-dimensional array located in device memory
template <typename T>
class DeviceArray
{
public:
	explicit DeviceArray(size_t p_size = 0ull,
	                     const cudaStream_t* stream = nullptr)
	{
		mpd_data = nullptr;
		m_size = 0;
		if (p_size > 0)
		{
			// Constructors should be synchronous
			allocate(p_size, {stream, true});
		}
	}

	~DeviceArray() { Deallocate({nullptr, true}); }

	bool allocate(size_t p_size, GPULaunchConfig p_launchConfig)
	{
		if (p_size > m_size)
		{
			if (m_size > 0)
			{
				Deallocate(p_launchConfig);
			}
			util::allocateDevice(&mpd_data, p_size, p_launchConfig);
			m_size = p_size;
			return true;
		}
		return false;
	}

	void copyFromHost(const T* source, size_t numElements,
	                  GPULaunchConfig p_launchConfig)
	{
		util::copyHostToDevice(mpd_data, source, numElements, p_launchConfig);
	}

	void copyToHost(T* dest, size_t numElements, GPULaunchConfig p_launchConfig)
	{
		util::copyDeviceToHost(dest, mpd_data, numElements, p_launchConfig);
	}

	void memset(int value, GPULaunchConfig p_launchConfig)
	{
		util::memsetDevice(mpd_data, value, m_size, p_launchConfig);
	}

	void Deallocate(GPULaunchConfig p_launchConfig)
	{
		if (m_size > 0)
		{
			util::deallocateDevice(mpd_data, p_launchConfig);
			mpd_data = nullptr;
			m_size = 0;
		}
	}

	bool isAllocated() const { return m_size > 0; }
	size_t getSize() const { return m_size; }

	T* getDevicePointer() { return mpd_data; }
	const T* getDevicePointer() const { return mpd_data; }

private:
	T* mpd_data;
	size_t m_size;
};

}  // namespace yrt
