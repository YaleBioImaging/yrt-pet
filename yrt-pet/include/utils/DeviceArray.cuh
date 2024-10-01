/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "utils/GPUMemory.cuh"

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
			allocate(p_size, stream, true);
		}
	}

	~DeviceArray() { Deallocate(); }

	bool allocate(size_t p_size, const cudaStream_t* stream = nullptr,
	              bool synchronize = true)
	{
		if (p_size > m_size)
		{
			if (m_size > 0)
			{
				Deallocate(stream);
			}
			Util::allocateDevice(&mpd_data, p_size, stream, synchronize);
			m_size = p_size;
			return true;
		}
		return false;
	}

	void copyFromHost(const T* source, size_t numElements,
	                  const cudaStream_t* stream = nullptr,
	                  bool synchronize = true)
	{
		Util::copyHostToDevice(mpd_data, source, numElements, stream,
		                       synchronize);
	}

	void copyToHost(T* dest, size_t numElements,
	                const cudaStream_t* stream = nullptr,
	                bool synchronize = true)
	{
		Util::copyDeviceToHost(dest, mpd_data, numElements, stream,
		                       synchronize);
	}

	void memset(int value, const cudaStream_t* stream = nullptr,
	            bool synchronize = true)
	{
		Util::memsetDevice(mpd_data, value, m_size, stream, synchronize);
	}

	void Deallocate(const cudaStream_t* stream = nullptr,
	                bool synchronize = true)
	{
		if (m_size > 0)
		{
			Util::deallocateDevice(mpd_data, stream, synchronize);
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
