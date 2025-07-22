/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/PageLockedBuffer.cuh"

#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"

#if BUILD_CUDA

#include <cuda.h>
#include <iostream>

namespace yrt
{

template <typename T>
PageLockedBuffer<T>::PageLockedBuffer()
    : mph_dataPointer(nullptr),
      m_size(0ull),
      m_isPageLocked(false),
      m_currentFlags(0u)
{
}

template <typename T>
PageLockedBuffer<T>::PageLockedBuffer(const size_t size,
                                      const unsigned int flags)
    : PageLockedBuffer()
{
	allocate(size, flags);
}

template <typename T>
void PageLockedBuffer<T>::allocate(const size_t size, const unsigned int flags)
{
	ASSERT_MSG(
	    mph_dataPointer == nullptr,
	    "Memory already allocated, cannot allocate twice on the same pointer");

	cudaError_t cudaError{};
	bool canPageLockMemory = false;  // Cannot page-lock until proven otherwise

	if (globals::isPinnedMemoryEnabled())
	{
		const size_t size_bytes = size * sizeof(T);
		cudaHostAlloc(reinterpret_cast<void**>(&mph_dataPointer), size_bytes,
		              flags);
		cudaError = cudaGetLastError();

		canPageLockMemory = cudaError == 0;  // No problem
	}

	if (!canPageLockMemory)
	{
		if (cudaError != 0)
		{
			// There was a problem
			std::cerr << "CUDA Error while allocating: "
			          << cudaGetErrorString(cudaError) << std::endl;
		}
		// Use regular memory
		mph_dataPointer = new T[size];
		m_isPageLocked = false;
	}
	else
	{
		// No problem, use page-locked memory
		m_isPageLocked = true;
		m_currentFlags = flags;
	}
	m_size = size;
}

template <typename T>
bool PageLockedBuffer<T>::reAllocateIfNeeded(const size_t newSize,
                                             const unsigned int flags)
{
	if (newSize > m_size || m_currentFlags != flags)
	{
		deallocate();
		allocate(newSize, flags);
		return true;
	}
	return false;
}
template <typename T>
void PageLockedBuffer<T>::deallocate()
{
	if (m_size > 0ull)
	{
		if (m_isPageLocked)
		{
			cudaFreeHost(mph_dataPointer);
			const cudaError_t cudaError = cudaGetLastError();
			if (cudaError != 0)
			{
				std::cerr << "CUDA Error while freeing: "
				          << cudaGetErrorString(cudaError) << std::endl;
			}
			else
			{
				m_size = 0ull;
			}
		}
		else
		{
			delete[] mph_dataPointer;
			m_size = 0ull;
		}
		mph_dataPointer = nullptr;
	}
}

template <typename T>
PageLockedBuffer<T>::~PageLockedBuffer()
{
	deallocate();
}

template <typename T>
T* PageLockedBuffer<T>::getPointer()
{
	return mph_dataPointer;
}

template <typename T>
const T* PageLockedBuffer<T>::getPointer() const
{
	return mph_dataPointer;
}

template <typename T>
size_t PageLockedBuffer<T>::getSize() const
{
	return m_size;
}

template class PageLockedBuffer<float>;
template class PageLockedBuffer<float3>;
template class PageLockedBuffer<float4>;
template class PageLockedBuffer<long>;
template class PageLockedBuffer<int>;
template class PageLockedBuffer<uint2>;
template class PageLockedBuffer<short>;
template class PageLockedBuffer<char>;

}  // namespace yrt

#endif
