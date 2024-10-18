/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/PageLockedBuffer.cuh"

#if BUILD_CUDA

#include <cuda.h>
#include <iostream>

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
	cudaHostAlloc(reinterpret_cast<void**>(&mph_dataPointer), size * sizeof(T),
	              flags);
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != 0)
	{
		std::cerr << "CUDA Error while allocating: "
		          << cudaGetErrorString(cudaError) << std::endl;
		mph_dataPointer = new T[size];
		m_isPageLocked = false;
	}
	else
	{
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

#endif
