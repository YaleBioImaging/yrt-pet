/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#if BUILD_CUDA

namespace yrt
{

// The goal of this class is to provide for a transparent way to create and
// destroy a page-locked buffer that uses "cudaHostAlloc". This cuda function
// can cause errors and leave a null pointer. However, in that case, this class
// will rollback to a regular C++ buffer (using the new/delete idiom), which
// increases memory safety.
template <typename T>
class PageLockedBuffer
{
public:
	PageLockedBuffer();

	explicit PageLockedBuffer(size_t size, unsigned int flags = 0u);

	virtual ~PageLockedBuffer();

	void allocate(size_t size, unsigned int flags = 0u);

	void deallocate();

	bool reAllocateIfNeeded(size_t newSize, unsigned int flags = 0u);

	T* getPointer();
	const T* getPointer() const;
	size_t getSize() const;

private:
	T* mph_dataPointer;
	size_t m_size;
	bool m_isPageLocked;
	unsigned int m_currentFlags;
};

}  // namespace yrt

#endif
