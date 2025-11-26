/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"

#if BUILD_CUDA

namespace yrt
{

// --- device-side placement-new / destructor kernels ---
template <typename T, typename... Args>
__global__ void constructOnDevice(void* mem, Args... args)
{
	// Constructs T at address mem using placement-new.
	new (mem) T(std::forward<Args>(args)...);
}

template <typename T>
__global__ void constructOnDeviceNoArgs(void* mem)
{
	// Constructs T at address mem using placement-new.
	new (mem) T();
}

template <typename T>
__global__ void destroyOnDevice(T* obj)
{
	obj->~T();
}


template <typename T>
class DeviceObject
{
public:
	DeviceObject() noexcept : mpd_object(nullptr) {}

	DeviceObject(const DeviceObject&) = delete;
	DeviceObject& operator=(const DeviceObject&) = delete;

	template <typename... Args>
	explicit DeviceObject(Args&&... args) : mpd_object(nullptr)
	{
		// 1) Allocate raw storage on device
		printf("\n\n\ndevice object\n\n\n");
		util::allocateDevice(&mpd_object, 1, {});

		// 2) Construct the object *on device* with placement-new
		constructOnDevice<T, Args...>
		    <<<1, 1>>>(mpd_object, std::forward<Args>(args)...);

		cudaDeviceSynchronize();
		ASSERT(cudaCheckError());
	}

	~DeviceObject() { reset(); }

	DeviceObject(DeviceObject&& other) noexcept : mpd_object(other.mpd_object)
	{
		other.mpd_object = nullptr;
	}

	DeviceObject& operator=(DeviceObject&& other) noexcept
	{
		if (this != &other)
		{
			reset();
			mpd_object = other.mpd_object;
			other.mpd_object = nullptr;
		}
		return *this;
	}

	// converting move ctor
	template <class U, std::enable_if_t<std::is_base_of_v<T, U>, int> = 0>
	DeviceObject(DeviceObject<U>&& other) noexcept
	    : mpd_object(other.mpd_object)
	{
		other.mpd_object = nullptr;
	}

	// converting move assign
	template <class U, std::enable_if_t<std::is_base_of_v<T, U>, int> = 0>
	DeviceObject<T>& operator=(DeviceObject<U>&& other) noexcept
	{
		if (reinterpret_cast<T*>(this) != reinterpret_cast<T*>(&other))
		{
			reset();
			mpd_object = other.mpd_object;
			other.mpd_object = nullptr;
		}
		return *this;
	}

	const T* getDevicePointer() const { return mpd_object; }
	T* getDevicePointer() { return mpd_object; }
	T** getDevicePointerPtr() { return &mpd_object; }
	explicit operator bool() const noexcept { return mpd_object != nullptr; }

	void reset()
	{
		if (mpd_object)
		{
			// Call device-side destructor, then free storage
			// destroyOnDevice<T><<<1, 1>>>(mpd_object);
			// Best-effort; avoid throwing from destructors
			// cudaDeviceSynchronize();
			util::deallocateDevice(mpd_object, {});
			mpd_object = nullptr;
		}
	}

	// protected:
	T* mpd_object;
};

template <typename T>
DeviceObject<T> makeDeviceObject()
{
	DeviceObject<T> dObject;
	// 1) Allocate raw storage on device
	util::allocateDevice(dObject.getDevicePointerPtr(), 1, {});

	// 2) Construct the object *on device* with placement-new (no arguments)
	constructOnDeviceNoArgs<T><<<1, 1>>>(dObject.getDevicePointer());

	cudaDeviceSynchronize();
	ASSERT(cudaCheckError());
	return dObject;
}

template <class Base, class Derived>
DeviceObject<Base> makeDeviceObjectDerived()
{
	static_assert(std::is_base_of<Base, Derived>::value,
				  "Derived must inherit from Base");

	// 1. Construct Derived on device
	DeviceObject<Derived> dDerived = makeDeviceObject<Derived>();

	// 2. Move pointer into a DeviceObject<Base> using your converting ctor
	DeviceObject<Base> dBase(std::move(dDerived));
	return dBase;
}

}  // namespace yrt

#endif  // BUILD_CUDA
