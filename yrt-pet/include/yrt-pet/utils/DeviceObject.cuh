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
__global__ void __yrt_construct_on_device(void* mem, Args... args) {
	// Constructs T at address mem using placement-new.
	new (mem) T(std::forward<Args>(args)...);
}

template <typename T>
__global__ void __yrt_destroy_on_device(T* obj) {
	obj->~T();
}


template <typename T>
class DeviceObject
{
public:
	DeviceObject() = delete;

	template <typename... Args>
	explicit DeviceObject(Args&&... args)
		: mpd_object(nullptr)
	{
		// 1) Allocate raw storage on device
		util::allocateDevice(&mpd_object, 1, {});

		// 2) Construct the object *on device* with placement-new
		__yrt_construct_on_device<T, Args...>
			<<<1, 1>>>(mpd_object, std::forward<Args>(args)...);

		cudaDeviceSynchronize();
		ASSERT(cudaCheckError());
	}

	~DeviceObject() {
		if (mpd_object) {
			// Call device-side destructor, then free storage
			__yrt_destroy_on_device<T><<<1, 1>>>(reinterpret_cast<T*>(mpd_object));
			// Best-effort; avoid throwing from destructors
			cudaDeviceSynchronize();
			util::deallocateDevice(mpd_object, {});
		}
	}

	DeviceObject(const DeviceObject&)            = delete;
	DeviceObject& operator=(const DeviceObject&) = delete;

	DeviceObject(DeviceObject&& other) noexcept : mpd_object(other.mpd_object) {
		other.mpd_object = nullptr;
	}
	DeviceObject& operator=(DeviceObject&& other) noexcept {
		if (this != &other) {
			this->~DeviceObject();
			mpd_object = other.mpd_object;
			other.mpd_object = nullptr;
		}
		return *this;
	}

	const T* getDevicePointer() const { return mpd_object; }
	T* getDevicePointer() { return mpd_object; }

private:
	T* mpd_object;
};

}  // namespace yrt

#endif // BUILD_CUDA