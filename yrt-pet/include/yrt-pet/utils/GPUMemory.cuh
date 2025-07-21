
#pragma once

#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/GPUTypes.cuh"
#include "yrt-pet/utils/GPUUtils.cuh"

namespace yrt
{
namespace util
{

template <typename T>
void allocateDevice(T** ppd_data, size_t p_numElems,
                    GPULaunchConfig p_launchConfig)
{
	if (p_launchConfig.stream != nullptr)
	{
		cudaMallocAsync(ppd_data, sizeof(T) * p_numElems,
		                *p_launchConfig.stream);
		if (p_launchConfig.synchronize)
		{
			cudaStreamSynchronize(*p_launchConfig.stream);
		}
	}
	else
	{
		cudaMalloc(ppd_data, sizeof(T) * p_numElems);
		if (p_launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	ASSERT(cudaCheckError());
}

template <typename T>
void deallocateDevice(T* ppd_data, GPULaunchConfig p_launchConfig)
{
	if (p_launchConfig.stream != nullptr)
	{
		cudaFreeAsync(ppd_data, *p_launchConfig.stream);
		if (p_launchConfig.synchronize)
		{
			cudaStreamSynchronize(*p_launchConfig.stream);
		}
	}
	else
	{
		cudaFree(ppd_data);
		if (p_launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	ASSERT(cudaCheckError());
}

template <typename T>
void copyHostToDevice(T* ppd_dest, const T* pph_src, size_t p_numElems,
                      GPULaunchConfig p_launchConfig)
{
	if (p_launchConfig.stream != nullptr)
	{
		cudaMemcpyAsync(ppd_dest, pph_src, p_numElems * sizeof(T),
		                cudaMemcpyHostToDevice, *p_launchConfig.stream);
		if (p_launchConfig.synchronize)
		{
			cudaStreamSynchronize(*p_launchConfig.stream);
		}
	}
	else
	{
		cudaMemcpy(ppd_dest, pph_src, p_numElems * sizeof(T),
		           cudaMemcpyHostToDevice);
		if (p_launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	ASSERT(cudaCheckError());
}

template <typename T>
void copyDeviceToHost(T* pph_dest, const T* ppd_src, size_t p_numElems,
                      GPULaunchConfig p_launchConfig)
{
	if (p_launchConfig.stream != nullptr)
	{
		cudaMemcpyAsync(pph_dest, ppd_src, p_numElems * sizeof(T),
		                cudaMemcpyDeviceToHost, *p_launchConfig.stream);
		if (p_launchConfig.synchronize)
		{
			cudaStreamSynchronize(*p_launchConfig.stream);
		}
	}
	else
	{
		cudaMemcpy(pph_dest, ppd_src, p_numElems * sizeof(T),
		           cudaMemcpyDeviceToHost);
		if (p_launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	ASSERT(cudaCheckError());
}

template <typename T>
void copyDeviceToDevice(T* ppd_dest, const T* ppd_src, size_t p_numElems,
                        GPULaunchConfig p_launchConfig)
{
	if (p_launchConfig.stream != nullptr)
	{
		cudaMemcpyAsync(ppd_dest, ppd_src, p_numElems * sizeof(T),
		                cudaMemcpyDeviceToDevice, *p_launchConfig.stream);
		if (p_launchConfig.synchronize)
		{
			cudaStreamSynchronize(*p_launchConfig.stream);
		}
	}
	else
	{
		cudaMemcpy(ppd_dest, ppd_src, p_numElems * sizeof(T),
		           cudaMemcpyDeviceToDevice);
		if (p_launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	ASSERT(cudaCheckError());
}

template <typename T>
void memsetDevice(T* ppd_data, int value, size_t p_numElems,
                  GPULaunchConfig p_launchConfig)
{
	if (p_launchConfig.stream != nullptr)
	{
		cudaMemsetAsync(ppd_data, value, sizeof(T) * p_numElems,
		                *p_launchConfig.stream);
		if (p_launchConfig.synchronize)
		{
			cudaStreamSynchronize(*p_launchConfig.stream);
		}
	}
	else
	{
		cudaMemset(ppd_data, value, sizeof(T) * p_numElems);
		if (p_launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	ASSERT(cudaCheckError());
}

}  // namespace util
}  // namespace yrt
