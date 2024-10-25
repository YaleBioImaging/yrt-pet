
#pragma once

#include "utils/Assert.hpp"
#include "utils/GPUUtils.cuh"

namespace Util
{
	template <typename T>
	void allocateDevice(T** ppd_data, size_t p_numElems,
	                    const cudaStream_t* pp_stream = nullptr,
	                    bool p_synchronize = true)
	{
		if (pp_stream != nullptr)
		{
			cudaMallocAsync(ppd_data, sizeof(T) * p_numElems, *pp_stream);
			if (p_synchronize)
			{
				cudaStreamSynchronize(*pp_stream);
			}
		}
		else
		{
			cudaMalloc(ppd_data, sizeof(T) * p_numElems);
			if (p_synchronize)
			{
				cudaDeviceSynchronize();
			}
		}
		ASSERT(cudaCheckError());
	}

	template <typename T>
	void deallocateDevice(T* ppd_data, const cudaStream_t* pp_stream = nullptr,
	                      bool p_synchronize = true)
	{
		if (pp_stream != nullptr)
		{
			cudaFreeAsync(ppd_data, *pp_stream);
			if (p_synchronize)
			{
				cudaStreamSynchronize(*pp_stream);
			}
		}
		else
		{
			cudaFree(ppd_data);
			if (p_synchronize)
			{
				cudaDeviceSynchronize();
			}
		}
		ASSERT(cudaCheckError());
	}

	template <typename T>
	void copyHostToDevice(T* ppd_dest, const T* pph_src, size_t p_numElems,
	                      const cudaStream_t* pp_stream = nullptr,
	                      bool p_synchronize = true)
	{
		if (pp_stream != nullptr)
		{
			cudaMemcpyAsync(ppd_dest, pph_src, p_numElems * sizeof(T),
			                cudaMemcpyHostToDevice, *pp_stream);
			if (p_synchronize)
			{
				cudaStreamSynchronize(*pp_stream);
			}
		}
		else
		{
			cudaMemcpy(ppd_dest, pph_src, p_numElems * sizeof(T),
			           cudaMemcpyHostToDevice);
			if (p_synchronize)
			{
				cudaDeviceSynchronize();
			}
		}
		ASSERT(cudaCheckError());
	}

	template <typename T>
	void copyDeviceToHost(T* pph_dest, const T* ppd_src, size_t p_numElems,
	                      const cudaStream_t* pp_stream = nullptr,
	                      bool p_synchronize = true)
	{
		if (pp_stream != nullptr)
		{
			cudaMemcpyAsync(pph_dest, ppd_src, p_numElems * sizeof(T),
			                cudaMemcpyDeviceToHost, *pp_stream);
			if (p_synchronize)
			{
				cudaStreamSynchronize(*pp_stream);
			}
		}
		else
		{
			cudaMemcpy(pph_dest, ppd_src, p_numElems * sizeof(T),
			           cudaMemcpyDeviceToHost);
			if (p_synchronize)
			{
				cudaDeviceSynchronize();
			}
		}
		ASSERT(cudaCheckError());
	}

	template <typename T>
	void copyDeviceToDevice(T* ppd_dest, const T* ppd_src, size_t p_numElems,
	                        const cudaStream_t* pp_stream = nullptr,
	                        bool p_synchronize = true)
	{
		if (pp_stream != nullptr)
		{
			cudaMemcpyAsync(ppd_dest, ppd_src, p_numElems * sizeof(T),
			                cudaMemcpyDeviceToDevice, *pp_stream);
			if (p_synchronize)
			{
				cudaStreamSynchronize(*pp_stream);
			}
		}
		else
		{
			cudaMemcpy(ppd_dest, ppd_src, p_numElems * sizeof(T),
			           cudaMemcpyDeviceToDevice);
			if (p_synchronize)
			{
				cudaDeviceSynchronize();
			}
		}
		ASSERT(cudaCheckError());
	}

	template <typename T>
	void memsetDevice(T* ppd_data, int value, size_t p_numElems,
	                  const cudaStream_t* pp_stream = nullptr,
	                  bool p_synchronize = true)
	{
		if (pp_stream != nullptr)
		{
			cudaMemsetAsync(ppd_data, value, sizeof(T) * p_numElems,
			                *pp_stream);
			if (p_synchronize)
			{
				cudaStreamSynchronize(*pp_stream);
			}
		}
		else
		{
			cudaMemset(ppd_data, value, sizeof(T) * p_numElems);
			if (p_synchronize)
			{
				cudaDeviceSynchronize();
			}
		}
		ASSERT(cudaCheckError());
	}

}  // namespace Util