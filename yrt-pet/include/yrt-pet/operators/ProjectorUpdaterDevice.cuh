/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/utils/Types.hpp"

#include <cuda_runtime.h>

namespace yrt
{

class ProjectorUpdaterDevice
{
public:
	__device__ ProjectorUpdaterDevice() {}
	__device__ virtual ~ProjectorUpdaterDevice() {}

	__device__ virtual float forwardUpdate(float weight, float* cur_img_ptr,
	                                       size_t offset, frame_t dynamicFrame,
	                                       size_t numVoxelPerFrame) const = 0;
	__device__ virtual void backUpdate(float value, float weight,
	                                   float* cur_img_ptr, size_t offset,
	                                   frame_t dynamicFrame,
	                                   size_t numVoxelPerFrame) = 0;
};

class ProjectorUpdaterDeviceDefault4D : public ProjectorUpdaterDevice
{
public:
	__device__ ProjectorUpdaterDeviceDefault4D() {}

	__device__ float forwardUpdate(float weight, float* cur_img_ptr,
	                               size_t offset, frame_t dynamicFrame,
	                               size_t numVoxelPerFrame) const override
	{
		return weight * cur_img_ptr[dynamicFrame * numVoxelPerFrame + offset];
	}

	__device__ void backUpdate(float value, float weight, float* cur_img_ptr,
	                           size_t offset, frame_t dynamicFrame,
	                           size_t numVoxelsPerFrame) override
	{
		const float output = value * weight;
		atomicAdd(cur_img_ptr + dynamicFrame * numVoxelsPerFrame + offset,
		          output);
	}
};

template <int Rank>
class ProjectorUpdaterDeviceLRUnrolled : public ProjectorUpdaterDevice
{
public:
	__device__ ProjectorUpdaterDeviceLRUnrolled(float* d_HBasis,
	                                            double* d_HBasisWrite,
	                                            int numFrames, bool updateH)
	    : mpd_HBasisDevice_ptr(d_HBasis),
	      mpd_HBasisDeviceWrite_ptr(d_HBasisWrite),
	      m_updateH(updateH),
	      m_numDynamicFrames(numFrames)
	{
	}

	__device__ float forwardUpdate(float weight, float* cur_img_ptr,
	                               size_t offset, frame_t dynamicFrame,
	                               size_t numVoxelPerFrame) const override
	{
		float cur_img_lr_val = 0.0f;
		const float* __restrict__ H_ptr = mpd_HBasisDevice_ptr + dynamicFrame;
		const float* __restrict__ img_ptr = cur_img_ptr + offset;
#pragma unroll
		for (int l = 0; l < Rank; ++l)
		{
			cur_img_lr_val = fmaf((*img_ptr), (*H_ptr), cur_img_lr_val);
			img_ptr += numVoxelPerFrame;
			H_ptr += m_numDynamicFrames;
		}
		return weight * cur_img_lr_val;
	}

	__device__ void backUpdate(float value, float weight, float* cur_img_ptr,
	                           size_t offset, frame_t dynamicFrame,
	                           size_t numVoxelPerFrame) override
	{
		const float AHy = value * weight;

		if (!m_updateH)
		{
			float* __restrict__ img_ptr = cur_img_ptr + offset;
			const float* __restrict__ H_ptr =
			    mpd_HBasisDevice_ptr + dynamicFrame;
#pragma unroll
			for (int l = 0; l < Rank; ++l)
			{
				atomicAdd(img_ptr, AHy * (*H_ptr));
				img_ptr += numVoxelPerFrame;
				H_ptr += m_numDynamicFrames;
			}
		}
		else
		{
			const float* __restrict__ img_ptr = cur_img_ptr + offset;
			auto* __restrict__ H_ptr = mpd_HBasisDeviceWrite_ptr + dynamicFrame;
#pragma unroll
			for (int l = 0; l < Rank; ++l)
			{
				atomicAdd(H_ptr, static_cast<double>(AHy * (*img_ptr)));
				img_ptr += numVoxelPerFrame;
				H_ptr += m_numDynamicFrames;
			}
		}
	}

protected:
	float* mpd_HBasisDevice_ptr;        // used by forward/backward (read-only)
	double* mpd_HBasisDeviceWrite_ptr;  // used by backward (write-only)
	bool m_updateH = false;
	int m_numDynamicFrames = 1;
};

class ProjectorUpdaterDeviceLR : public ProjectorUpdaterDevice
{
public:
	// We optimize for ranks 8, 10, 12, 14, 16, 18, 20
	static constexpr int MaxRankForUnrolled = 20;
	static constexpr int MinRankForUnrolled = 8;
	static constexpr int RankStepForUnrolled = 2;
	__device__ ProjectorUpdaterDeviceLR(float* d_HBasis, double* d_HBasisWrite,
	                                    int rank, int numFrames, bool updateH)
	    : mpd_HBasisDevice_ptr(d_HBasis),
	      mpd_HBasisDeviceWrite_ptr(d_HBasisWrite),
	      m_updateH(updateH),
	      m_rank(rank),
	      m_numDynamicFrames(numFrames)
	{
	}

	__device__ float forwardUpdate(float weight,
	                               float* __restrict__ cur_img_ptr,
	                               size_t offset, frame_t dynamicFrame,
	                               size_t numVoxelPerFrame) const override
	{
		float cur_img_lr_val = 0.0f;
		const auto* __restrict__ H_ptr = mpd_HBasisDevice_ptr + dynamicFrame;
		const float* __restrict__ img_ptr = cur_img_ptr + offset;

		for (int l = 0; l < m_rank; ++l)
		{
			cur_img_lr_val = fmaf((*img_ptr), (*H_ptr), cur_img_lr_val);
			img_ptr += numVoxelPerFrame;
			H_ptr += m_numDynamicFrames;
		}
		return weight * cur_img_lr_val;
	}

	__device__ void backUpdate(float value, float weight,
	                           float* __restrict__ cur_img_ptr, size_t offset,
	                           frame_t dynamicFrame,
	                           size_t numVoxelPerFrame) override
	{
		const double AHy = value * weight;

		if (!m_updateH)
		{
			float* __restrict__ img_ptr = cur_img_ptr + offset;
			const auto* __restrict__ H_ptr =
			    mpd_HBasisDevice_ptr + dynamicFrame;
			for (int l = 0; l < m_rank; ++l)
			{
				atomicAdd(img_ptr, AHy * (*H_ptr));
				img_ptr += numVoxelPerFrame;
				H_ptr += m_numDynamicFrames;
			}
		}
		else
		{
			const float* __restrict__ img_ptr = cur_img_ptr + offset;
			auto* __restrict__ H_ptr = mpd_HBasisDeviceWrite_ptr + dynamicFrame;
			for (int l = 0; l < m_rank; ++l)
			{
				atomicAdd(H_ptr, AHy * (*img_ptr));
				img_ptr += numVoxelPerFrame;
				H_ptr += m_numDynamicFrames;
			}
		}
	}

protected:
	float* mpd_HBasisDevice_ptr;        // used by forward/backward (read-only)
	double* mpd_HBasisDeviceWrite_ptr;  // used by backward (write-only)
	bool m_updateH = false;
	int m_rank = 1;
	int m_numDynamicFrames = 1;
};

template <int MaxRank = ProjectorUpdaterDeviceLR::MaxRankForUnrolled>
__device__ ProjectorUpdaterDevice*
    dispatchCreateUpdaterLR(int rank, float* d_HBasis, double* d_HBasisWrite,
                            int numFrames, bool updateH)
{
	static_assert(MaxRank <= ProjectorUpdaterDeviceLR::MaxRankForUnrolled);
	static_assert(MaxRank > 0);

	if (rank == MaxRank)
	{
		return new ProjectorUpdaterDeviceLRUnrolled<MaxRank>(
		    d_HBasis, d_HBasisWrite, numFrames, updateH);
	}
	if constexpr (MaxRank >= ProjectorUpdaterDeviceLR::MinRankForUnrolled)
	{
		// Compile-time recursion: generates a chain of ifs, no runtime
		// recursion
		return dispatchCreateUpdaterLR<
		    MaxRank - ProjectorUpdaterDeviceLR::RankStepForUnrolled>(
		    rank, d_HBasis, d_HBasisWrite, numFrames, updateH);
	}
	else
	{
		return new ProjectorUpdaterDeviceLR(d_HBasis, d_HBasisWrite, rank,
		                                    numFrames, updateH);
	}
}

using UpdaterPointer = ProjectorUpdaterDevice**;

inline __global__ void constructUpdaterOnDevice(UpdaterPointer ppd_updater,
                                                const UpdaterType p_updaterType,
                                                float* HBasis_ptr,
                                                double* HBasisWrite_ptr,
                                                int rank, int numFrames,
                                                bool updateH)
{
	// It is necessary to create object representing a function
	// directly in global memory of the GPU device for virtual
	// functions to work correctly, i.e. virtual function table
	// HAS to be on GPU as well.
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		if (p_updaterType == UpdaterType::DEFAULT4D)
		{
			*ppd_updater = new ProjectorUpdaterDeviceDefault4D();
		}
		else if (p_updaterType == UpdaterType::LR)
		{
			*ppd_updater = dispatchCreateUpdaterLR(
			    rank, HBasis_ptr, HBasisWrite_ptr, numFrames, updateH);
		}
		else
		{
			// TODO : Have the LRDUALUPDATE instantiated here once the class
			//  exists
		}
	}
}

inline __global__ void destroyUpdaterOnDevice(UpdaterPointer ppd_updater)
{
	delete *ppd_updater;
}

class ProjectorUpdaterDeviceWrapper
{
public:
	ProjectorUpdaterDeviceWrapper()
	    : m_updaterType(UpdaterType::DEFAULT4D), mpd_updater(nullptr)
	{
	}

	~ProjectorUpdaterDeviceWrapper()
	{
		if (mpd_updater != nullptr)
		{
			destroyUpdaterOnDevice<<<1, 1>>>(mpd_updater);
		}
	}

	void initUpdater(const UpdaterType updaterType)
	{
		m_updaterType = updaterType;
		cudaMalloc(&mpd_updater, sizeof(UpdaterPointer));
		constructUpdaterOnDevice<<<1, 1>>>(mpd_updater, updaterType, nullptr,
		                                   nullptr, 0, 0, false);
		cudaDeviceSynchronize();
	}

	void initUpdater(const UpdaterType updaterType,
	                 const Array2DBase<float>& p_HBasis, bool updateH = false)
	{
		m_updaterType = updaterType;
		float* HBasisDevice_ptr = nullptr;
		double* HBasisWriteDevice_ptr = nullptr;
		if (updaterType == UpdaterType::LR ||
		    updaterType == UpdaterType::LRDUALUPDATE)
		{
			if (p_HBasis.getSizeTotal() == 0)
			{
				throw std::invalid_argument(
				    "LR updater was requested but HBasis is empty");
			}

			mpd_HBasisDeviceArray = std::make_unique<DeviceArray<float>>();
			mpd_HBasisWriteDeviceArray =
			    std::make_unique<DeviceArray<double>>();
			m_updateH = updateH;
			setHBasis(p_HBasis);
			HBasisDevice_ptr = mpd_HBasisDeviceArray->getDevicePointer();
			allocateForHBasisWriteDevice();
			HBasisWriteDevice_ptr =
			    mpd_HBasisWriteDeviceArray->getDevicePointer();
		}
		cudaMalloc(&mpd_updater, sizeof(UpdaterPointer));
		constructUpdaterOnDevice<<<1, 1>>>(
		    mpd_updater, updaterType, HBasisDevice_ptr, HBasisWriteDevice_ptr,
		    m_rank, m_numDynamicFrames, m_updateH);
		cudaDeviceSynchronize();
	}

	UpdaterType getUpdaterType() const { return m_updaterType; }

	UpdaterPointer getUpdaterDevicePointer() { return mpd_updater; }

	bool isUpdaterInit() const { return mpd_updater != nullptr; }

	void allocateForHBasisDevice() const
	{
		// Allocate HBasis buffers
		const GPULaunchConfig launchConfig{nullptr, true};
		mpd_HBasisDeviceArray->allocate(m_rank * m_numDynamicFrames,
		                                launchConfig);
	}

	void allocateForHBasisWriteDevice() const
	{
		// Allocate HBasis buffers
		const GPULaunchConfig launchConfig{nullptr, true};
		mpd_HBasisWriteDeviceArray->allocate(m_rank * m_numDynamicFrames,
		                                     launchConfig);
	}

	// TODO NOW: Fix capitalization
	void SyncHostToDeviceHBasis()
	{
		const GPULaunchConfig launchConfig{nullptr, true};
		auto HBasis_ptr = mp_HBasis.getRawPointer();
		mpd_HBasisDeviceArray->copyFromHost(
		    HBasis_ptr, m_rank * m_numDynamicFrames, launchConfig);
	}

	void SyncDeviceToHostHBasis()
	{
		const GPULaunchConfig launchConfig{nullptr, true};
		auto HBasis_ptr = mp_HBasis.getRawPointer();
		mpd_HBasisDeviceArray->copyToHost(
		    HBasis_ptr, m_rank * m_numDynamicFrames, launchConfig);
	}

	template <typename TSrc, typename TDest>
	void copyAndConvert(TDest* p_dest, TSrc* p_src, size_t p_numElems)
	{
		util::parallelForChunked(
		    p_numElems, globals::numThreads(),
		    [p_dest, p_src](size_t binIdx, int tid)
		    { p_dest[binIdx] = static_cast<TDest>(p_src[binIdx]); });
	}

	void syncHostToDeviceHBasisWrite()
	{
		const GPULaunchConfig launchConfig{nullptr, true};
		if (!mpd_HBasisWriteDeviceArray->isAllocated())
		{
			allocateForHBasisWriteDevice();
		}
		auto HBasisWrite_ptr = mp_HWrite.getRawPointer();
		auto HBasisWriteDouble_ptr =
		    std::make_unique<double[]>(m_rank * m_numDynamicFrames);
		copyAndConvert(HBasisWriteDouble_ptr.get(), HBasisWrite_ptr,
		               m_rank * m_numDynamicFrames);

		mpd_HBasisWriteDeviceArray->copyFromHost(HBasisWriteDouble_ptr.get(),
		                                         m_rank * m_numDynamicFrames,
		                                         launchConfig);
		cudaDeviceSynchronize();
	}

	void syncDeviceToHostHBasisWrite()
	{
		const GPULaunchConfig launchConfig{nullptr, true};

		if (mp_HWrite.getSizeTotal() != m_rank * m_numDynamicFrames)
		{
			throw std::logic_error(
			    "Host mp_HWrite dimension does not match Device side.");
		}
		auto HBasisWrite_ptr = mp_HWrite.getRawPointer();
		auto HBasisWriteDouble_ptr =
		    std::make_unique<double[]>(m_rank * m_numDynamicFrames);

		mpd_HBasisWriteDeviceArray->copyToHost(HBasisWriteDouble_ptr.get(),
		                                       m_rank * m_numDynamicFrames,
		                                       launchConfig);
		cudaDeviceSynchronize();
		copyAndConvert(HBasisWrite_ptr, HBasisWriteDouble_ptr.get(),
		               m_rank * m_numDynamicFrames);
	}

	void setHBasis(const Array2DBase<float>& pr_HBasis)
	{
		// initialize host side mp_HBasis and class members
		mp_HBasis.bind(pr_HBasis);

		const std::array<size_t, 2> dims = pr_HBasis.getDims();
		m_rank = static_cast<int>(dims[0]);
		m_numDynamicFrames = static_cast<int>(dims[1]);

		// initialize device side mpd_HBasisDeviceArray
		if (!mpd_HBasisDeviceArray->isAllocated())
		{
			allocateForHBasisDevice();
		}
		SyncHostToDeviceHBasis();
	}

	DeviceArray<float> getHBasisDevice() const
	{
		return *mpd_HBasisDeviceArray.get();
	}

	const Array2DAlias<float>& getHBasis()
	{
		SyncDeviceToHostHBasis();
		return mp_HBasis;
	}

	std::unique_ptr<Array2DOwned<float>> getHBasisCopy()
	{
		SyncDeviceToHostHBasis();
		auto dims = mp_HBasis.getDims();
		auto out = std::make_unique<Array2DOwned<float>>();
		out->allocate(dims[0], dims[1]);
		out->copy(mp_HBasis);
		return out;
	}

	void setUpdateH(bool updateH) { m_updateH = updateH; }

	bool getUpdateH() const { return m_updateH; }

	void setHBasisWrite(const Array2DBase<float>& pr_HWrite)
	{
		mp_HWrite.bind(pr_HWrite);
		// initialize device side mpd_HBasisWriteDeviceArray
		if (!mpd_HBasisWriteDeviceArray->isAllocated())
		{
			allocateForHBasisWriteDevice();
		}
		syncHostToDeviceHBasisWrite();
	}

	const Array2DAlias<float>& getHBasisWrite()
	{
		syncDeviceToHostHBasisWrite();
		return mp_HWrite;
	}


private:
	UpdaterType m_updaterType;
	UpdaterPointer mpd_updater;
	std::unique_ptr<DeviceArray<float>> mpd_HBasisDeviceArray;
	std::unique_ptr<DeviceArray<double>> mpd_HBasisWriteDeviceArray;
	Array2DAlias<float> mp_HBasis;
	Array2DAlias<float> mp_HWrite;
	int m_updateH = false;
	int m_rank = 1;
	int m_numDynamicFrames = 1;
};

}  // namespace yrt
