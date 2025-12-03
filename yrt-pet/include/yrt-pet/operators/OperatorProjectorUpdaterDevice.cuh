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

class OperatorProjectorUpdaterDevice
{
public:
	__device__ OperatorProjectorUpdaterDevice() {}
	__device__ virtual ~OperatorProjectorUpdaterDevice() {}

	__device__ virtual float forwardUpdate(float weight, float* cur_img_ptr,
	                                       size_t offset, frame_t dynamicFrame,
	                                       size_t numVoxelPerFrame) const = 0;
	__device__ virtual void backUpdate(float value, float weight,
	                                   float* cur_img_ptr, size_t offset,
	                                   frame_t dynamicFrame,
	                                   size_t numVoxelPerFrame) = 0;
};


class OperatorProjectorUpdaterDeviceDefault3D
    : public OperatorProjectorUpdaterDevice
{
public:
	__device__ OperatorProjectorUpdaterDeviceDefault3D() {}

	__device__ float forwardUpdate(float weight, float* cur_img_ptr,
	                               size_t offset, frame_t dynamicFrame,
	                               size_t numVoxelPerFrame) const override

	{
		return weight * cur_img_ptr[offset];
	}

	__device__ void backUpdate(float value, float weight, float* cur_img_ptr,
	                           size_t offset, frame_t dynamicFrame,
	                           size_t numVoxelPerFrame) override
	{
		float output = value * weight;
		float* ptr = &cur_img_ptr[offset];
		atomicAdd(ptr, output);
	}
};

class OperatorProjectorUpdaterDeviceDefault4D
    : public OperatorProjectorUpdaterDevice
{
public:
	__device__ OperatorProjectorUpdaterDeviceDefault4D() {}

	__device__ float forwardUpdate(float weight, float* cur_img_ptr,
	                               size_t offset, frame_t dynamicFrame,
	                               size_t numVoxelPerFrame) const override
	{
		return weight * cur_img_ptr[dynamicFrame * numVoxelPerFrame + offset];
	}

	__device__ void backUpdate(float value, float weight, float* cur_img_ptr,
	                           size_t offset, frame_t dynamicFrame,
	                           size_t numVoxelPerFrame) override
	{
		float output = value * weight;
		atomicAdd(cur_img_ptr + dynamicFrame * numVoxelPerFrame + offset,
		          output);
	}
};

template <int Rank>
class OperatorProjectorUpdaterDeviceLRUnrolled
    : public OperatorProjectorUpdaterDevice
{
public:
	__device__ OperatorProjectorUpdaterDeviceLRUnrolled(float* d_HBasis,
	                                                    float* d_HBasisWrite,
	                                                    int numFrames,
	                                                    bool updateH)
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
		const float* H_ptr = mpd_HBasisDevice_ptr;

#pragma unroll
		for (int l = 0; l < Rank; ++l)
		{
			const float cur_H_ptr =
			    *(H_ptr + l * m_numDynamicFrames + dynamicFrame);
			const size_t offset_rank = l * numVoxelPerFrame;
			cur_img_lr_val += cur_img_ptr[offset + offset_rank] * cur_H_ptr;
		}
		return weight * cur_img_lr_val;
	}

	__device__ void backUpdate(float value, float weight, float* cur_img_ptr,
	                           size_t offset, frame_t dynamicFrame,
	                           size_t numVoxelPerFrame) override
	{
		const float Ay = value * weight;

		if (!m_updateH)
		{
			const float* H_ptr = mpd_HBasisDevice_ptr;
#pragma unroll
			for (int l = 0; l < Rank; ++l)
			{
				const float cur_H_ptr =
				    *(H_ptr + l * m_numDynamicFrames + dynamicFrame);
				const size_t offset_rank = l * numVoxelPerFrame;
				const float output = Ay * cur_H_ptr;
				atomicAdd(cur_img_ptr + offset + offset_rank, output);
			}
		}
		else
		{
			float* H_ptr = mpd_HBasisDeviceWrite_ptr;
#pragma unroll
			for (int l = 0; l < Rank; ++l)
			{
				const size_t offset_rank = l * numVoxelPerFrame;
				const float output = Ay * cur_img_ptr[offset + offset_rank];
				atomicAdd(H_ptr + l * m_numDynamicFrames + dynamicFrame,
				          output);
			}
		}
	}

protected:
	float* mpd_HBasisDevice_ptr;       // used by forward/backward (read-only)
	float* mpd_HBasisDeviceWrite_ptr;  // used by backward (write-only)
	bool m_updateH = false;
	int m_numDynamicFrames = 1;
};

class OperatorProjectorUpdaterDeviceLR : public OperatorProjectorUpdaterDevice
{
public:
	// We optimize for ranks 8, 10, 12, 14, 16, 18, 20
	static constexpr int MaxRankForUnrolled = 20;
	static constexpr int MinRankForUnrolled = 8;
	static constexpr int RankStepForUnrolled = 2;
	__device__ OperatorProjectorUpdaterDeviceLR(float* d_HBasis,
	                                            float* d_HBasisWrite, int rank,
	                                            int numFrames, bool updateH)
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
		const float* __restrict__ H_ptr = mpd_HBasisDevice_ptr + dynamicFrame;
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
		const float AHy = value * weight;
		const float* __restrict__ H_ptr = mpd_HBasisDevice_ptr + dynamicFrame;
		float* __restrict__ img_ptr = cur_img_ptr + offset;

		if (!m_updateH)
		{
			for (int l = 0; l < m_rank; ++l)
			{
				atomicAdd(img_ptr, AHy * (*H_ptr));
				img_ptr += numVoxelPerFrame;
				H_ptr += m_numDynamicFrames;
			}
		}
		else
		{
			float* H_ptr = mpd_HBasisDeviceWrite_ptr;
			for (int l = 0; l < m_rank; ++l)
			{
				const size_t offset_rank = l * numVoxelPerFrame;
				const float output = AHy * cur_img_ptr[offset + offset_rank];
				atomicAdd(H_ptr + l * m_numDynamicFrames + dynamicFrame,
						  output);
			}
		}
	}

protected:
	float* mpd_HBasisDevice_ptr;       // used by forward/backward (read-only)
	float* mpd_HBasisDeviceWrite_ptr;  // used by backward (write-only)
	bool m_updateH = false;
	int m_rank = 1;
	int m_numDynamicFrames = 1;
};

template <int MaxRank = OperatorProjectorUpdaterDeviceLR::MaxRankForUnrolled>
__device__ OperatorProjectorUpdaterDevice*
    dispatchCreateUpdaterLR(int rank, float* d_HBasis, float* d_HBasisWrite,
                            int numFrames, bool updateH)
{
	static_assert(MaxRank <=
	              OperatorProjectorUpdaterDeviceLR::MaxRankForUnrolled);
	static_assert(MaxRank > 0);

	if (rank == MaxRank)
	{
		return new OperatorProjectorUpdaterDeviceLRUnrolled<MaxRank>(
		    d_HBasis, d_HBasisWrite, numFrames, updateH);
	}
	if constexpr (MaxRank >=
	              OperatorProjectorUpdaterDeviceLR::MinRankForUnrolled)
	{
		// Compile-time recursion: generates a chain of ifs, no runtime
		// recursion
		return dispatchCreateUpdaterLR<
		    MaxRank - OperatorProjectorUpdaterDeviceLR::RankStepForUnrolled>(
		    rank, d_HBasis, d_HBasisWrite, numFrames, updateH);
	}
	else
	{
		return new OperatorProjectorUpdaterDeviceLR(d_HBasis, d_HBasisWrite,
		                                            rank, numFrames, updateH);
	}
}

using UpdaterPointer = OperatorProjectorUpdaterDevice**;

inline __global__ void constructUpdaterOnDevice(
    UpdaterPointer ppd_updater,
    const OperatorProjectorParams::ProjectorUpdaterType p_updaterType,
    float* HBasis_ptr, float* HBasisWrite_ptr, int rank, int numFrames,
    bool updateH)
{
	// It is necessary to create object representing a function
	// directly in global memory of the GPU device for virtual
	// functions to work correctly, i.e. virtual function table
	// HAS to be on GPU as well.
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		if (p_updaterType ==
		    OperatorProjectorParams::ProjectorUpdaterType::DEFAULT3D)
		{
			*ppd_updater = new OperatorProjectorUpdaterDeviceDefault3D();
		}
		else if (p_updaterType ==
		         OperatorProjectorParams::ProjectorUpdaterType::DEFAULT4D)
		{
			*ppd_updater = new OperatorProjectorUpdaterDeviceDefault4D();
		}
		else if (p_updaterType ==
		         OperatorProjectorParams::ProjectorUpdaterType::LR)
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

class OperatorProjectorUpdaterDeviceWrapper
{
public:
	OperatorProjectorUpdaterDeviceWrapper() : mpd_updater(nullptr) {}

	~OperatorProjectorUpdaterDeviceWrapper()
	{
		if (mpd_updater != nullptr)
		{
			destroyUpdaterOnDevice<<<1, 1>>>(mpd_updater);
		}
	}

	void initUpdater(
	    const OperatorProjectorParams::ProjectorUpdaterType updaterType)
	{
		cudaMalloc(&mpd_updater, sizeof(UpdaterPointer));
		constructUpdaterOnDevice<<<1, 1>>>(mpd_updater, updaterType, nullptr,
		                                   nullptr, 0, 0, false);
		cudaDeviceSynchronize();
	}

	void initUpdater(
	    const OperatorProjectorParams::ProjectorUpdaterType updaterType,
	    const Array2DBase<float>& p_HBasis, bool updateH = false)
	{
		float* HBasisDevice_ptr = nullptr;
		float* HBasisWriteDevice_ptr = nullptr;
		if (updaterType == OperatorProjectorParams::ProjectorUpdaterType::LR ||
		    updaterType ==
		        OperatorProjectorParams::ProjectorUpdaterType::LRDUALUPDATE)
		{
			if (p_HBasis.getSizeTotal() == 0)
			{
				throw std::invalid_argument(
				    "LR updater was requested but HBasis is empty");
			}
			mpd_HBasisDeviceArray = std::make_unique<DeviceArray<float>>();
			mpd_HBasisWriteDeviceArray = std::make_unique<DeviceArray<float>>();
			m_updateH = updateH;
			setHBasis(p_HBasis);
			HBasisDevice_ptr = mpd_HBasisDeviceArray->getDevicePointer();
			HBasisWriteDevice_ptr =
			    mpd_HBasisWriteDeviceArray->getDevicePointer();
		}
		cudaMalloc(&mpd_updater, sizeof(UpdaterPointer));
		constructUpdaterOnDevice<<<1, 1>>>(
		    mpd_updater, updaterType, HBasisDevice_ptr, HBasisWriteDevice_ptr,
		    m_rank, m_numDynamicFrames, m_updateH);
		cudaDeviceSynchronize();
	}

	UpdaterPointer getUpdaterDevicePointer()
	{
		if (mpd_updater == nullptr)
		{
			return nullptr;
		}
		return mpd_updater;
	}

	bool allocateForHBasisDevice() const
	{
		// Allocate HBasis buffers
		const GPULaunchConfig launchConfig{nullptr, true};
		return mpd_HBasisDeviceArray->allocate(m_rank * m_numDynamicFrames,
		                                       launchConfig);
	}

	bool allocateForHBasisWriteDevice() const
	{
		// Allocate HBasis buffers
		const GPULaunchConfig launchConfig{nullptr, true};
		return mpd_HBasisWriteDeviceArray->allocate(m_rank * m_numDynamicFrames,
		                                            launchConfig);
	}


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

	void SyncHostToDeviceHBasisWrite()
	{
		const GPULaunchConfig launchConfig{nullptr, true};
		if (!mpd_HBasisWriteDeviceArray->isAllocated())
		{
			allocateForHBasisWriteDevice();
		}
		auto HBasisWrite_ptr = mp_HWrite.getRawPointer();
		mpd_HBasisWriteDeviceArray->copyFromHost(
		    HBasisWrite_ptr, m_rank * m_numDynamicFrames, launchConfig);
	}

	void SyncDeviceToHostHBasisWrite()
	{
		const GPULaunchConfig launchConfig{nullptr, true};

		if (!mpd_HBasisWriteDeviceArray->isAllocated())
		{
			allocateForHBasisWriteDevice();
		}
		auto HBasisWrite_ptr = mp_HWrite.getRawPointer();
		mpd_HBasisWriteDeviceArray->copyToHost(
		    HBasisWrite_ptr, m_rank * m_numDynamicFrames, launchConfig);
	}

	void setHBasis(const Array2DBase<float>& pr_HBasis)
	{
		// initialize host side mp_HBasis and class members
		mp_HBasis.bind(pr_HBasis);
		auto HBasis_ptr = pr_HBasis.getRawPointer();
		auto dims = pr_HBasis.getDims();
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

	std::unique_ptr<Array2D<float>> getHBasisCopy()
	{
		SyncDeviceToHostHBasis();
		auto dims = mp_HBasis.getDims();
		auto out = std::make_unique<Array2D<float>>();
		out->allocate(dims[0], dims[1]);
		out->copy(mp_HBasis);
		return out;
	}

	void setUpdateH(bool updateH) { m_updateH = updateH; }

	bool getUpdateH() const { return m_updateH; }

	void setHBasisWrite(const Array2DBase<float>& pr_HWrite)
	{
		mp_HWrite.bind(pr_HWrite);
		SyncHostToDeviceHBasisWrite();
		// TODO : use multithread with one H per thread in gpu ?
		// initializeWriteThread();
	}

	const Array2DAlias<float>& getHBasisWrite()
	{
		SyncDeviceToHostHBasisWrite();
		return mp_HWrite;
	}


private:
	UpdaterPointer mpd_updater;
	std::unique_ptr<DeviceArray<float>> mpd_HBasisDeviceArray;
	std::unique_ptr<DeviceArray<float>> mpd_HBasisWriteDeviceArray;
	Array2DAlias<float> mp_HBasis;
	Array2DAlias<float> mp_HWrite;
	int m_updateH = false;
	int m_rank = 1;
	int m_numDynamicFrames = 1;
};

}  // namespace yrt
