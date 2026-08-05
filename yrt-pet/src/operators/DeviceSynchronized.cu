/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/DeviceSynchronized.cuh"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"
#include "yrt-pet/utils/Globals.hpp"


namespace yrt
{
namespace util
{

GPULaunchParams3D initiateDeviceParameters(const ImageParams& params)
{
	GPULaunchParams3D launchParams;
	if (params.nz * params.nt > 1)
	{
		const size_t threadsPerBlockDimImage = globals::ThreadsPerBlockImg3d;
		const auto threadsPerBlockDimImage_float =
		    static_cast<float>(threadsPerBlockDimImage);
		const auto threadsPerBlockDimImage_uint =
		    static_cast<unsigned int>(threadsPerBlockDimImage);

		launchParams.gridSize = {
		    static_cast<unsigned int>(
		        std::ceil(params.nx / threadsPerBlockDimImage_float)),
		    static_cast<unsigned int>(
		        std::ceil(params.ny / threadsPerBlockDimImage_float)),
		    static_cast<unsigned int>(std::ceil(
		        params.nz * params.nt / threadsPerBlockDimImage_float))};

		launchParams.blockSize = {threadsPerBlockDimImage_uint,
		                          threadsPerBlockDimImage_uint,
		                          threadsPerBlockDimImage_uint};
	}
	else
	{
		const size_t threadsPerBlockDimImage = globals::ThreadsPerBlockImg2d;
		const auto threadsPerBlockDimImage_float =
		    static_cast<float>(threadsPerBlockDimImage);
		const auto threadsPerBlockDimImage_uint =
		    static_cast<unsigned int>(threadsPerBlockDimImage);

		launchParams.gridSize = {
		    static_cast<unsigned int>(
		        std::ceil(params.nx / threadsPerBlockDimImage_float)),
		    static_cast<unsigned int>(
		        std::ceil(params.ny / threadsPerBlockDimImage_float)),
		    1};

		launchParams.blockSize = {threadsPerBlockDimImage_uint,
		                          threadsPerBlockDimImage_uint, 1};
	}
	return launchParams;
}

GPULaunchParams initiateDeviceParameters(size_t batchSize)
{
	GPULaunchParams launchParams{};
	launchParams.gridSize = static_cast<unsigned int>(std::ceil(
	    batchSize / static_cast<float>(globals::ThreadsPerBlockData)));
	launchParams.blockSize = globals::ThreadsPerBlockData;
	return launchParams;
}

}  // namespace util

const cudaStream_t* DeviceSynchronized::getMainStream() const
{
	return mp_mainStream;
}

const cudaStream_t* DeviceSynchronized::getAuxStream() const
{
	return mp_auxStream;
}

DeviceSynchronized::DeviceSynchronized(const cudaStream_t* pp_mainStream,
                                       const cudaStream_t* pp_auxStream)
{
	mp_mainStream = pp_mainStream;
	mp_auxStream = pp_auxStream;
}

}  // namespace yrt
