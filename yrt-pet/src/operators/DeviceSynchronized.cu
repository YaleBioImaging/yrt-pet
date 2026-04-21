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

CUScannerParams DeviceSynchronized::getCUScannerParams(const Scanner& scanner)
{
	CUScannerParams params;
	params.crystalSize_trans = scanner.crystalSize_trans;
	params.crystalSize_z = scanner.crystalSize_z;
	params.numDets = scanner.getNumDets();
	return params;
}

CUImageParams DeviceSynchronized::getCUImageParams(const ImageParams& imgParams)
{
	CUImageParams params;

	params.nx = imgParams.nx;
	params.ny = imgParams.ny;
	params.nz = imgParams.nz;

	params.length_x = imgParams.length_x;
	params.length_y = imgParams.length_y;
	params.length_z = imgParams.length_z;

	params.vx = imgParams.vx;
	params.vy = imgParams.vy;
	params.vz = imgParams.vz;

	params.off_x = imgParams.off_x;
	params.off_y = imgParams.off_y;
	params.off_z = imgParams.off_z;

	params.fovRadius = imgParams.fovRadius;

	return params;
}

CUImage DeviceSynchronized::getCUImage(ImageDevice& img)
{
	return {getCUImageParams(img.getParams()), img.getDevicePointer()};
}

DeviceSynchronized::DeviceSynchronized(const cudaStream_t* pp_mainStream,
                                       const cudaStream_t* pp_auxStream)
{
	mp_mainStream = pp_mainStream;
	mp_auxStream = pp_auxStream;
}

}  // namespace yrt
