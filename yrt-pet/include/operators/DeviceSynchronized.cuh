/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "recon/CUParameters.hpp"
#include "utils/GPUTypes.cuh"

#include <cuda_runtime_api.h>


namespace Util
{
	// Takes a reference of an image, and automatically determines
	// the 3D grid/block parameters.
	GPULaunchParams3D initiateDeviceParameters(const ImageParams& params);
	// Takes a size of the data to be processed, and automatically determines
	// the grid/block parameters
	GPULaunchParams initiateDeviceParameters(size_t batchSize);
}  // namespace Util

class DeviceSynchronized
{
public:
	const cudaStream_t* getMainStream() const;
	const cudaStream_t* getAuxStream() const;

	static CUScannerParams getCUScannerParams(const Scanner& scanner);
	static CUImageParams getCUImageParams(const ImageParams& imgParams);

protected:
	explicit DeviceSynchronized(bool p_synchronized = true,
							const cudaStream_t* pp_mainStream = nullptr,
							const cudaStream_t* pp_auxStream = nullptr);

	bool m_synchronized;
	const cudaStream_t* mp_mainStream;
	const cudaStream_t* mp_auxStream;
};

