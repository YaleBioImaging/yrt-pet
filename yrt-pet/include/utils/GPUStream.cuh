/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cuda_runtime_api.h>

class GPUStream
{
public:
	explicit GPUStream(unsigned int flags=cudaStreamNonBlocking);
	~GPUStream();
	const cudaStream_t& getStream() const;
private:
	cudaStream_t m_stream;
};
