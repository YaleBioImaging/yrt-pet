/*
* This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/GPUTypes.cuh"

namespace yrt
{

void synchronizeIfNeeded(const GPULaunchConfig& config)
{
	if (config.synchronize)
	{
		if (config.stream != nullptr)
		{
			cudaStreamSynchronize(*config.stream);
		}
		else
		{
			cudaDeviceSynchronize();
		}
	}
}

}  // namespace yrt
