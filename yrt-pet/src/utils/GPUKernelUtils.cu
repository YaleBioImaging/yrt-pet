/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/GPUKernelUtils.cuh"

namespace yrt
{
__global__ void applyAttenuationFactors_kernel(const float* pd_attImgProjData,
                                               const float* pd_imgProjData,
                                               float* pd_destProjData,
                                               float unitFactor,
                                               const size_t maxNumberOfEvents)
{
	const long eventId = blockIdx.x * blockDim.x + threadIdx.x;
	if (eventId < maxNumberOfEvents)
	{
		const float attProj = pd_attImgProjData[eventId];
		pd_destProjData[eventId] =
		    pd_imgProjData[eventId] * exp(-attProj * unitFactor);
	}
}
}  // namespace yrt
