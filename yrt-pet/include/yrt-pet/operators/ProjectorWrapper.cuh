/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/ProjectionPsfManagerDevice.cuh"
#include "yrt-pet/operators/SiddonKernels.cuh"

#include <cuda_runtime.h>

namespace yrt
{

// This is a wrapper which calls the appropriate projector depending on the
//  value of "projectorType". The arguments of this function are all the
//  arguments that would be provided to any projector call
template <bool IsForward, bool HasTOF, bool UseUpdater>
__device__ void
    projectAny(float& value, CUImage d_image, UpdaterPointer pd_updater,
               float3 p1, float3 p2, float3 n1, float3 n2, frame_t dynamicFrame,
               const TimeOfFlightHelper* pd_tofHelper, float tofValue,
               ProjectionPsfKernelStruct projPsfKernelStruct,
               CUScannerParams scannerParams, int numRays, size_t randomId,
               ProjectorType projectorType)
{
	const bool hasProjPsf = projPsfKernelStruct.kernels != nullptr;

	switch (projectorType)
	{
	case ProjectorType::SIDDON:
		if (numRays == 1)
		{
			// Single-ray Siddon
			projectSiddon<IsForward, HasTOF, true, false, UseUpdater>(
			    value, d_image.devicePointer, pd_updater, p1, p2, n1, n2,
			    dynamicFrame, pd_tofHelper, tofValue, scannerParams,
			    d_image.params, numRays, randomId);
		}
		else
		{
			// Multi-ray Siddon
			projectSiddon<IsForward, HasTOF, true, true, UseUpdater>(
			    value, d_image.devicePointer, pd_updater, p1, p2, n1, n2,
			    dynamicFrame, pd_tofHelper, tofValue, scannerParams,
			    d_image.params, numRays, randomId);
		}
		break;
	case ProjectorType::DD:
		if (!hasProjPsf)
		{
			// No projection-space PSF
			projectDD<IsForward, HasTOF, false, UseUpdater>(
			    value, d_image.devicePointer, pd_updater, p1, p2, n1, n2,
			    dynamicFrame, pd_tofHelper, tofValue, projPsfKernelStruct,
			    scannerParams, d_image.params);
		}
		else
		{
			// With projection-space PSF
			projectDD<IsForward, HasTOF, true, UseUpdater>(
			    value, d_image.devicePointer, pd_updater, p1, p2, n1, n2,
			    dynamicFrame, pd_tofHelper, tofValue, projPsfKernelStruct,
			    scannerParams, d_image.params);
		}
		break;
	default:
		// Code should never reach here.
		break;
	}
}

}  // namespace yrt
