/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/OperatorProjectorDevice.cuh"


class ProjectionDataDevice;
class ImageDevice;

class OperatorProjectorSiddon_GPU : public OperatorProjectorDevice
{
public:
	// TODO NOW: Support multi-ray siddon
	explicit OperatorProjectorSiddon_GPU(
	    const OperatorProjectorParams& projParams, bool p_synchronized = true,
	    const cudaStream_t* mainStream = nullptr,
	    const cudaStream_t* auxStream = nullptr);

protected:
	void applyAOnLoadedBatch(ImageDevice& img,
	                         ProjectionDataDevice& dat) override;
	void applyAHOnLoadedBatch(ProjectionDataDevice& dat,
	                          ImageDevice& img) override;

private:
	template <bool IsForward>
	void applyOnLoadedBatch(ProjectionDataDevice& dat, ImageDevice& img);

	template <bool IsForward, bool HasTOF>
	static void launchKernel(
	    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
	    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
	    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
	    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
	    CUImageParams imgParams, size_t batchSize, unsigned int gridSize,
	    unsigned int blockSize, const cudaStream_t* stream, bool synchronize);
};
