/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/OperatorProjectorDevice.cuh"

namespace yrt
{
class ProjectionDataDevice;
class ImageDevice;

class OperatorProjectorSiddon_GPU : public OperatorProjectorDevice
{
public:
	explicit OperatorProjectorSiddon_GPU(
	    const OperatorProjectorParams& projParams,
	    const cudaStream_t* mainStream = nullptr,
	    const cudaStream_t* auxStream = nullptr);

protected:
	void applyAOnLoadedBatch(ImageDevice& img, ProjectionDataDevice& dat,
	                         bool synchronize) override;
	void applyAHOnLoadedBatch(ProjectionDataDevice& dat, ImageDevice& img,
	                          bool synchronize) override;

private:
	template <bool IsForward>
	void applyOnLoadedBatch(ProjectionDataDevice& dat, ImageDevice& img,
	                        bool synchronize);

	template <bool IsForward, bool HasTOF>
	void launchKernel(float* pd_projValues, float* pd_image,
	                  const float4* pd_lorDet1Pos, const float4* pd_lorDet2Pos,
	                  const float4* pd_lorDet1Orient,
	                  const float4* pd_lorDet2Orient,
	                  OperatorProjectorUpdaterDevice* pd_updater,
	                  const frame_t* pd_dynamicFrame, const float* pd_lorTOFValue,
	                  const TimeOfFlightHelper* pd_tofHelper,
	                  CUScannerParams scannerParams, CUImageParams imgParams,
	                  size_t batchSize, unsigned int gridSize,
	                  unsigned int blockSize, const cudaStream_t* stream,
	                  bool synchronize);

	int p_numRays;
};
}  // namespace yrt
