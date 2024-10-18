/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/OperatorDevice.cuh"
#include "operators/OperatorProjector.hpp"


class ProjectionDataDevice;
class ImageDevice;

class OperatorProjectorDD_GPU : public OperatorProjectorDevice
{
public:
	explicit OperatorProjectorDD_GPU(const OperatorProjectorParams& projParams,
	                                 bool p_synchronized = true,
	                                 const cudaStream_t* mainStream = nullptr,
	                                 const cudaStream_t* auxStream = nullptr);

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

private:
	void applyAttenuationOnLoadedBatchIfNeeded(
	    const ProjectionDataDevice* imgProjData, bool duringForward);
	void applyAttenuationOnLoadedBatchIfNeeded(
	    const ProjectionDataDevice* imgProjData,
	    ProjectionDataDevice* destProjData, bool duringForward);
	void applyAdditiveOnLoadedBatchIfNeeded(ProjectionDataDevice* imgProjData);

	void applyAttenuationFactors(const ProjectionDataDevice* attImgProj,
	                             const ProjectionDataDevice* imgProj,
	                             ProjectionDataDevice* destProj,
	                             float unitFactor);

	template <bool IsForward>
	void applyOnLoadedBatch(ProjectionDataDevice* dat, ImageDevice* img);

	template <bool IsForward, bool HasTOF, bool HasProjPsf>
	static void launchKernel(
	    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
	    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
	    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
	    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
	    ProjectionPsfProperties
	        projectionPsfProperties,
	    CUScannerParams scannerParams, CUImageParams imgParams,
	    size_t batchSize, unsigned int gridSize, unsigned int blockSize,
	    const cudaStream_t* stream, bool synchronize);
};
