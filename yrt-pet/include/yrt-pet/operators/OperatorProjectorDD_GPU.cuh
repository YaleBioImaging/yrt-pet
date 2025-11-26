/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/OperatorProjectorDevice.cuh"

namespace yrt
{
class ProjectionDataDevice;
class ImageDevice;

class OperatorProjectorDD_GPU : public OperatorProjectorDevice
{
public:
	explicit OperatorProjectorDD_GPU(
	    const OperatorProjectorParams& projParams,
	    const std::vector<Constraint*>& constraints = {},
	    const cudaStream_t* mainStream = nullptr,
	    const cudaStream_t* auxStream = nullptr);

	std::set<ProjectionPropertyType>
	    getProjectionPropertyTypesInternal() const override;

protected:
	void applyAOnLoadedBatch(ImageDevice& img, ProjectionDataDevice& dat,
	                         bool synchronize) override;
	void applyAHOnLoadedBatch(ProjectionDataDevice& dat, ImageDevice& img,
	                          bool synchronize) override;

private:
	template <bool IsForward>
	void applyOnLoadedBatch(ProjectionDataDevice& dat, ImageDevice& img,
	                        bool synchronize);

	template <bool IsForward, bool HasTOF, bool HasProjPsf>
	static void launchKernel(
	    float* pd_projValues, float* pd_image, const char* pd_projProperties,
	    const ProjectionPropertyManager* pd_projPropManager,
	    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
	    ProjectionPsfProperties projectionPsfProperties,
	    CUScannerParams scannerParams, CUImageParams imgParams,
	    size_t batchSize, unsigned int gridSize, unsigned int blockSize,
	    const cudaStream_t* stream, bool synchronize);
};
}  // namespace yrt
