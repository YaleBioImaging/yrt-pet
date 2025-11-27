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

class OperatorProjectorSiddon_GPU : public OperatorProjectorDevice
{
public:
	explicit OperatorProjectorSiddon_GPU(
	    const OperatorProjectorParams& projParams,
	    const std::vector<Constraint*>& constraints = {},
	    const cudaStream_t* mainStream = nullptr,
	    const cudaStream_t* auxStream = nullptr);

	std::set<ProjectionPropertyType>
	    getProjectionPropertyTypesInternal() const override;

	int getNumRays() const;
	void setNumRays(int n);

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
	void launchKernel(
	    float* pd_projValues, float* pd_image,
	    UpdaterPointer pd_updater,
	    const char* pd_projProperties,
	    const ProjectionPropertyManager* pd_projPropManager,
	    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
	    CUImageParams imgParams, size_t batchSize, unsigned int gridSize,
	    unsigned int blockSize, const cudaStream_t* stream, bool synchronize);

	int m_numRays;
};

template <bool IsForward, bool HasTOF, bool IsIncremental, bool IsMultiRay>
__global__ void Siddon_kernel(
    float* pd_projValues, float* pd_image,
    UpdaterPointer pd_updater,
    const char* pd_projProperties,
    const ProjectionPropertyManager* pd_projPropManager,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, int p_numRays, size_t batchSize);

}  // namespace yrt
