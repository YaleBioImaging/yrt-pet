/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/OperatorProjectorDevice.cuh"

namespace yrt
{
class ProjectionListDevice;
class ImageDevice;

class OperatorProjectorSiddon_GPU : public OperatorProjectorDevice
{
public:
	explicit OperatorProjectorSiddon_GPU(
	    const ProjectorParams& projParams, const BinIterator* binIter = nullptr,
	    const std::vector<Constraint*>& constraints = {},
	    const cudaStream_t* mainStream = nullptr,
	    const cudaStream_t* auxStream = nullptr, size_t p_memAvailBytes = 0);

	static std::set<ProjectionPropertyType> getNeededProperties(int numRays);

	int getNumRays() const;
	void setNumRays(int n);

	int getNumRays() const;
	void setNumRays(int n);

protected:
	void applyAOnLoadedBatch(ImageDevice& img, ProjectionListDevice& dat,
	                         bool synchronize) override;
	void applyAHOnLoadedBatch(ProjectionListDevice& dat, ImageDevice& img,
	                          bool synchronize) override;

private:
	template <bool IsForward>
	void applyOnLoadedBatch(ProjectionListDevice& dat, ImageDevice& img,
	                        bool synchronize);

	template <bool IsForward, bool HasTOF, bool IsMultiRay>
	static void launchKernel(float* pd_projValues, float* pd_image,
	                  UpdaterPointer pd_updater,
	                  const ProjectionPropertyManager* pd_projPropManager,
	                  const PropertyUnit* pd_projProperties,
	                  const TimeOfFlightHelper* pd_tofHelper,
	                  int numRays,
	                  CUScannerParams scannerParams, CUImageParams imgParams,
	                  size_t batchSize, unsigned int gridSize,
	                  unsigned int blockSize, const cudaStream_t* stream,
	                  bool synchronize);

	int m_numRays;
};

}  // namespace yrt
