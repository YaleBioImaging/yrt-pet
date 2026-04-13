/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorSiddon_GPU.cuh"

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionListDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/ProjectorSiddon.hpp"
#include "yrt-pet/operators/ProjectorUpdaterDevice.cuh"
#include "yrt-pet/operators/SiddonKernels.cuh"
#include "yrt-pet/operators/TimeOfFlight.hpp"
#include "yrt-pet/recon/CUParameters.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_operatorprojectorsiddon_gpu(py::module& m)
{
	auto c = py::class_<OperatorProjectorSiddon_GPU, OperatorProjectorDevice>(
	    m, "OperatorProjectorSiddon_GPU");
	c.def(py::init<const ProjectorParams&, const BinIterator*>(),
	      "proj_params"_a, "bin_iter"_a = nullptr);
	c.def(py::init<const ProjectorParams&, const BinIterator*,
	               std::vector<Constraint*>>(),
	      "proj_params"_a, "bin_iter"_a, "constraints"_a);
	c.def("setNumRays", &OperatorProjectorSiddon_GPU::setNumRays, "num_rays"_a);
	c.def("getNumRays", &OperatorProjectorSiddon_GPU::getNumRays);
}
}  // namespace yrt

#endif

namespace yrt
{

OperatorProjectorSiddon_GPU::OperatorProjectorSiddon_GPU(
    const ProjectorParams& projParams, const BinIterator* binIter,
    const std::vector<Constraint*>& constraints, const cudaStream_t* mainStream,
    const cudaStream_t* auxStream, size_t p_memAvailBytes)
    : OperatorProjectorDevice(projParams, binIter, constraints, mainStream,
                              auxStream, p_memAvailBytes),
      m_numRays(projParams.numRays)
{
	ASSERT_MSG(projParams.projectorType == ProjectorType::SIDDON,
	           "Projector params must have, as projector type, \"Siddon\"");
}

std::set<ProjectionPropertyType>
    OperatorProjectorSiddon_GPU::getNeededProperties(int numRays)
{
	return ProjectorSiddon::getNeededProperties(numRays);
}

int OperatorProjectorSiddon_GPU::getNumRays() const
{
	return m_numRays;
}

void OperatorProjectorSiddon_GPU::setNumRays(int n)
{
	m_numRays = n;
}

void OperatorProjectorSiddon_GPU::applyAOnLoadedBatch(ImageDevice& img,
                                                      ProjectionListDevice& dat,
                                                      bool synchronize)
{
	applyOnLoadedBatch<true>(dat, img, synchronize);
}
void OperatorProjectorSiddon_GPU::applyAHOnLoadedBatch(
    ProjectionListDevice& dat, ImageDevice& img, bool synchronize)
{
	applyOnLoadedBatch<false>(dat, img, synchronize);
}

template <bool IsForward>
void OperatorProjectorSiddon_GPU::applyOnLoadedBatch(ProjectionListDevice& dat,
                                                     ImageDevice& img,
                                                     bool synchronize)
{
	// TODO: Maybe the setBatchSize logic should be in OperatorProjectorDevice
	//  rather than in the Siddon or DD implementation
	setBatchSize(dat.getLoadedBatchSize());
	const auto cuScannerParams = getCUScannerParams(getScanner());
	const auto cuImageParams = getCUImageParams(img.getParams());

	float* pd_image = img.getDevicePointer();
	float* pd_projValues = dat.getProjValuesDevicePointer();
	UpdaterPointer pd_updater = getUpdaterDevicePointer();
	const ProjectionPropertyManager* pd_projPropManager =
	    dat.getProjectionPropertyManagerDevicePointer();
	const PropertyUnit* pd_projProp =
	    dat.getProjectionPropertiesDevicePointer();
	const TimeOfFlightHelper* pd_tofHelper = getTOFHelperDevicePointer();

	if (m_numRays == 1)
	{
		// We assume there is no Projection-space PSF to do
		if (pd_tofHelper == nullptr)
		{
			launchKernel<IsForward, false, false>(
			    pd_projValues, pd_image, pd_updater, pd_projPropManager,
			    pd_projProp, nullptr /*No TOF*/, 1 /* Single ray */,
			    cuScannerParams, cuImageParams, getBatchSize(), getGridSize(),
			    getBlockSize(), getMainStream(), synchronize);
		}
		else
		{
			launchKernel<IsForward, true, false>(
			    pd_projValues, pd_image, pd_updater, pd_projPropManager,
			    pd_projProp, pd_tofHelper, 1 /* Single ray */, cuScannerParams,
			    cuImageParams, getBatchSize(), getGridSize(), getBlockSize(),
			    getMainStream(), synchronize);
		}
	}
	else
	{
		if (pd_tofHelper == nullptr)
		{
			launchKernel<IsForward, false, true>(
			    pd_projValues, pd_image, pd_updater, pd_projPropManager,
			    pd_projProp, nullptr /*No TOF*/, m_numRays, cuScannerParams,
			    cuImageParams, getBatchSize(), getGridSize(), getBlockSize(),
			    getMainStream(), synchronize);
		}
		else
		{
			launchKernel<IsForward, true, true>(
			    pd_projValues, pd_image, pd_updater, pd_projPropManager,
			    pd_projProp, pd_tofHelper, m_numRays, cuScannerParams,
			    cuImageParams, getBatchSize(), getGridSize(), getBlockSize(),
			    getMainStream(), synchronize);
		}
	}
}

template <bool IsForward, bool HasTOF, bool IsMultiRay>
void OperatorProjectorSiddon_GPU::launchKernel(
    float* pd_projValues, float* pd_image, UpdaterPointer pd_updater,
    const ProjectionPropertyManager* pd_projPropManager,
    const PropertyUnit* pd_projProperties,
    const TimeOfFlightHelper* pd_tofHelper, int numRays,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize,
    unsigned int gridSize, unsigned int blockSize, const cudaStream_t* stream,
    bool synchronize)
{
	ASSERT_MSG(pd_projValues != nullptr && pd_projPropManager != nullptr,
	           "Projection space not allocated on device");
	ASSERT_MSG(pd_image != nullptr, "Image space not allocated on device");

	if (stream != nullptr)
	{
		if (pd_updater != nullptr)
		{
			projectSiddon_kernel<IsForward, HasTOF, true, IsMultiRay, true>
			    <<<gridSize, blockSize, 0, *stream>>>(
			        pd_projValues, pd_image, pd_updater, pd_projPropManager,
			        pd_projProperties, pd_tofHelper, scannerParams, imgParams,
			        numRays, batchSize);
		}
		else
		{
			projectSiddon_kernel<IsForward, HasTOF, true, IsMultiRay, false>
			    <<<gridSize, blockSize, 0, *stream>>>(
			        pd_projValues, pd_image, nullptr /* No updater */, pd_projPropManager,
			        pd_projProperties, pd_tofHelper, scannerParams, imgParams,
			        numRays, batchSize);
		}
	}
	else
	{
		if (pd_updater != nullptr)
		{
			projectSiddon_kernel<IsForward, HasTOF, true, IsMultiRay, true>
			    <<<gridSize, blockSize>>>(pd_projValues, pd_image, pd_updater,
			                              pd_projPropManager, pd_projProperties,
			                              pd_tofHelper, scannerParams,
			                              imgParams, numRays, batchSize);
		}
		else
		{
			projectSiddon_kernel<IsForward, HasTOF, true, IsMultiRay, false>
			    <<<gridSize, blockSize>>>(pd_projValues, pd_image, nullptr /* No updater */,
			                              pd_projPropManager, pd_projProperties,
			                              pd_tofHelper, scannerParams,
			                              imgParams, numRays, batchSize);
		}
	}

	synchronizeIfNeeded({stream, synchronize});

	ASSERT(cudaCheckError());
}

}  // namespace yrt
