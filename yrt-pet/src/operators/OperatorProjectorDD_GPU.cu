/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorDD_GPU.cuh"

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionListDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/DDKernels.cuh"
#include "yrt-pet/operators/ProjectionPsfManagerDevice.cuh"
#include "yrt-pet/operators/ProjectorDD.hpp"
#include "yrt-pet/operators/ProjectorUpdaterDevice.cuh"
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
void py_setup_operatorprojectordd_gpu(py::module& m)
{
	auto c = py::class_<OperatorProjectorDD_GPU, OperatorProjectorDevice>(
	    m, "OperatorProjectorDD_GPU");
	c.def(py::init<const ProjectorParams&, const BinIterator*>(),
	      "proj_params"_a, "bin_iter"_a = nullptr);
	c.def(py::init<const ProjectorParams&, const BinIterator*,
	               std::vector<Constraint*>>(),
	      "proj_params"_a, "bin_iter"_a, "constraints"_a);
	c.def(py::init<const ProjectorParams&>(), "proj_params"_a);
}
}  // namespace yrt

#endif

namespace yrt
{

OperatorProjectorDD_GPU::OperatorProjectorDD_GPU(
    const ProjectorParams& projParams, const BinIterator* binIter,
    const std::vector<Constraint*>& constraints, const cudaStream_t* mainStream,
    const cudaStream_t* auxStream, size_t p_memAvailBytes)
    : OperatorProjectorDevice(projParams, binIter, constraints, mainStream,
                              auxStream, p_memAvailBytes)
{
	ASSERT_MSG(projParams.projectorType == ProjectorType::DD,
	           "Projector params must have, as projector type, \"DD\"");
}

std::set<ProjectionPropertyType> OperatorProjectorDD_GPU::getNeededProperties()
{
	return ProjectorDD::getNeededProperties();
}

void OperatorProjectorDD_GPU::applyAOnLoadedBatch(ImageDevice& img,
                                                  ProjectionListDevice& dat,
                                                  bool synchronize)
{
	applyOnLoadedBatch<true>(dat, img, synchronize);
}
void OperatorProjectorDD_GPU::applyAHOnLoadedBatch(ProjectionListDevice& dat,
                                                   ImageDevice& img,
                                                   bool synchronize)
{
	applyOnLoadedBatch<false>(dat, img, synchronize);
}

template <bool IsForward>
void OperatorProjectorDD_GPU::applyOnLoadedBatch(ProjectionListDevice& dat,
                                                 ImageDevice& img,
                                                 bool synchronize)
{
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
	const float* pd_projPsfKernels = getProjPsfKernelsDevicePointer(!IsForward);

	if (pd_projPsfKernels == nullptr)
	{
		if (pd_tofHelper == nullptr)
		{
			launchKernel<IsForward, false, false>(
			    pd_projValues, pd_image, pd_updater, pd_projPropManager,
			    pd_projProp, nullptr /*No TOF*/, {} /*No ProjPSF*/,
			    nullptr /*No ProjPSF*/, cuScannerParams, cuImageParams,
			    getBatchSize(), getGridSize(), getBlockSize(), getMainStream(),
			    synchronize);
		}
		else
		{
			launchKernel<IsForward, true, false>(
			    pd_projValues, pd_image, pd_updater, pd_projPropManager,
			    pd_projProp, pd_tofHelper, {} /*No ProjPSF*/,
			    nullptr /*No ProjPSF*/, cuScannerParams, cuImageParams,
			    getBatchSize(), getGridSize(), getBlockSize(), getMainStream(),
			    synchronize);
		}
	}
	else
	{
		const ProjectionPsfProperties projectionPsfProperties =
		    mp_projPsfManager->getProjectionPsfProperties();

		if (pd_tofHelper == nullptr)
		{
			launchKernel<IsForward, false, true>(
			    pd_projValues, pd_image, pd_updater, pd_projPropManager,
			    pd_projProp, nullptr /*No TOF*/, projectionPsfProperties,
			    pd_projPsfKernels, cuScannerParams, cuImageParams,
			    getBatchSize(), getGridSize(), getBlockSize(), getMainStream(),
			    synchronize);
		}
		else
		{
			launchKernel<IsForward, true, true>(
			    pd_projValues, pd_image, pd_updater, pd_projPropManager,
			    pd_projProp, pd_tofHelper, projectionPsfProperties,
			    pd_projPsfKernels, cuScannerParams, cuImageParams,
			    getBatchSize(), getGridSize(), getBlockSize(), getMainStream(),
			    synchronize);
		}
	}
}

template <bool IsForward, bool HasTOF, bool HasProjPSF>
void OperatorProjectorDD_GPU::launchKernel(
    float* pd_projValues, float* pd_image, UpdaterPointer pd_updater,
    const ProjectionPropertyManager* pd_projPropManager,
    const PropertyUnit* pd_projProperties,
    const TimeOfFlightHelper* pd_tofHelper,
    ProjectionPsfProperties projectionPsfProperties,
    const float* pd_projPsfKernels, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize, unsigned int gridSize,
    unsigned int blockSize, const cudaStream_t* stream, bool synchronize)
{
	ASSERT_MSG(pd_projValues != nullptr && pd_projProperties != nullptr,
	           "Projection space not allocated on device");
	ASSERT_MSG(pd_image != nullptr, "Image space not allocated on device");

	if (stream != nullptr)
	{
		if (pd_updater != nullptr)
		{
			projectDD_kernel<IsForward, HasTOF, HasProjPSF, true>
			    <<<gridSize, blockSize, 0, *stream>>>(
			        pd_projValues, pd_image, pd_updater, pd_projPropManager,
			        pd_projProperties, pd_tofHelper,
			        {projectionPsfProperties, pd_projPsfKernels}, scannerParams,
			        imgParams, batchSize);
		}
		else
		{
			projectDD_kernel<IsForward, HasTOF, HasProjPSF, false>
			    <<<gridSize, blockSize, 0, *stream>>>(
			        pd_projValues, pd_image, nullptr /* No updater */,
			        pd_projPropManager, pd_projProperties, pd_tofHelper,
			        {projectionPsfProperties, pd_projPsfKernels}, scannerParams,
			        imgParams, batchSize);
		}
	}
	else
	{
		if (pd_updater != nullptr)
		{
			projectDD_kernel<IsForward, HasTOF, HasProjPSF, true>
			    <<<gridSize, blockSize>>>(
			        pd_projValues, pd_image, pd_updater, pd_projPropManager,
			        pd_projProperties, pd_tofHelper,
			        {projectionPsfProperties, pd_projPsfKernels}, scannerParams,
			        imgParams, batchSize);
		}
		else
		{
			projectDD_kernel<IsForward, HasTOF, HasProjPSF, false>
			    <<<gridSize, blockSize>>>(
			        pd_projValues, pd_image, nullptr /* No updater */,
			        pd_projPropManager, pd_projProperties, pd_tofHelper,
			        {projectionPsfProperties, pd_projPsfKernels}, scannerParams,
			        imgParams, batchSize);
		}
	}

	synchronizeIfNeeded({stream, synchronize});

	ASSERT(cudaCheckError());
}

}  // namespace yrt
