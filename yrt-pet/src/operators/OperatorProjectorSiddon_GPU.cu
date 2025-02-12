/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorProjectorSiddon_GPU.cuh"

#include "datastruct/image/Image.hpp"
#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "operators/OperatorProjectorSiddon_GPUKernels.cuh"
#include "utils/Assert.hpp"
#include "utils/GPUUtils.cuh"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_operatorprojectorsiddon_gpu(py::module& m)
{
	auto c = py::class_<OperatorProjectorSiddon_GPU, OperatorProjectorDevice>(
	    m, "OperatorProjectorSiddon_GPU");
	c.def(py::init<const OperatorProjectorParams&>(), py::arg("projParams"));
}
#endif

OperatorProjectorSiddon_GPU::OperatorProjectorSiddon_GPU(
    const OperatorProjectorParams& projParams, bool p_synchronized,
    const cudaStream_t* mainStream, const cudaStream_t* auxStream)
    : OperatorProjectorDevice(projParams, p_synchronized, mainStream, auxStream)
{
}

void OperatorProjectorSiddon_GPU::applyAOnLoadedBatch(ImageDevice& img,
                                                  ProjectionDataDevice& dat)
{
	applyOnLoadedBatch<true>(dat, img);
}
void OperatorProjectorSiddon_GPU::applyAHOnLoadedBatch(ProjectionDataDevice& dat,
                                                   ImageDevice& img)
{
	applyOnLoadedBatch<false>(dat, img);
}

template <bool IsForward>
void OperatorProjectorSiddon_GPU::applyOnLoadedBatch(ProjectionDataDevice& dat,
                                                 ImageDevice& img)
{
	setBatchSize(dat.getCurrentBatchSize());
	const auto cuScannerParams = getCUScannerParams(getScanner());
	const auto cuImageParams = getCUImageParams(img.getParams());
	const TimeOfFlightHelper* tofHelperDevicePointer =
	    getTOFHelperDevicePointer();

	// We assume there is no Projection-space PSF to do

	if (tofHelperDevicePointer == nullptr)
	{
		OperatorProjectorSiddon_GPU::launchKernel<IsForward, false>(
		    dat.getProjValuesDevicePointer(), img.getDevicePointer(),
		    dat.getLorDet1PosDevicePointer(),
		    dat.getLorDet2PosDevicePointer(),
		    dat.getLorDet1OrientDevicePointer(),
		    dat.getLorDet2OrientDevicePointer(), nullptr /*No TOF*/,
		    nullptr /*No TOF*/,
		    cuScannerParams, cuImageParams, getBatchSize(), getGridSize(),
		    getBlockSize(), getMainStream(), isSynchronized());
	}
	else
	{
		OperatorProjectorSiddon_GPU::launchKernel<IsForward, true>(
		    dat.getProjValuesDevicePointer(), img.getDevicePointer(),
		    dat.getLorDet1PosDevicePointer(),
		    dat.getLorDet2PosDevicePointer(),
		    dat.getLorDet1OrientDevicePointer(),
		    dat.getLorDet2OrientDevicePointer(),
		    dat.getLorTOFValueDevicePointer(), tofHelperDevicePointer,
		    cuScannerParams, cuImageParams, getBatchSize(), getGridSize(), getBlockSize(),
		    getMainStream(), isSynchronized());
	}
}

template <bool IsForward, bool HasTOF>
void OperatorProjectorSiddon_GPU::launchKernel(
		float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
		const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
		const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
		const TimeOfFlightHelper* pd_tofHelper,
		CUScannerParams scannerParams, CUImageParams imgParams,
		size_t batchSize, unsigned int gridSize, unsigned int blockSize,
		const cudaStream_t* stream, bool synchronize)
{
	ASSERT_MSG(pd_projValues != nullptr && pd_lorDet1Pos != nullptr &&
	               pd_lorDet2Pos != nullptr && pd_lorDet1Orient != nullptr &&
	               pd_lorDet2Orient != nullptr,
	           "Projection space not allocated on device");
	ASSERT_MSG(pd_image != nullptr, "Image space not allocated on device");

	if (stream != nullptr)
	{
		OperatorProjectorSiddonCU_kernel<IsForward, HasTOF, true>
		    <<<gridSize, blockSize, 0, *stream>>>(
		        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
		        pd_lorDet1Orient, pd_lorDet2Orient, pd_lorTOFValue,
		        pd_tofHelper,
		        scannerParams, imgParams, batchSize);
		if (synchronize)
		{
			cudaStreamSynchronize(*stream);
		}
	}
	else
	{
		OperatorProjectorSiddonCU_kernel<IsForward, HasTOF, true>
		    <<<gridSize, blockSize>>>(
		        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
		        pd_lorDet1Orient, pd_lorDet2Orient, pd_lorTOFValue,
		        pd_tofHelper,
		        scannerParams, imgParams, batchSize);
		if (synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}
