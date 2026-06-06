/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorJoseph_GPU.cuh"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionDataDevice.cuh"
#include "yrt-pet/operators/OperatorProjectorJoseph_GPUKernels.cuh"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"

#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_operatorprojectorjoseph_gpu(py::module& m)
{
	auto c = py::class_<OperatorProjectorJoseph_GPU, OperatorProjectorDevice>(
	    m, "OperatorProjectorJoseph_GPU");
	c.def(py::init<const OperatorProjectorParams&>(), py::arg("projParams"));
}
}  // namespace yrt

#endif

namespace yrt
{
OperatorProjectorJoseph_GPU::OperatorProjectorJoseph_GPU(
    const OperatorProjectorParams& projParams, const cudaStream_t* mainStream,
    const cudaStream_t* auxStream)
    : OperatorProjectorDevice(projParams, mainStream, auxStream)
{
	if (projParams.numRays != 1)
	{
		throw std::invalid_argument(
		    "CUDA Joseph projector supports only single-ray projection");
	}
}

void OperatorProjectorJoseph_GPU::applyAOnLoadedBatch(ImageDevice& img,
                                                      ProjectionDataDevice& dat,
                                                      bool synchronize)
{
	applyOnLoadedBatch<true>(dat, img, synchronize);
}
void OperatorProjectorJoseph_GPU::applyAHOnLoadedBatch(
    ProjectionDataDevice& dat, ImageDevice& img, bool synchronize)
{
	applyOnLoadedBatch<false>(dat, img, synchronize);
}

template <bool IsForward>
void OperatorProjectorJoseph_GPU::applyOnLoadedBatch(ProjectionDataDevice& dat,
                                                     ImageDevice& img,
                                                     bool synchronize)
{
	setBatchSize(dat.getLoadedBatchSize());
	const auto cuScannerParams = getCUScannerParams(getScanner());
	const auto cuImageParams = getCUImageParams(img.getParams());
	const TimeOfFlightHelper* tofHelperDevicePointer =
	    getTOFHelperDevicePointer();

	if (tofHelperDevicePointer == nullptr)
	{
		OperatorProjectorJoseph_GPU::launchKernel<IsForward, false>(
		    dat.getProjValuesDevicePointer(), img.getDevicePointer(),
		    dat.getLorDet1PosDevicePointer(), dat.getLorDet2PosDevicePointer(),
		    dat.getLorDet1OrientDevicePointer(),
		    dat.getLorDet2OrientDevicePointer(), nullptr /*No TOF*/,
		    nullptr /*No TOF*/, cuScannerParams, cuImageParams, getBatchSize(),
		    getGridSize(), getBlockSize(), getMainStream(), synchronize);
	}
	else
	{
		OperatorProjectorJoseph_GPU::launchKernel<IsForward, true>(
		    dat.getProjValuesDevicePointer(), img.getDevicePointer(),
		    dat.getLorDet1PosDevicePointer(), dat.getLorDet2PosDevicePointer(),
		    dat.getLorDet1OrientDevicePointer(),
		    dat.getLorDet2OrientDevicePointer(),
		    dat.getLorTOFValueDevicePointer(), tofHelperDevicePointer,
		    cuScannerParams, cuImageParams, getBatchSize(), getGridSize(),
		    getBlockSize(), getMainStream(), synchronize);
	}
}

template <bool IsForward, bool HasTOF>
void OperatorProjectorJoseph_GPU::launchKernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize, unsigned int gridSize,
    unsigned int blockSize, const cudaStream_t* stream, bool synchronize)
{
	ASSERT_MSG(pd_projValues != nullptr && pd_lorDet1Pos != nullptr &&
	               pd_lorDet2Pos != nullptr && pd_lorDet1Orient != nullptr &&
	               pd_lorDet2Orient != nullptr,
	           "Projection space not allocated on device");
	ASSERT_MSG(pd_image != nullptr, "Image space not allocated on device");

	if (stream != nullptr)
	{
		OperatorProjectorJosephCU_kernel<IsForward, HasTOF>
		    <<<gridSize, blockSize, 0, *stream>>>(
		        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
		        pd_lorDet1Orient, pd_lorDet2Orient, pd_lorTOFValue,
		        pd_tofHelper, scannerParams, imgParams, batchSize);
		if (synchronize)
		{
			cudaStreamSynchronize(*stream);
		}
	}
	else
	{
		OperatorProjectorJosephCU_kernel<IsForward, HasTOF>
		    <<<gridSize, blockSize>>>(
		        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
		        pd_lorDet1Orient, pd_lorDet2Orient, pd_lorTOFValue,
		        pd_tofHelper, scannerParams, imgParams, batchSize);
		if (synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}
}  // namespace yrt
