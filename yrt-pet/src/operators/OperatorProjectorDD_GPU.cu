/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorDD_GPU.cuh"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionDataDevice.cuh"
#include "yrt-pet/operators/OperatorProjectorDD_GPUKernels.cuh"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_operatorprojectordd_gpu(py::module& m)
{
	auto c = py::class_<OperatorProjectorDD_GPU, OperatorProjectorDevice>(
	    m, "OperatorProjectorDD_GPU");
	c.def(py::init<const OperatorProjectorParams&>(), py::arg("projParams"));
}
}  // namespace yrt

#endif

namespace yrt
{

OperatorProjectorDD_GPU::OperatorProjectorDD_GPU(
    const OperatorProjectorParams& projParams, const cudaStream_t* mainStream,
    const cudaStream_t* auxStream)
    : OperatorProjectorDevice(projParams, mainStream, auxStream)
{
}

void OperatorProjectorDD_GPU::applyAOnLoadedBatch(ImageDevice& img,
                                                  ProjectionDataDevice& dat,
                                                  bool synchronize)
{
	applyOnLoadedBatch<true>(dat, img, synchronize);
}
void OperatorProjectorDD_GPU::applyAHOnLoadedBatch(ProjectionDataDevice& dat,
                                                   ImageDevice& img,
                                                   bool synchronize)
{
	applyOnLoadedBatch<false>(dat, img, synchronize);
}

template <bool IsForward>
void OperatorProjectorDD_GPU::applyOnLoadedBatch(ProjectionDataDevice& dat,
                                                 ImageDevice& img,
                                                 bool synchronize)
{
	setBatchSize(dat.getLoadedBatchSize());
	const auto cuScannerParams = getCUScannerParams(getScanner());
	const auto cuImageParams = getCUImageParams(img.getParams());
	const TimeOfFlightHelper* tofHelperDevicePointer =
	    getTOFHelperDevicePointer();
	const float* projPsfDevicePointer =
	    getProjPsfKernelsDevicePointer(!IsForward);

	if (projPsfDevicePointer == nullptr)
	{
		if (tofHelperDevicePointer == nullptr)
		{
			launchKernel<IsForward, false, false>(
			    dat.getProjValuesDevicePointer(), img.getDevicePointer(),
			    dat.getLorDet1PosDevicePointer(),
			    dat.getLorDet2PosDevicePointer(),
			    dat.getLorDet1OrientDevicePointer(),
			    dat.getLorDet2OrientDevicePointer(), nullptr /*No TOF*/,
			    nullptr /*No TOF*/, nullptr /*No ProjPSF*/, {} /*No ProjPSF*/,
			    cuScannerParams, cuImageParams, getBatchSize(), getGridSize(),
			    getBlockSize(), getMainStream(), synchronize);
		}
		else
		{
			launchKernel<IsForward, true, false>(
			    dat.getProjValuesDevicePointer(), img.getDevicePointer(),
			    dat.getLorDet1PosDevicePointer(),
			    dat.getLorDet2PosDevicePointer(),
			    dat.getLorDet1OrientDevicePointer(),
			    dat.getLorDet2OrientDevicePointer(),
			    dat.getLorTOFValueDevicePointer(), tofHelperDevicePointer,
			    nullptr /*No ProjPSF*/, {} /*No ProjPSF*/, cuScannerParams,
			    cuImageParams, getBatchSize(), getGridSize(), getBlockSize(),
			    getMainStream(), synchronize);
		}
	}
	else
	{
		const ProjectionPsfProperties projectionPsfProperties =
		    mp_projPsfManager->getProjectionPsfProperties();

		if (tofHelperDevicePointer == nullptr)
		{
			launchKernel<IsForward, false, true>(
			    dat.getProjValuesDevicePointer(), img.getDevicePointer(),
			    dat.getLorDet1PosDevicePointer(),
			    dat.getLorDet2PosDevicePointer(),
			    dat.getLorDet1OrientDevicePointer(),
			    dat.getLorDet2OrientDevicePointer(), nullptr /*No TOF*/,
			    nullptr /*No TOF*/, projPsfDevicePointer,
			    projectionPsfProperties, cuScannerParams, cuImageParams,
			    getBatchSize(), getGridSize(), getBlockSize(), getMainStream(),
			    synchronize);
		}
		else
		{
			launchKernel<IsForward, true, true>(
			    dat.getProjValuesDevicePointer(), img.getDevicePointer(),
			    dat.getLorDet1PosDevicePointer(),
			    dat.getLorDet2PosDevicePointer(),
			    dat.getLorDet1OrientDevicePointer(),
			    dat.getLorDet2OrientDevicePointer(),
			    dat.getLorTOFValueDevicePointer(), tofHelperDevicePointer,
			    projPsfDevicePointer, projectionPsfProperties, cuScannerParams,
			    cuImageParams, getBatchSize(), getGridSize(), getBlockSize(),
			    getMainStream(), synchronize);
		}
	}
}

template <bool IsForward, bool HasTOF, bool HasProjPsf>
void OperatorProjectorDD_GPU::launchKernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, const float* pd_projPsfKernels,
    ProjectionPsfProperties projectionPsfProperties,
    CUScannerParams scannerParams, CUImageParams imgParams, size_t batchSize,
    unsigned int gridSize, unsigned int blockSize, const cudaStream_t* stream,
    bool synchronize)
{
	ASSERT_MSG(pd_projValues != nullptr && pd_lorDet1Pos != nullptr &&
	               pd_lorDet2Pos != nullptr && pd_lorDet1Orient != nullptr &&
	               pd_lorDet2Orient != nullptr,
	           "Projection space not allocated on device");
	ASSERT_MSG(pd_image != nullptr, "Image space not allocated on device");

	if (stream != nullptr)
	{
		OperatorProjectorDDCU_kernel<IsForward, HasTOF, HasProjPsf>
		    <<<gridSize, blockSize, 0, *stream>>>(
		        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
		        pd_lorDet1Orient, pd_lorDet2Orient, pd_lorTOFValue,
		        pd_tofHelper, pd_projPsfKernels, projectionPsfProperties,
		        scannerParams, imgParams, batchSize);
		if (synchronize)
		{
			cudaStreamSynchronize(*stream);
		}
	}
	else
	{
		OperatorProjectorDDCU_kernel<IsForward, HasTOF, HasProjPsf>
		    <<<gridSize, blockSize>>>(
		        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
		        pd_lorDet1Orient, pd_lorDet2Orient, pd_lorTOFValue,
		        pd_tofHelper, pd_projPsfKernels, projectionPsfProperties,
		        scannerParams, imgParams, batchSize);
		if (synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}
}  // namespace yrt
