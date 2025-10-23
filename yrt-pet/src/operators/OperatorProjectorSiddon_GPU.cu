/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorSiddon_GPU.cuh"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionDataDevice.cuh"
#include "yrt-pet/operators/OperatorProjectorSiddon_GPUKernels.cuh"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_operatorprojectorsiddon_gpu(py::module& m)
{
	auto c = py::class_<OperatorProjectorSiddon_GPU, OperatorProjectorDevice>(
	    m, "OperatorProjectorSiddon_GPU");
	c.def(py::init<const OperatorProjectorParams&>(), py::arg("projParams"));
}
}  // namespace yrt

#endif

namespace yrt
{
OperatorProjectorSiddon_GPU::OperatorProjectorSiddon_GPU(
    const OperatorProjectorParams& projParams, const cudaStream_t* mainStream,
    const cudaStream_t* auxStream)
    : OperatorProjectorDevice(projParams, mainStream, auxStream),
      p_numRays{projParams.numRays}
{
}

void OperatorProjectorSiddon_GPU::applyAOnLoadedBatch(ImageDevice& img,
                                                      ProjectionDataDevice& dat,
                                                      bool synchronize)
{
	applyOnLoadedBatch<true>(dat, img, synchronize);
}
void OperatorProjectorSiddon_GPU::applyAHOnLoadedBatch(
    ProjectionDataDevice& dat, ImageDevice& img, bool synchronize)
{
	applyOnLoadedBatch<false>(dat, img, synchronize);
}

template <bool IsForward>
void OperatorProjectorSiddon_GPU::applyOnLoadedBatch(ProjectionDataDevice& dat,
                                                     ImageDevice& img,
                                                     bool synchronize)
{
	setBatchSize(dat.getLoadedBatchSize());
	const auto cuScannerParams = getCUScannerParams(getScanner());
	const auto cuImageParams = getCUImageParams(img.getParams());
	const TimeOfFlightHelper* tofHelperDevicePointer =
	    getTOFHelperDevicePointer();

	// We assume there is no Projection-space PSF to do

	if (tofHelperDevicePointer == nullptr)
	{
		OperatorProjectorSiddon_GPU::launchKernel<IsForward, false>(
		    dat.getProjValuesDevicePointer(), img.getDevicePointer(),
		    dat.getLorDet1PosDevicePointer(), dat.getLorDet2PosDevicePointer(),
		    dat.getLorDet1OrientDevicePointer(),
		    dat.getLorDet2OrientDevicePointer(), getUpdaterDevicePointer(),
		    dat.getDynamicFrameDevicePointer(),
		    nullptr /*No TOF*/,
		    nullptr /*No TOF*/, cuScannerParams, cuImageParams, getBatchSize(),
		    getGridSize(), getBlockSize(), getMainStream(), synchronize);
	}
	else
	{
		OperatorProjectorSiddon_GPU::launchKernel<IsForward, true>(
		    dat.getProjValuesDevicePointer(), img.getDevicePointer(),
		    dat.getLorDet1PosDevicePointer(), dat.getLorDet2PosDevicePointer(),
		    dat.getLorDet1OrientDevicePointer(),
		    dat.getLorDet2OrientDevicePointer(), getUpdaterDevicePointer(),
		    dat.getDynamicFrameDevicePointer(),
		    dat.getLorTOFValueDevicePointer(), tofHelperDevicePointer,
		    cuScannerParams, cuImageParams, getBatchSize(), getGridSize(),
		    getBlockSize(), getMainStream(), synchronize);
	}
}

template <bool IsForward, bool HasTOF>
void OperatorProjectorSiddon_GPU::launchKernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, OperatorProjectorUpdaterDevice* pd_updater,
    const frame_t* pd_dynamicFrame, const float* pd_lorTOFValue,
    const TimeOfFlightHelper* pd_tofHelper, CUScannerParams scannerParams,
    CUImageParams imgParams, size_t batchSize, unsigned int gridSize,
    unsigned int blockSize, const cudaStream_t* stream, bool synchronize)
{
	ASSERT_MSG(pd_projValues != nullptr && pd_lorDet1Pos != nullptr &&
	               pd_lorDet2Pos != nullptr && pd_lorDet1Orient != nullptr &&
	               pd_lorDet2Orient != nullptr,
	           "Projection space not allocated on device");
	ASSERT_MSG(pd_image != nullptr, "Image space not allocated on device");

	if (p_numRays == 1)
	{
		if (stream != nullptr)
		{
			OperatorProjectorSiddonCU_kernel<IsForward, HasTOF, true, false>
			    <<<gridSize, blockSize, 0, *stream>>>(
			        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
			        pd_lorDet1Orient, pd_lorDet2Orient, pd_updater,
			        pd_dynamicFrame, pd_lorTOFValue, pd_tofHelper,
			        scannerParams, imgParams, 1, batchSize);
			if (synchronize)
			{
				cudaStreamSynchronize(*stream);
			}
		}
		else
		{
			OperatorProjectorSiddonCU_kernel<IsForward, HasTOF, true, false>
			    <<<gridSize, blockSize>>>(
			        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
			        pd_lorDet1Orient, pd_lorDet2Orient, pd_updater,
			        pd_dynamicFrame, pd_lorTOFValue, pd_tofHelper,
			        scannerParams, imgParams, 1, batchSize);
			if (synchronize)
			{
				cudaDeviceSynchronize();
			}
		}
	}
	else
	{
		if (stream != nullptr)
		{
			OperatorProjectorSiddonCU_kernel<IsForward, HasTOF, true, true>
			    <<<gridSize, blockSize, 0, *stream>>>(
			        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
			        pd_lorDet1Orient, pd_lorDet2Orient, pd_updater,
			        pd_dynamicFrame, pd_lorTOFValue, pd_tofHelper,
			        scannerParams, imgParams, p_numRays, batchSize);
			if (synchronize)
			{
				cudaStreamSynchronize(*stream);
			}
		}
		else
		{
			OperatorProjectorSiddonCU_kernel<IsForward, HasTOF, true, true>
			    <<<gridSize, blockSize>>>(
			        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
			        pd_lorDet1Orient, pd_lorDet2Orient, pd_updater,
			        pd_dynamicFrame, pd_lorTOFValue, pd_tofHelper,
			        scannerParams, imgParams, p_numRays, batchSize);
			if (synchronize)
			{
				cudaDeviceSynchronize();
			}
		}
	}
	cudaCheckError();
}
}  // namespace yrt
