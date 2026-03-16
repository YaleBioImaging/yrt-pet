/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorVarPsfDevice.cuh"
#include <vector>
#include <yrt-pet/datastruct/image/ImageDevice.cuh>
#include <yrt-pet/datastruct/image/ImageSpaceKernels.cuh>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace yrt
{
void py_setup_operatorvarpsfdevice(py::module& m)
{
	auto c = py::class_<OperatorVarPsfDevice, OperatorVarPsf>(
	    m, "OperatorVarPsfDevice");
	c.def(py::init<const ImageParams&>());
	c.def(py::init<const std::string&, const ImageParams&>());
	c.def("readFromFile", &OperatorVarPsf::readFromFile, py::arg("fname"),
	      "Read the variant PSF from CSV LUT");
	c.def("copyVarPsfToDevice", &OperatorVarPsfDevice::copyVarPsfToDevice,
	      py::arg("synchronize") = true, "Upload the PSF kernels to the GPU");
	c.def(
	    "applyA",
	    [](OperatorVarPsfDevice& self, const Image* img_in, Image* img_out)
	    { self.applyA(img_in, img_out); }, py::arg("img_in"),
	    py::arg("img_out"),
	    "Apply the forward variable PSF operator on host images");
	c.def(
	    "applyAH",
	    [](OperatorVarPsfDevice& self, const Image* img_in, Image* img_out)
	    { self.applyAH(img_in, img_out); }, py::arg("img_in"),
	    py::arg("img_out"),
	    "Apply the adjoint variable PSF operator on host images");
	c.def(
	    "applyA",
	    [](OperatorVarPsfDevice& self, const ImageDevice* img_in,
	       ImageDevice* img_out) { self.applyA(img_in, img_out); },
	    py::arg("img_in"), py::arg("img_out"),
	    "Apply the forward variable PSF operator on device images");
	c.def(
	    "applyAH",
	    [](OperatorVarPsfDevice& self, const ImageDevice* img_in,
	       ImageDevice* img_out) { self.applyAH(img_in, img_out); },
	    py::arg("img_in"), py::arg("img_out"),
	    "Apply the adjoint variable PSF operator on device images");
}
}  // namespace yrt
#endif

namespace yrt
{

OperatorVarPsfDevice::OperatorVarPsfDevice(const ImageParams& p_imageParams,
                                           const cudaStream_t* pp_stream)
    : DeviceSynchronized{pp_stream, pp_stream}, OperatorVarPsf{p_imageParams}
{
	initDeviceArraysIfNeeded();
}

OperatorVarPsfDevice::OperatorVarPsfDevice(const std::string& pr_imagePsf_fname,
                                           const ImageParams& p_imageParams,
                                           const cudaStream_t* pp_stream)
    : DeviceSynchronized{pp_stream, pp_stream},
      OperatorVarPsf{pr_imagePsf_fname, p_imageParams}
{
	initDeviceArraysIfNeeded();
	copyVarPsfToDevice(true);
}

void OperatorVarPsfDevice::initDeviceArraysIfNeeded()
{
	if (!mpd_kernelsFlat)
		mpd_kernelsFlat = std::make_unique<DeviceArray<float>>();
	if (!mpd_kernelOffsets)
		mpd_kernelOffsets = std::make_unique<DeviceArray<int>>();
	if (!mpd_kernelDims)
		mpd_kernelDims = std::make_unique<DeviceArray<int>>();
	if (!mpd_kernelHalfSizes)
		mpd_kernelHalfSizes = std::make_unique<DeviceArray<int>>();
}

void OperatorVarPsfDevice::allocateDeviceArraysVarPsf(size_t nKernels,
                                                      size_t totalKernelSize,
                                                      bool synchronize)
{
	initDeviceArraysIfNeeded();
	// allocate storage for flattened kernels and associated metadata
	mpd_kernelsFlat->allocate(totalKernelSize, {getAuxStream(), synchronize});
	mpd_kernelOffsets->allocate(nKernels, {getAuxStream(), synchronize});
	mpd_kernelDims->allocate(nKernels * 3, {getAuxStream(), synchronize});
	mpd_kernelHalfSizes->allocate(nKernels * 3, {getAuxStream(), synchronize});
}

void OperatorVarPsfDevice::copyVarPsfToDevice(bool synchronize)
{
	const size_t nKernels = m_kernelLUT.size();
	// end copy if no PSF defined
	if (nKernels == 0)
		return;

	std::vector<int> offsets(nKernels);
	std::vector<int> dims(nKernels * 3);
	std::vector<int> halfSizes(nKernels * 3);
	std::vector<float> flatKernels;
	size_t currentOffset = 0;
	for (size_t k = 0; k < nKernels; ++k)
	{
		const auto& kernel = *m_kernelLUT[k];
		const auto& arr = kernel.getArray();
		const auto dimsArr = arr.getDims();
		const int sz = static_cast<int>(dimsArr[0]);
		const int sy = static_cast<int>(dimsArr[1]);
		const int sx = static_cast<int>(dimsArr[2]);
		const size_t kernelSize = static_cast<size_t>(sx) *
		                          static_cast<size_t>(sy) *
		                          static_cast<size_t>(sz);
		offsets[k] = static_cast<int>(currentOffset);
		dims[3 * k] = sx;
		dims[3 * k + 1] = sy;
		dims[3 * k + 2] = sz;
		halfSizes[3 * k] = (sx - 1) / 2;
		halfSizes[3 * k + 1] = (sy - 1) / 2;
		halfSizes[3 * k + 2] = (sz - 1) / 2;
		const float* raw = arr.getRawPointer();
		flatKernels.insert(flatKernels.end(), raw, raw + kernelSize);
		currentOffset += kernelSize;
	}

	// Allocate device arrays and copy data
	allocateDeviceArraysVarPsf(nKernels, flatKernels.size(), synchronize);
	mpd_kernelsFlat->copyFromHost(flatKernels.data(), flatKernels.size(),
	                              {getAuxStream(), synchronize});
	mpd_kernelOffsets->copyFromHost(offsets.data(), offsets.size(),
	                                {getAuxStream(), synchronize});
	mpd_kernelDims->copyFromHost(dims.data(), dims.size(),
	                             {getAuxStream(), synchronize});
	mpd_kernelHalfSizes->copyFromHost(halfSizes.data(), halfSizes.size(),
	                                  {getAuxStream(), synchronize});

	// Compute LUT dimensions and store gap parameters (in mm)
	lut_x_dim = static_cast<int>(std::floor(m_xRange / m_xGap)) + 1;
	lut_y_dim = static_cast<int>(std::floor(m_yRange / m_yGap)) + 1;
	lut_z_dim = static_cast<int>(std::floor(m_zRange / m_zGap)) + 1;
	d_xGap = m_xGap;
	d_yGap = m_yGap;
	d_zGap = m_zGap;

	if (synchronize)
	{
		const auto* stream = getAuxStream();
		if (stream)
			cudaStreamSynchronize(*stream);
		else
			cudaDeviceSynchronize();
	}
}

void OperatorVarPsfDevice::applyA(const Variable* in, Variable* out)
{
	applyA(in, out, true);
}

void OperatorVarPsfDevice::applyAH(const Variable* in, Variable* out)
{
	applyAH(in, out, true);
}

void OperatorVarPsfDevice::applyA(const Variable* in, Variable* out,
                                  bool synchronize) const
{
	apply<false>(in, out, synchronize);
}

void OperatorVarPsfDevice::applyAH(const Variable* in, Variable* out,
                                   bool synchronize) const
{
	apply<true>(in, out, synchronize);
}

template <bool Transpose>
void OperatorVarPsfDevice::apply(const Variable* in, Variable* out,
                                 bool synchronize) const
{
	// Determine whether the input/output are host or device images
	const auto* img_in = dynamic_cast<const Image*>(in);
	auto* img_out = dynamic_cast<Image*>(out);
	std::unique_ptr<ImageDevice> inputImageDevice;
	const ImageDevice* inputImageDevice_ptr;
	if (img_in != nullptr)
	{
		// Host input: copy to device
		inputImageDevice = std::make_unique<ImageDeviceOwned>(
		    img_in->getParams(), getMainStream());
		reinterpret_cast<ImageDeviceOwned*>(inputImageDevice.get())->allocate();
		inputImageDevice->transferToDeviceMemory(img_in, synchronize);
		inputImageDevice_ptr = inputImageDevice.get();
	}
	else
	{
		inputImageDevice_ptr = dynamic_cast<const ImageDevice*>(in);
		ASSERT_MSG(inputImageDevice_ptr, "Input is not an image");
	}

	std::unique_ptr<ImageDevice> outputImageDevice;
	ImageDevice* outputImageDevice_ptr;
	if (img_out != nullptr)
	{
		// Host output: allocate device buffer
		outputImageDevice = std::make_unique<ImageDeviceOwned>(
		    img_out->getParams(), getMainStream());
		reinterpret_cast<ImageDeviceOwned*>(outputImageDevice.get())
		    ->allocate();
		outputImageDevice_ptr = outputImageDevice.get();
	}
	else
	{
		outputImageDevice_ptr = dynamic_cast<ImageDevice*>(out);
		ASSERT_MSG(outputImageDevice_ptr, "Output is not an image");
	}

	if constexpr (Transpose)
	{
		varconvolveDevice<false>(*inputImageDevice_ptr, *outputImageDevice_ptr,
		                         false);
	}
	else
	{
		varconvolveDevice<true>(*inputImageDevice_ptr, *outputImageDevice_ptr,
		                        false);
	}

	// Transfer back to host
	if (img_out != nullptr)
	{
		outputImageDevice->transferToHostMemory(img_out, false);
	}

	// Synchronise the main stream if requested
	if (synchronize)
	{
		const auto* stream = getMainStream();
		if (stream != nullptr)
			cudaStreamSynchronize(*stream);
		else
			cudaDeviceSynchronize();
	}
}

template void OperatorVarPsfDevice::apply<false>(const Variable* in,
                                                 Variable* out,
                                                 bool synchronize) const;
template void OperatorVarPsfDevice::apply<true>(const Variable* in,
                                                Variable* out,
                                                bool synchronize) const;

template <bool IS_FWD>
void OperatorVarPsfDevice::varconvolveDevice(const ImageDevice& inputImage,
                                             ImageDevice& outputImage,
                                             bool synchronize) const
{
	const float* pd_input = inputImage.getDevicePointer();
	float* pd_output = outputImage.getDevicePointer();
	ASSERT_MSG(pd_input != nullptr && pd_output != nullptr,
	           "Device image pointers must not be null");
	const float* pd_kernelsFlat = mpd_kernelsFlat->getDevicePointer();
	const int* pd_kernelOffsets = mpd_kernelOffsets->getDevicePointer();
	const int* pd_kernelDims = mpd_kernelDims->getDevicePointer();
	const int* pd_kernelHalf = mpd_kernelHalfSizes->getDevicePointer();

	const int nx = inputImage.getParams().nx;
	const int ny = inputImage.getParams().ny;
	const int nz = inputImage.getParams().nz;
	const float vx = inputImage.getParams().vx;
	const float vy = inputImage.getParams().vy;
	const float vz = inputImage.getParams().vz;
	const float xCenter = static_cast<float>(nx) * vx / 2.0f;
	const float yCenter = static_cast<float>(ny) * vy / 2.0f;
	const float zCenter = static_cast<float>(nz) * vz / 2.0f;

	const cudaStream_t* stream = getMainStream();
	const GPULaunchParams3D launchParams = inputImage.getLaunchParams();

	if (stream != nullptr)
	{
		if constexpr (IS_FWD)
		{
			convolve3D_kernel<false>
			    <<<launchParams.gridSize, launchParams.blockSize, 0, *stream>>>(
			        pd_input, pd_output, pd_kernelsFlat, pd_kernelOffsets,
			        pd_kernelDims, pd_kernelHalf, lut_x_dim, lut_y_dim,
			        lut_z_dim, d_xGap, d_yGap, d_zGap, xCenter, yCenter,
			        zCenter, vx, vy, vz, nx, ny, nz);
			if (synchronize)
			{
				cudaStreamSynchronize(*stream);
			}
		}
		else
		{
			convolve3D_kernel<true>
			    <<<launchParams.gridSize, launchParams.blockSize, 0, *stream>>>(
			        pd_input, pd_output, pd_kernelsFlat, pd_kernelOffsets,
			        pd_kernelDims, pd_kernelHalf, lut_x_dim, lut_y_dim,
			        lut_z_dim, d_xGap, d_yGap, d_zGap, xCenter, yCenter,
			        zCenter, vx, vy, vz, nx, ny, nz);
			if (synchronize)
			{
				cudaStreamSynchronize(*stream);
			}
		}
	}
	else
	{
		if constexpr (IS_FWD)
		{
			convolve3D_kernel<false>
			    <<<launchParams.gridSize, launchParams.blockSize, 0>>>(
			        pd_input, pd_output, pd_kernelsFlat, pd_kernelOffsets,
			        pd_kernelDims, pd_kernelHalf, lut_x_dim, lut_y_dim,
			        lut_z_dim, d_xGap, d_yGap, d_zGap, xCenter, yCenter,
			        zCenter, vx, vy, vz, nx, ny, nz);
			if (synchronize)
			{
				cudaDeviceSynchronize();
			}
		}
		else
		{
			convolve3D_kernel<true>
			    <<<launchParams.gridSize, launchParams.blockSize, 0>>>(
			        pd_input, pd_output, pd_kernelsFlat, pd_kernelOffsets,
			        pd_kernelDims, pd_kernelHalf, lut_x_dim, lut_y_dim,
			        lut_z_dim, d_xGap, d_yGap, d_zGap, xCenter, yCenter,
			        zCenter, vx, vy, vz, nx, ny, nz);
			if (synchronize)
			{
				cudaDeviceSynchronize();
			}
		}
	}
	cudaCheckError();
}

template void
    OperatorVarPsfDevice::varconvolveDevice<true>(const ImageDevice& inputImage,
                                                  ImageDevice& outputImage,
                                                  bool synchronize) const;
template void OperatorVarPsfDevice::varconvolveDevice<false>(
    const ImageDevice& inputImage, ImageDevice& outputImage,
    bool synchronize) const;

}  // namespace yrt
