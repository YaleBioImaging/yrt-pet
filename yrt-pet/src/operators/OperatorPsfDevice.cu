/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorPsfDevice.cuh"

#include <datastruct/image/ImageSpaceKernels.cuh>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void py_setup_operatorpsfdevice(py::module& m)
{
	auto c = py::class_<OperatorPsfDevice, OperatorPsf>(m, "OperatorPsfDevice");
	c.def(py::init<>());
	c.def(py::init<const std::string&>());
	c.def("convolve",
	      static_cast<void (OperatorPsfDevice::*)(
	          const Image* in, Image* out, const std::vector<float>& kernelX,
	          const std::vector<float>& kernelY,
	          const std::vector<float>& kernelZ) const>(
	          &OperatorPsfDevice::convolve));
	c.def("convolve",
	      static_cast<void (OperatorPsfDevice::*)(
	          const ImageDevice& inputImageDevice,
	          ImageDevice& outputImageDevice, const std::vector<float>& kernelX,
	          const std::vector<float>& kernelY,
	          const std::vector<float>& kernelZ) const>(
	          &OperatorPsfDevice::convolve));
	c.def(
	    "applyA",
	    [](OperatorPsfDevice& self, const Image* img_in, Image* img_out)
	    { self.applyA(img_in, img_out); }, py::arg("img_in"),
	    py::arg("img_out"));
	c.def(
	    "applyAH",
	    [](OperatorPsfDevice& self, const Image* img_in, Image* img_out)
	    { self.applyAH(img_in, img_out); }, py::arg("img_in"),
	    py::arg("img_out"));
	c.def(
	    "applyA",
	    [](OperatorPsfDevice& self, const ImageDevice* img_in,
	       ImageDevice* img_out) { self.applyA(img_in, img_out); },
	    py::arg("img_in"), py::arg("img_out"));
	c.def(
	    "applyAH",
	    [](OperatorPsfDevice& self, const ImageDevice* img_in,
	       ImageDevice* img_out) { self.applyAH(img_in, img_out); },
	    py::arg("img_in"), py::arg("img_out"));
}
#endif

OperatorPsfDevice::OperatorPsfDevice() : OperatorPsf{}
{
	initDeviceArraysIfNeeded();
}

OperatorPsfDevice::OperatorPsfDevice(const std::string& imageSpacePsf_fname,
                                     const cudaStream_t* pp_stream)
    : OperatorPsf{}
{
	readFromFileInternal(imageSpacePsf_fname, pp_stream);
}

void OperatorPsfDevice::readFromFileInternal(
    const std::string& imageSpacePsf_fname, const cudaStream_t* pp_stream)
{
	OperatorPsf::readFromFile(imageSpacePsf_fname);
	copyToDevice(pp_stream);
}

void OperatorPsfDevice::readFromFile(const std::string& imageSpacePsf_fname)
{
	readFromFileInternal(imageSpacePsf_fname, nullptr);
}

void OperatorPsfDevice::readFromFile(const std::string& imageSpacePsf_fname,
                                     const cudaStream_t* pp_stream)
{
	readFromFileInternal(imageSpacePsf_fname, pp_stream);
}

void OperatorPsfDevice::copyToDevice(const cudaStream_t* pp_stream)
{
	initDeviceArraysIfNeeded();
	allocateDeviceArrays(pp_stream, m_synchronized);

	mpd_kernelX->copyFromHost(m_kernelX.data(), m_kernelX.size(), pp_stream,
	                          m_synchronized);
	mpd_kernelY->copyFromHost(m_kernelY.data(), m_kernelY.size(), pp_stream,
	                          m_synchronized);
	mpd_kernelZ->copyFromHost(m_kernelZ.data(), m_kernelZ.size(), pp_stream,
	                          m_synchronized);
	mpd_kernelX_flipped->copyFromHost(m_kernelX_flipped.data(),
	                                  m_kernelX_flipped.size(), pp_stream,
	                                  m_synchronized);
	mpd_kernelY_flipped->copyFromHost(m_kernelY_flipped.data(),
	                                  m_kernelY_flipped.size(), pp_stream,
	                                  m_synchronized);
	mpd_kernelZ_flipped->copyFromHost(m_kernelZ_flipped.data(),
	                                  m_kernelZ_flipped.size(), pp_stream,
	                                  m_synchronized);
}

template <bool Transpose>
void OperatorPsfDevice::apply(const Variable* in, Variable* out) const
{
	const auto img_in = dynamic_cast<const Image*>(in);
	auto img_out = dynamic_cast<Image*>(out);

	std::unique_ptr<ImageDevice> inputImageDevice;
	const ImageDevice* inputImageDevice_ptr;
	if (img_in != nullptr)
	{
		// Input image is in host
		inputImageDevice =
		    std::make_unique<ImageDeviceOwned>(img_in->getParams());
		reinterpret_cast<ImageDeviceOwned*>(inputImageDevice.get())->allocate();
		inputImageDevice->transferToDeviceMemory(img_in, m_synchronized);
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
		// Input image is in host
		outputImageDevice =
		    std::make_unique<ImageDeviceOwned>(img_out->getParams());
		reinterpret_cast<ImageDeviceOwned*>(outputImageDevice.get())
		    ->allocate();
		outputImageDevice_ptr = outputImageDevice.get();
	}
	else
	{
		outputImageDevice_ptr = dynamic_cast<ImageDevice*>(out);
		ASSERT_MSG(outputImageDevice_ptr, "Output is not an image");
	}

	convolveDevice<Transpose>(*inputImageDevice_ptr, *outputImageDevice_ptr);

	// Transfer to host
	if (img_out != nullptr)
	{
		outputImageDevice->transferToHostMemory(img_out, m_synchronized);
	}
}

void OperatorPsfDevice::applyA(const Variable* in, Variable* out)
{
	apply<false>(in, out);
}

void OperatorPsfDevice::applyAH(const Variable* in, Variable* out)
{
	apply<true>(in, out);
}

void OperatorPsfDevice::initDeviceArrayIfNeeded(
    std::unique_ptr<DeviceArray<float>>& ppd_kernel)
{
	if (ppd_kernel == nullptr)
	{
		ppd_kernel = std::make_unique<DeviceArray<float>>();
	}
}

void OperatorPsfDevice::allocateDeviceArray(DeviceArray<float>& prd_kernel,
                                            size_t newSize,
                                            const cudaStream_t* stream,
                                            bool synchronize)
{
	prd_kernel.allocate(newSize, stream, synchronize);
}

void OperatorPsfDevice::initDeviceArraysIfNeeded()
{
	initDeviceArrayIfNeeded(mpd_kernelX);
	initDeviceArrayIfNeeded(mpd_kernelY);
	initDeviceArrayIfNeeded(mpd_kernelZ);
	initDeviceArrayIfNeeded(mpd_kernelX_flipped);
	initDeviceArrayIfNeeded(mpd_kernelY_flipped);
	initDeviceArrayIfNeeded(mpd_kernelZ_flipped);
}

void OperatorPsfDevice::allocateDeviceArrays(const cudaStream_t* stream,
                                             bool synchronize)
{
	allocateDeviceArray(*mpd_kernelX, m_kernelX.size(), stream, synchronize);
	allocateDeviceArray(*mpd_kernelY, m_kernelY.size(), stream, synchronize);
	allocateDeviceArray(*mpd_kernelZ, m_kernelZ.size(), stream, synchronize);
	allocateDeviceArray(*mpd_kernelX_flipped, m_kernelX.size(), stream,
	                    synchronize);
	allocateDeviceArray(*mpd_kernelY_flipped, m_kernelY.size(), stream,
	                    synchronize);
	allocateDeviceArray(*mpd_kernelZ_flipped, m_kernelZ.size(), stream,
	                    synchronize);
}

void OperatorPsfDevice::convolve(const ImageDevice& inputImageDevice,
                                 ImageDevice& outputImageDevice,
                                 const std::vector<float>& kernelX,
                                 const std::vector<float>& kernelY,
                                 const std::vector<float>& kernelZ) const
{
	DeviceArray<float> d_kernelX{kernelX.size(), mp_auxStream};
	d_kernelX.copyFromHost(kernelX.data(), kernelX.size(), mp_auxStream,
	                       m_synchronized);

	DeviceArray<float> d_kernelY{kernelY.size(), mp_auxStream};
	d_kernelY.copyFromHost(kernelY.data(), kernelY.size(), mp_auxStream,
	                       m_synchronized);

	DeviceArray<float> d_kernelZ{kernelZ.size(), mp_auxStream};
	d_kernelZ.copyFromHost(kernelZ.data(), kernelZ.size(), mp_auxStream, true);

	convolveDevice(inputImageDevice, outputImageDevice, d_kernelX, d_kernelY,
	               d_kernelZ, mp_mainStream, true);
}

void OperatorPsfDevice::convolve(const Image* in, Image* out,
                                 const std::vector<float>& kernelX,
                                 const std::vector<float>& kernelY,
                                 const std::vector<float>& kernelZ) const
{
	ASSERT(in != nullptr);
	ASSERT(out != nullptr);

	ImageDeviceOwned inputImageDevice{in->getParams(), mp_auxStream};
	inputImageDevice.allocate();
	inputImageDevice.transferToDeviceMemory(in, m_synchronized);

	ImageDeviceOwned outputImageDevice{out->getParams(), mp_auxStream};
	outputImageDevice.allocate();

	convolve(inputImageDevice, outputImageDevice, kernelX, kernelY, kernelZ);

	outputImageDevice.transferToHostMemory(out, m_synchronized);
}

template <bool Flipped>
void OperatorPsfDevice::convolveDevice(const ImageDevice& inputImage,
                                       ImageDevice& outputImage) const
{
	if constexpr (Flipped)
	{
		convolveDevice(inputImage, outputImage, *mpd_kernelZ_flipped,
		               *mpd_kernelZ_flipped, *mpd_kernelZ_flipped,
		               mp_mainStream, m_synchronized);
	}
	else
	{
		convolveDevice(inputImage, outputImage, *mpd_kernelX, *mpd_kernelY,
		               *mpd_kernelZ, mp_mainStream, m_synchronized);
	}
}
template void
    OperatorPsfDevice::convolveDevice<false>(const ImageDevice& inputImage,
                                             ImageDevice& outputImage) const;
template void
    OperatorPsfDevice::convolveDevice<true>(const ImageDevice& inputImage,
                                            ImageDevice& outputImage) const;

void OperatorPsfDevice::convolveDevice(const ImageDevice& inputImage,
                                       ImageDevice& outputImage,
                                       const DeviceArray<float>& kernelX,
                                       const DeviceArray<float>& kernelY,
                                       const DeviceArray<float>& kernelZ,
                                       const cudaStream_t* stream,
                                       bool synchronize)
{
	const ImageParams& params = inputImage.getParams();
	ASSERT_MSG(params.isSameDimensionsAs(outputImage.getParams()),
	           "Image parameters mismatch");

	const float* pd_inputImage = inputImage.getDevicePointer();
	float* pd_outputImage = outputImage.getDevicePointer();
	ASSERT_MSG(pd_inputImage != nullptr,
	           "Input device Image not allocated yet");
	ASSERT_MSG(pd_outputImage != nullptr,
	           "Output device Image not allocated yet");

	ImageDeviceOwned indermediaryImage{params};
	indermediaryImage.allocate(synchronize);
	float* pd_indermediaryImage = indermediaryImage.getDevicePointer();

	const float* pd_kernelX = kernelX.getDevicePointer();
	const float* pd_kernelY = kernelY.getDevicePointer();
	const float* pd_kernelZ = kernelZ.getDevicePointer();
	ASSERT_MSG(pd_kernelX != nullptr && pd_kernelY != nullptr &&
	               pd_kernelZ != nullptr,
	           "Convolution kernel not initialized");
	const std::array<size_t, 3> kerSize{kernelX.getSize(), kernelY.getSize(),
	                                    kernelZ.getSize()};
	ASSERT_MSG(kerSize[0] % 2 != 0, "Kernel size must be odd");
	ASSERT_MSG(kerSize[1] % 2 != 0, "Kernel size must be odd");
	ASSERT_MSG(kerSize[2] % 2 != 0, "Kernel size must be odd");

	GPULaunchParams3D launchParams = inputImage.getLaunchParams();

	if (stream != nullptr)
	{
		// Convolve along X-axis
		convolve3DSeparable_kernel<0>
		    <<<launchParams.gridSize, launchParams.blockSize, 0, *stream>>>(
		        pd_inputImage, pd_outputImage, pd_kernelX, kerSize[0],
		        params.nx, params.ny, params.nz);

		// Convolve along Y-axis
		convolve3DSeparable_kernel<1>
		    <<<launchParams.gridSize, launchParams.blockSize, 0, *stream>>>(
		        pd_outputImage, pd_indermediaryImage, pd_kernelY, kerSize[1],
		        params.nx, params.ny, params.nz);

		// Convolve along Z-axis
		convolve3DSeparable_kernel<2>
		    <<<launchParams.gridSize, launchParams.blockSize, 0, *stream>>>(
		        pd_indermediaryImage, pd_outputImage, pd_kernelZ, kerSize[2],
		        params.nx, params.ny, params.nz);

		if (synchronize)
		{
			cudaStreamSynchronize(*stream);
		}
	}
	else
	{
		// Convolve along X-axis
		convolve3DSeparable_kernel<0>
		    <<<launchParams.gridSize, launchParams.blockSize, 0>>>(
		        pd_inputImage, pd_outputImage, pd_kernelX, kerSize[0],
		        params.nx, params.ny, params.nz);

		// Convolve along Y-axis
		convolve3DSeparable_kernel<1>
		    <<<launchParams.gridSize, launchParams.blockSize, 0>>>(
		        pd_outputImage, pd_indermediaryImage, pd_kernelY, kerSize[1],
		        params.nx, params.ny, params.nz);

		// Convolve along Z-axis
		convolve3DSeparable_kernel<2>
		    <<<launchParams.gridSize, launchParams.blockSize, 0>>>(
		        pd_indermediaryImage, pd_outputImage, pd_kernelZ, kerSize[2],
		        params.nx, params.ny, params.nz);

		if (synchronize)
		{
			cudaDeviceSynchronize();
		}
	}

	cudaCheckError();
}
