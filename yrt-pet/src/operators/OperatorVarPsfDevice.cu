/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorVarPsfDevice.cuh"

#include <datastruct/image/ImageSpaceKernels.cuh>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#incldue <>
namespace py = pybind11;

namespace yrt
{
void py_setup_operatorpsfdevice(py::module& m)
{
	auto c = py::class_<OperatorVarPsfDevice, OperatorVarPsf>(m, "OperatorVarPsfDevice");
	c.def(py::init<const ImageParams&>());
	c.def(py::init<const std::string&, const ImageParams&>());
	c.def("readFromFile", &OperatorVarPsfDevice::readFromFile, py::arg("fname"),
		  "Read the variant PSF from CSV LUT");
	c.def(
		"applyA",
		[](OperatorVarPsfDevice& self, const Image* img_in, Image* img_out)
		{ self.applyA(img_in, img_out); }, py::arg("img_in"),
		py::arg("img_out"));
	c.def(
		"applyAH",
		[](OperatorVarPsfDevice& self, const Image* img_in, Image* img_out)
		{ self.applyAH(img_in, img_out); }, py::arg("img_in"),
		py::arg("img_out"));
	c.def(
		"applyA",
		[](OperatorVarPsfDevice& self, const ImageDevice* img_in,
		   ImageDevice* img_out) { self.applyA(img_in, img_out); },
		py::arg("img_in"), py::arg("img_out"));
	c.def(
		"applyAH",
		[](OperatorVarPsfDevice& self, const ImageDevice* img_in,
		   ImageDevice* img_out) { self.applyAH(img_in, img_out); },
		py::arg("img_in"), py::arg("img_out"));
}
}
#endif

namespace yrt
{

OperatorVarPsfDevice::OperatorVarPsfDevice(const cudaStream_t* pp_stream, const ImageParams& p_imageParams)
    : DeviceSynchronized{pp_stream, pp_stream},
      OperatorVarPsf{}
{
	initDeviceArraysIfNeeded();
}

OperatorVarPsfDevice::OperatorVarPsfDevice(const std::string& pr_imagePsf_fname,
                                     const cudaStream_t* pp_stream, const ImageParams& p_imageParams)
    : OperatorVarPsfDevice{pp_stream}
{
	OperatorVarPsf::readFromFileInternal(pr_imagePsf_fname);
	copyVarPsfToDevice(true);
}

DeviceVarPsf OperatorVarPsfDevice::copyVarPsfToDevice(const yrt::OperatorVarPsf& cpuVarPsf, bool synchronize)
{
	initDeviceArraysIfNeeded();
	allocateDeviceArrays(synchronize);

	using KernelPtr = std::unique_ptr<yrt::ConvolutionKernel>;
	const auto& kernelLUT = cpuVarPsf.m_kernelLUT;

	int numKernels = static_cast<int>(kernelLUT.size());
	std::vector<int> sizes(numKernels);
	std::vector<int> offsets(numKernels);

	//flatten kernel
	std::vector<float> allData;
	int currentOffset = 0;
	for (int i = 0; i < numKernels; i++) {
		const auto& kernel = *kernelLUT[i];
		const auto& arr = kernel.getArray();

		int sx = arr.getSizeX();
		int sy = arr.getSizeY();
		int sz = arr.getSizeZ();
		int size = sx * sy * sz;

		sizes[i] = size;
		offsets[i] = currentOffset;

		for (int z = 0; z < sz; z++) {
			for (int y = 0; y < sy; y++) {
				for (int x = 0; x < sx; x++) {
					allData.push_back(arr(x,y,z));
				}
			}
		}
		currentOffset += size;
	}

	//allocate GPU memory
	DeviceVarPsf dVarPsf;
	dVarPsf.numKernels = numKernels;

	cudaMalloc(&dVarPsf.kernels, allData.size() * sizeof(float));
	cudaMalloc(&dVarPsf.offsets, numKernels * sizeof(int));
	cudaMalloc(&dVarPsf.sizes,   numKernels * sizeof(int));

	cudaMemcpy(dVarPsf.kernels, allData.data(),
			   allData.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dVarPsf.offsets, offsets.data(),
			   numKernels * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dVarPsf.sizes, sizes.data(),
			   numKernels * sizeof(int), cudaMemcpyHostToDevice);

	return dVarPsf;
	//?? free device memory after using?
}

void OperatorVarPsfDevice::initDeviceArrayIfNeeded(
	std::unique_ptr<DeviceArray<float>>& ppd_kernel)
{
	if (ppd_kernel == nullptr)
	{
		ppd_kernel = std::make_unique<DeviceArray<float>>();
	}
}

void OperatorVarPsfDevice::initDeviceArraysIfNeeded()
{
	initDeviceArrayIfNeeded(mpd_kernelsFlat);
	initDeviceArrayIfNeeded(mpd_kernelOffsets);
	initDeviceArrayIfNeeded(mpd_kernelDims);
	initDeviceArrayIfNeeded(mpd_kernelHalfSizes);
	initDeviceArrayIfNeeded(mpd_kernelLUT);
}

void OperatorVarPsfDevice::allocateDeviceArraysVarPsf(size_t nKernels, size_t totalKernelSize, bool synchronize)
//??not sure if need
{
	initDeviceArraysIfNeeded();

	mpd_kernelsFlat->allocate(totalKernelSize, {getAuxStream(), synchronize});
	mpd_kernelOffsets->allocate(nKernels, {getAuxStream(), synchronize});
	mpd_kernelDims->allocate(nKernels * 3, {getAuxStream(), synchronize});
	mpd_kernelHalfSizes->allocate(nKernels * 3, {getAuxStream(), synchronize});

	// LUT dims
	int x_dim = static_cast<int>(std::floor(m_xRange / m_xGap)) + 1;
	int y_dim = static_cast<int>(std::floor(m_yRange / m_yGap)) + 1;
	int z_dim = static_cast<int>(std::floor(m_zRange / m_zGap)) + 1;
	mpd_kernelLUT->allocate(static_cast<size_t>(x_dim) * y_dim * z_dim, {getAuxStream(), synchronize});
}

template <bool IS_FWD>
void OperatorPsfDevice::varconvolveDevice(const ImageDevice& inputImage,
									   ImageDevice& outputImage, bool synchronize) const
{
	const float* pd_input = inputImage.getDevicePointer();
	float* pd_output = outputImage.getDevicePointer();
	const float* pd_kernelsFlat = mpd_kernelsFlat->getDevicePointer();
	const int* pd_kernelOffsets = mpd_kernelOffsets->getDevicePointer();
	const int* pd_kernelDims = mpd_kernelDims->getDevicePointer();
	const int* pd_kernelHalf = mpd_kernelHalfSizes->getDevicePointer();
	const int* pd_kernelLUT = mpd_kernelLUT->getDevicePointer();

	int nx = inputImage.getParams().nx;
	int ny = inputImage.getParams().ny;
	int nz = inputImage.getParams().nz;

	const cudaStream_t* stream = getMainStream();

	if constexpr (IS_FWD)
	{
		convolve3D_kernel_F(pd_input, pd_output,
			pd_kernelsFlat, pd_kernelOffsets, pd_kernelDims, pd_kernelHalf,
			pd_kernelLUT,
			lut_x_dim, lut_y_dim, lut_z_dim,
			d_xGap, d_yGap, d_zGap, d_xCenter, d_yCenter, d_zCenter,
			nx, ny, nz);
	}
	else
	{
		//cudaMemsetAsync(pd_output, 0, sizeof(float)*total, (stream? *stream: 0));
		convolve3D_kernel_T(pd_input, pd_output,
			pd_kernelsFlat, pd_kernelOffsets, pd_kernelDims, pd_kernelHalf,
			pd_kernelLUT,
			lut_x_dim, lut_y_dim, lut_z_dim,
			d_xGap, d_yGap, d_zGap, d_xCenter, d_yCenter, d_zCenter,
			nx, ny, nz);
	}
	if (synchronize) {
		if (stream) cudaStreamSynchronize(*stream);
		else cudaDeviceSynchronize();
	}
	cudaCheckError();
}

/*void OperatorVarPsfDevice::applyA(const Variable* in, Variable* out)
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
}*/

template <bool Transpose>
void OperatorVarPsfDevice::apply(const Variable* in, Variable* out,
							  bool synchronize) const
{
	const auto img_in = dynamic_cast<const Image*>(in);
	auto img_out = dynamic_cast<Image*>(out);

	std::unique_ptr<ImageDevice> inputImageDevice;
	const ImageDevice* inputImageDevice_ptr;
	if (img_in != nullptr)
	{
		// Input image is in host
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
		// Input image is in host
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

	varconvolveDevice<Transpose>(*inputImageDevice_ptr, *outputImageDevice_ptr,
							  false);

	// Transfer to host
	if (img_out != nullptr)
	{
		outputImageDevice->transferToHostMemory(img_out, false);
	}

	if (synchronize)
	{
		if (getMainStream() != nullptr)
		{
			cudaStreamSynchronize(*getMainStream());
		}
		else
		{
			cudaDeviceSynchronize();
		}
	}
}

void OperatorVarPsfDevice::initDeviceArrayIfNeeded(
    std::unique_ptr<DeviceArray<float>>& ppd_kernel)
{
	if (ppd_kernel == nullptr)
	{
		ppd_kernel = std::make_unique<DeviceArray<float>>();
	}
}

void OperatorVarPsfDevice::allocateDeviceArray(DeviceArray<float>& prd_kernel,
                                            size_t newSize, bool synchronize)
{
	prd_kernel.allocate(newSize, {getAuxStream(), synchronize});
}

void OperatorVarPsfDevice::initDeviceArraysIfNeeded()
{
	initDeviceArrayIfNeeded(mpd_kernelX);
	initDeviceArrayIfNeeded(mpd_kernelY);
	initDeviceArrayIfNeeded(mpd_kernelZ);
	initDeviceArrayIfNeeded(mpd_kernelX_flipped);
	initDeviceArrayIfNeeded(mpd_kernelY_flipped);
	initDeviceArrayIfNeeded(mpd_kernelZ_flipped);
}

//10.2 write to here
void OperatorVarPsfDevice::convolve(const ImageDevice& inputImageDevice,
                                 ImageDevice& outputImageDevice,
                                 const std::vector<float>& kernelX,
                                 const std::vector<float>& kernelY,
                                 const std::vector<float>& kernelZ,
                                 bool synchronize) const
{
	DeviceArray<float> d_kernelX{kernelX.size(), mp_auxStream};
	d_kernelX.copyFromHost(kernelX.data(), kernelX.size(),
	                       {mp_auxStream, false});

	DeviceArray<float> d_kernelY{kernelY.size(), mp_auxStream};
	d_kernelY.copyFromHost(kernelY.data(), kernelY.size(),
	                       {mp_auxStream, false});

	DeviceArray<float> d_kernelZ{kernelZ.size(), mp_auxStream};
	d_kernelZ.copyFromHost(kernelZ.data(), kernelZ.size(),
	                       {mp_auxStream, true});

	convolveDevice(inputImageDevice, outputImageDevice, d_kernelX, d_kernelY,
	               d_kernelZ, synchronize);
}

void OperatorVarPsfDevice::convolve(const Image* in, Image* out,
                                 const std::vector<float>& kernelX,
                                 const std::vector<float>& kernelY,
                                 const std::vector<float>& kernelZ) const
{
	// By default, synchronize
	convolve(in, out, kernelX, kernelY, kernelZ, true);
}

void OperatorVarPsfDevice::convolve(const Image* in, Image* out,
                                 const std::vector<float>& kernelX,
                                 const std::vector<float>& kernelY,
                                 const std::vector<float>& kernelZ,
                                 bool synchronize) const
{
	ASSERT(in != nullptr);
	ASSERT(out != nullptr);

	ImageDeviceOwned inputImageDevice{in->getParams(), mp_auxStream};
	inputImageDevice.allocate(false);
	inputImageDevice.transferToDeviceMemory(in, false);

	ImageDeviceOwned outputImageDevice{out->getParams(), mp_auxStream};
	outputImageDevice.allocate(false);

	convolve(inputImageDevice, outputImageDevice, kernelX, kernelY, kernelZ,
	         synchronize);

	outputImageDevice.transferToHostMemory(out, synchronize);
}

template <bool Flipped>
void OperatorVarPsfDevice::convolveDevice(const ImageDevice& inputImage,
                                       ImageDevice& outputImage,
                                       bool synchronize) const
{
	if constexpr (Flipped)
	{
		convolveDevice(inputImage, outputImage, *mpd_kernelX_flipped,
		               *mpd_kernelY_flipped, *mpd_kernelZ_flipped, synchronize);
	}
	else
	{
		convolveDevice(inputImage, outputImage, *mpd_kernelX, *mpd_kernelY,
		               *mpd_kernelZ, synchronize);
	}
}
template void
OperatorVarPsfDevice::convolveDevice<false>(const ImageDevice& inputImage,
                                             ImageDevice& outputImage,
                                             bool synchronize) const;
template void
OperatorVarPsfDevice::convolveDevice<true>(const ImageDevice& inputImage,
                                            ImageDevice& outputImage,
                                            bool synchronize) const;

void OperatorVarPsfDevice::convolveDevice(const ImageDevice& inputImage,
                                       ImageDevice& outputImage,
                                       const DeviceArray<float>& kernelX,
                                       const DeviceArray<float>& kernelY,
                                       const DeviceArray<float>& kernelZ,
                                       bool synchronize) const
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

	// Everything here is done in the main stream
	const auto* stream = getMainStream();

	allocateTemporaryDeviceImageIfNeeded(params, {stream, true});
	float* pd_intermediaryImage = mpd_intermediaryImage->getDevicePointer();

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
	ASSERT_MSG(kerSize[0] <= static_cast<unsigned int>(params.nx),
	           "Kernel size in X is larger than the image dimensions");
	ASSERT_MSG(kerSize[1] <= static_cast<unsigned int>(params.ny),
	           "Kernel size in Y is larger than the image dimensions");
	ASSERT_MSG(kerSize[2] <= static_cast<unsigned int>(params.nz),
	           "Kernel size in Z is larger than the image dimensions");

	const GPULaunchParams3D launchParams = inputImage.getLaunchParams();

	if (stream != nullptr)
	{
		// Convolve along X-axis
		convolve3DSeparable_kernel<0>
		    <<<launchParams.gridSize, launchParams.blockSize, 0, *stream>>>(
		        pd_inputImage, pd_intermediaryImage, pd_kernelX, kerSize[0],
		        params.nx, params.ny, params.nz);

		// Convolve along Y-axis
		convolve3DSeparable_kernel<1>
		    <<<launchParams.gridSize, launchParams.blockSize, 0, *stream>>>(
		        pd_intermediaryImage, pd_outputImage, pd_kernelY, kerSize[1],
		        params.nx, params.ny, params.nz);

		// Convolve along Z-axis
		convolve3DSeparable_kernel<2>
		    <<<launchParams.gridSize, launchParams.blockSize, 0, *stream>>>(
		        pd_outputImage, pd_intermediaryImage, pd_kernelZ, kerSize[2],
		        params.nx, params.ny, params.nz);

		outputImage.copyFromDeviceImage(mpd_intermediaryImage.get(), false);

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
		        pd_inputImage, pd_intermediaryImage, pd_kernelX, kerSize[0],
		        params.nx, params.ny, params.nz);

		// Convolve along Y-axis
		convolve3DSeparable_kernel<1>
		    <<<launchParams.gridSize, launchParams.blockSize, 0>>>(
		        pd_intermediaryImage, pd_outputImage, pd_kernelY, kerSize[1],
		        params.nx, params.ny, params.nz);

		// Convolve along Z-axis
		convolve3DSeparable_kernel<2>
		    <<<launchParams.gridSize, launchParams.blockSize, 0>>>(
		        pd_outputImage, pd_intermediaryImage, pd_kernelZ, kerSize[2],
		        params.nx, params.ny, params.nz);

		outputImage.copyFromDeviceImage(mpd_intermediaryImage.get(), false);

		if (synchronize)
		{
			cudaDeviceSynchronize();
		}
	}

	cudaCheckError();
}
}