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

}
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
	// Constructors should be synchronized
	readFromFileInternal(pr_imagePsf_fname, true);
}

void OperatorVarPsfDevice::readFromFileInternal(
    const std::string& pr_imagePsf_fname, bool p_synchronize)
{
	OperatorVarPsf::readFromFile(pr_imagePsf_fname);
	copyToDevice(p_synchronize);
}

void OperatorVarPsfDevice::readFromFile(const std::string& pr_imagePsf_fname)
{
	readFromFileInternal(pr_imagePsf_fname, true);
}

void OperatorVarPsfDevice::readFromFile(const std::string& pr_imagePsf_fname,
                                     bool p_synchronize)
{
	readFromFileInternal(pr_imagePsf_fname, p_synchronize);
}

void OperatorVarPsfDevice::copyToDevice(bool synchronize)
{
	const GPULaunchConfig launchConfig{getAuxStream(), synchronize};

	initDeviceArraysIfNeeded();
	allocateDeviceArrays(synchronize);

	for (size_t i = 0; i < m_kernelLUT.size(); ++i)
	{
		if (mpd_kernelLUT.size() <= i)
		{
			mpd_kernelLUT.emplace_back(std::make_unique<DeviceArray<float>>());
		}

		const auto& kernelArray = m_kernelLUT[i]->getArray();
		if (!mpd_kernelLUT[i]->isAllocated())
		{
			mpd_kernelLUT[i]->allocate(kernelArray.getSizeTotal(), launchConfig);
		}

		mpd_kernelLUT[i]->copyFromHost(kernelArray.getPointer(),
									   kernelArray.getSizeTotal(), launchConfig);
	}

}

void OperatorVarPsfDevice::allocateTemporaryDeviceImageIfNeeded(
    const ImageParams& params, GPULaunchConfig config) const
{
	const auto* stream = config.stream;
	if (mpd_intermediaryImage == nullptr ||
	    !(mpd_intermediaryImage->getParams().isSameDimensionsAs(params)))
	{
		mpd_intermediaryImage =
		    std::make_unique<ImageDeviceOwned>(params, stream);
	}

	if (!mpd_intermediaryImage->isMemoryValid())
	{
		mpd_intermediaryImage->allocate(config.synchronize);
	}
}

template <bool IS_FWD>
void OperatorPsfDevice::varconvolveDevice(const ImageDevice& inputImage,
									   ImageDevice& outputImage, bool synchronize) const
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

	const auto* stream = getMainStream();




	int pp = blockIdx.x * blockDim.x + threadIdx.x;
    if (pp >= nx * ny * nz) return;

    int i = pp % nx;
    int j = (pp / nx) % ny;
    int k = pp / (nx * ny);

    float temp_x = abs((i + 0.5) * vx - x_center);
    float temp_y = abs((j + 0.5) * vy - y_center);
    float temp_z = abs((k + 0.5) * vz - z_center);

    // Finding the nearest kernel (simplified for illustration)
    int best_idx = 0;
    float best_dist = FLT_MAX;
    for (int idx = 0; idx < num_kernels; ++idx) {
        float dist = (temp_x - sigma_lookup[idx].x) * (temp_x - sigma_lookup[idx].x) +
                     (temp_y - sigma_lookup[idx].y) * (temp_y - sigma_lookup[idx].y) +
                     (temp_z - sigma_lookup[idx].z) * (temp_z - sigma_lookup[idx].z);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = idx;
        }
    }
    const Sigma& chosen_sigma = sigma_lookup[best_idx];

    int kernel_size_x = min(4, max(1, static_cast<int>(floor((chosen_sigma.sigmax * chosen_sigma.kernel_width_control) / vx)) - 1));
    int kernel_size_y = min(4, max(1, static_cast<int>(floor((chosen_sigma.sigmay * chosen_sigma.kernel_width_control) / vy)) - 1));
    int kernel_size_z = min(4, max(1, static_cast<int>(floor((chosen_sigma.sigmaz * chosen_sigma.kernel_width_control) / vz)) - 1));

    const float* psf_kernel = chosen_sigma.psf_kernel;

    int idx = 0;
    float temp1 = inPtr[IDX3(i, j, k, nx, ny)];

    for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z; ++z_diff) {
        for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; ++y_diff) {
            for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x; ++x_diff, ++idx) {
                int ii = util::circular(nx, i + x_diff);
                int jj = util::circular(ny, j + y_diff);
                int kk = util::circular(nz, k + z_diff);

                if (IS_FWD) {
                    atomicAdd(&outPtr[IDX3(i, j, k, nx, ny)], inPtr[IDX3(ii, jj, kk, nx, ny)] * psf_kernel[idx]);
                } else {
                    atomicAdd(&outPtr[IDX3(ii, jj, kk, nx, ny)], temp1 * psf_kernel[idx]);
                }
            }
        }
    }
}


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

	convolveDevice<Transpose>(*inputImageDevice_ptr, *outputImageDevice_ptr,
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

void OperatorVarPsfDevice::allocateDeviceArrays(bool synchronize)
{
	allocateDeviceArray(*mpd_kernelX, m_kernelX.size(), synchronize);
	allocateDeviceArray(*mpd_kernelY, m_kernelY.size(), synchronize);
	allocateDeviceArray(*mpd_kernelZ, m_kernelZ.size(), synchronize);
	allocateDeviceArray(*mpd_kernelX_flipped, m_kernelX.size(), synchronize);
	allocateDeviceArray(*mpd_kernelY_flipped, m_kernelY.size(), synchronize);
	allocateDeviceArray(*mpd_kernelZ_flipped, m_kernelZ.size(), synchronize);
}

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
