/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/image/ImageDevice.cuh"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageSpaceKernels.cuh"
#include "yrt-pet/operators/DeviceSynchronized.cuh"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/GPUMemory.cuh"
#include "yrt-pet/utils/GPUTypes.cuh"
#include "yrt-pet/utils/GPUUtils.cuh"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;
namespace yrt
{
void py_setup_imagedevice(py::module& m)
{
	auto c = py::class_<ImageDevice, ImageBase>(m, "ImageDevice");

	c.def(
	    "copyFromNumpy",
	    [](ImageDevice& self, py::buffer& np_data)
	    {
		    try
		    {
			    const py::buffer_info buffer = np_data.request();
			    if (buffer.ndim != 4 && buffer.ndim != 3)
			    {
				    throw std::invalid_argument(
				        "The buffer given has to have 3 or 4 dimensions");
			    }
			    if (buffer.format != py::format_descriptor<float>::format())
			    {
				    throw std::invalid_argument(
				        "The buffer given has to have a float32 format");
			    }
			    std::vector<int> dims = {
			        self.getParams().nt, self.getParams().nz,
			        self.getParams().ny, self.getParams().nx};
			    for (int i = 0; i < buffer.ndim; i++)
			    {
				    if (buffer.shape[i] != dims[buffer.ndim == 4 ? i : i + 1])
				    {
					    throw std::invalid_argument(
					        "The buffer shape does not match with the image "
					        "parameters");
				    }
			    }

			    self.transferToDeviceMemory(
			        reinterpret_cast<float*>(buffer.ptr), true);
		    }
		    catch (const std::exception& e)
		    {
			    std::cerr << "Error in given buffer: " << e.what() << std::endl;
		    }
	    },
	    "Copy from a given numpy array to device", "numpy_array"_a);
	c.def(
	    "transferToHostMemory",
	    [](const ImageDevice& self)
	    {
		    const auto& params = self.getParams();
		    auto hostImage = std::make_unique<ImageOwned>(params);
		    hostImage->allocate();
		    self.transferToHostMemory(hostImage.get(), true);
		    return hostImage;
	    },
	    "Copy image to a new host-side image and return it");
	c.def(
	    "transferToHostMemory",
	    [](ImageDevice& self, Image* img) { self.transferToHostMemory(img); },
	    "Copy image to a given host-side Image", "out"_a);
	c.def(
	    "copyFromDeviceImage", [](ImageDevice* self, const ImageDevice* other)
	    { self->copyFromDeviceImage(other, true); },
	    "Copy from a given device image", "src_device_image"_a);
	c.def(
	    "copyFromHostImage", [](ImageDevice* self, const Image* other)
	    { self->copyFromHostImage(other, true); },
	    "Copy from a given host image", "src_host_image"_a);

	c.def("addFirstImageToSecond", &ImageDevice::addFirstImageToSecond,
	      "img_out"_a);
	c.def("applyThreshold", &ImageDevice::applyThreshold, "mask_image"_a,
	      "threshold"_a, "val_le_scale"_a, "val_le_off"_a, "val_gt_scale"_a,
	      "val_gt_off"_a);
	c.def("writeToFile", &ImageDevice::writeToFile, "image_fname"_a);
	c.def("getNumFrames", &ImageDevice::getNumFrames);

	c.def(
	    "applyThresholdDevice",
	    [](ImageDevice& self, const ImageDevice* maskImg, const float threshold,
	       const float val_le_scale, const float val_le_off,
	       const float val_gt_scale, const float val_gt_off)
	    {
		    self.applyThresholdDevice(maskImg, threshold, val_le_scale,
		                              val_le_off, val_gt_scale, val_gt_off,
		                              true);
	    },
	    "mask_image"_a, "threshold"_a, "val_le_scale"_a, "val_le_off"_a,
	    "val_gt_scale"_a, "val_gt_off"_a);

	auto c_owned =
	    py::class_<ImageDeviceOwned, ImageDevice>(m, "ImageDeviceOwned");
	c_owned.def(
	    py::init(
	        [](const ImageParams& imgParams)
	        { return std::make_unique<ImageDeviceOwned>(imgParams, nullptr); }),
	    "Create ImageDevice using image parameters (will not allocate)",
	    "img_params"_a);
	c_owned.def(
	    py::init(
	        [](const std::string& filename)
	        { return std::make_unique<ImageDeviceOwned>(filename, nullptr); }),
	    "Create ImageDevice using filename (will allocate)", "filename"_a);
	c_owned.def(
	    py::init(
	        [](const ImageParams& imgParams, const std::string& filename)
	        {
		        return std::make_unique<ImageDeviceOwned>(imgParams, filename,
		                                                  nullptr);
	        }),
	    "Create ImageDevice using image parameters and filename (will "
	    "allocate)",
	    "img_params"_a, "filename"_a);
	c_owned.def(
	    py::init(
	        [](const Image* img_ptr)
	        { return std::make_unique<ImageDeviceOwned>(img_ptr, nullptr); }),
	    "Create a ImageDevice using a host-side Image (will allocate)",
	    "img"_a);
	c_owned.def("allocate",
	            [](ImageDeviceOwned& self) { self.allocate(true); });

	auto c_alias =
	    py::class_<ImageDeviceAlias, ImageDevice>(m, "ImageDeviceAlias");
	c_alias.def(
	    py::init(
	        [](const ImageParams& imgParams)
	        { return std::make_unique<ImageDeviceAlias>(imgParams, nullptr); }),
	    "Create ImageDevice using image parameters (will not allocate)",
	    "img_params"_a);
	c_alias.def("getDevicePointer", &ImageDeviceAlias::getDevicePointerInULL);
	c_alias.def("setDevicePointer",
	            static_cast<void (ImageDeviceAlias::*)(size_t)>(
	                &ImageDeviceAlias::setDevicePointer),
	            "Set a device address for the image array. For "
	            "usage with PyTorch, use \'myArray.data_ptr()\'",
	            "data_ptr"_a);
	c_alias.def("isDevicePointerSet", &ImageDeviceAlias::isDevicePointerSet,
	            "Returns true if the device pointer is not null");
}
}  // namespace yrt

#endif  // if BUILD_PYBIND11

namespace yrt
{
ImageDevice::ImageDevice(const cudaStream_t* stream_ptr)
    : ImageBase{}, mp_stream(stream_ptr)
{
}

ImageDevice::ImageDevice(const ImageParams& imgParams,
                         const cudaStream_t* stream_ptr)
    : ImageBase(imgParams), mp_stream(stream_ptr)
{
	setDeviceParams(imgParams);
}

void ImageDevice::setDeviceParams(const ImageParams& params)
{
	m_launchParams = util::initiateDeviceParameters(params);
	m_imgSize = params.nt * params.nx * params.ny * params.nz;
}

const cudaStream_t* ImageDevice::getStream() const
{
	return mp_stream;
}

bool ImageDevice::isMemoryValid() const
{
	return getDevicePointer() != nullptr;
}

size_t ImageDevice::getImageSize() const
{
	return m_imgSize;
}

void ImageDevice::transferToDeviceMemory(const float* ph_img_ptr,
                                         bool p_synchronize)
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");
	util::copyHostToDevice(getDevicePointer(), ph_img_ptr, m_imgSize,
	                       {mp_stream, p_synchronize});
}

void ImageDevice::transferToDeviceMemory(const Image* ph_img_ptr,
                                         bool p_synchronize)
{
	const auto& dstParams = getParams();
	const auto& srcParams = ph_img_ptr->getParams();

	ASSERT_MSG(dstParams.isSameDimensionsAs(srcParams),
	           "Image dimensions mismatch");

	const int dstT = dstParams.nt;
	const int srcT = srcParams.nt;

	const float* ph_ptr = ph_img_ptr->getRawPointer();

	if (dstT == srcT)
	{
		transferToDeviceMemory(ph_ptr, p_synchronize);
		return;
	}
	if (srcT == 1 && dstT > 1)
	{
		float* d_base = getDevicePointer();  // raw device pointer
		ASSERT_MSG(d_base != nullptr, "Device Image not allocated yet");

		const size_t spatialSize = dstParams.nx * dstParams.ny * dstParams.nz;
		ASSERT_MSG(
		    getImageSize() == spatialSize * static_cast<size_t>(dstT),
		    "ImageDevice::transferToDeviceMemory: inconsistent device size");

		// Copy and broadcast
		for (int t = 0; t < dstT; ++t)
		{
			float* d_frame = d_base + t * spatialSize;
			util::copyHostToDevice(d_frame, ph_ptr, spatialSize,
			                       {mp_stream, p_synchronize});
		}
		return;
	}
	ASSERT_MSG(false,
	           "ImageDevice::transferToDeviceMemory(Image*): "
	           "unsupported combination of time dimensions (host vs device)");
}

void ImageDevice::transferToHostMemory(float* ph_img_ptr,
                                       bool p_synchronize) const
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated");
	ASSERT_MSG(ph_img_ptr != nullptr, "Host image not allocated");
	util::copyDeviceToHost(ph_img_ptr, getDevicePointer(), m_imgSize,
	                       {mp_stream, p_synchronize});
}

void ImageDevice::transferToHostMemory(Image* ph_img_ptr,
                                       bool p_synchronize) const
{
	ASSERT_MSG(ph_img_ptr->getParams().isSameDimensionsAs(getParams()),
	           "Dimensions mismatch between host and device image.");

	// We allow to transfer a device image to a host image that has more frames
	//  than the device image. It would only transfer the first frames.
	ASSERT_MSG(ph_img_ptr->getParams().nt >= getParams().nt,
	           "The number of frames in the host image is smaller than the "
	           "device image");

	float* ph_ptr = ph_img_ptr->getRawPointer();

	std::cout << "Transferring image from Device to Host..." << std::endl;
	transferToHostMemory(ph_ptr, p_synchronize);
}

int ImageDevice::getNumFrames() const
{
	return getParams().nt;
}

GPULaunchParams3D ImageDevice::getLaunchParams() const
{
	return m_launchParams;
}

void ImageDevice::applyThresholdDevice(const ImageDevice* maskImg,
                                       const float threshold,
                                       const float val_le_scale,
                                       const float val_le_off,
                                       const float val_gt_scale,
                                       const float val_gt_off, bool synchronize)
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");
	auto maskParams = maskImg->getParams();
	if (mp_stream != nullptr)
	{
		applyThreshold_kernel<<<m_launchParams.gridSize,
		                        m_launchParams.blockSize, 0, *mp_stream>>>(
		    getDevicePointer(), maskImg->getDevicePointer(), threshold,
		    val_le_scale, val_le_off, val_gt_scale, val_gt_off, getParams().nx,
		    getParams().ny, getParams().nz);
		if (synchronize)
		{
			cudaStreamSynchronize(*mp_stream);
		}
	}
	else
	{
		applyThreshold_kernel<<<m_launchParams.gridSize,
		                        m_launchParams.blockSize>>>(
		    getDevicePointer(), maskImg->getDevicePointer(), threshold,
		    val_le_scale, val_le_off, val_gt_scale, val_gt_off, getParams().nx,
		    getParams().ny, getParams().nz);
		if (synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}


void ImageDevice::applyThresholdBroadcastDevice(
    const ImageDevice* maskImg, const float threshold, const float val_le_scale,
    const float val_le_off, const float val_gt_scale, const float val_gt_off,
    bool synchronize)
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");
	if (mp_stream != nullptr)
	{
		applyThresholdBroadcast_kernel<<<
		    m_launchParams.gridSize, m_launchParams.blockSize, 0, *mp_stream>>>(
		    getDevicePointer(), maskImg->getDevicePointer(), threshold,
		    val_le_scale, val_le_off, val_gt_scale, val_gt_off, getParams().nx,
		    getParams().ny, getParams().nz, getParams().nt);
		if (synchronize)
		{
			cudaStreamSynchronize(*mp_stream);
		}
	}
	else
	{
		applyThresholdBroadcast_kernel<<<m_launchParams.gridSize,
		                                 m_launchParams.blockSize>>>(
		    getDevicePointer(), maskImg->getDevicePointer(), threshold,
		    val_le_scale, val_le_off, val_gt_scale, val_gt_off, getParams().nx,
		    getParams().ny, getParams().nz, getParams().nt);
		if (synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}


void ImageDevice::applyThresholdBroadcast(const ImageBase* maskImg,
                                          float threshold, float val_le_scale,
                                          float val_le_off, float val_gt_scale,
                                          float val_gt_off)
{
	const auto maskImg_ImageDevice = dynamic_cast<const ImageDevice*>(maskImg);
	ASSERT_MSG(maskImg_ImageDevice != nullptr,
	           "Input image has the wrong type");
	applyThresholdBroadcastDevice(maskImg_ImageDevice, threshold, val_le_scale,
	                              val_le_off, val_gt_scale, val_gt_off, true);
}

void ImageDevice::updateEMThresholdDynamicDevice(ImageDevice* updateImg,
                                                 const ImageDevice* sensImg,
                                                 float threshold,
                                                 bool synchronize)
{
	if (getNumFrames() == 1)
	{
		// Use the static version
		updateEMThresholdStaticDevice(updateImg, sensImg, threshold,
		                              synchronize);
	}
	else
	{
		// Use the dynamic version
		if (sensImg->getNumFrames() == 1)
		{
			// Update only using the first frame of the sensitivity image
			updateEMThresholdDynamicWithScaling(updateImg, sensImg, nullptr,
			                                    threshold, synchronize);
		}
		else
		{
			updateEMThresholdDynamicWith4DSens(updateImg, sensImg, threshold,
			                                   synchronize);
		}
	}
}

void ImageDevice::updateEMThresholdDynamicWithScaling(
    ImageDevice* updateImg, const ImageDevice* sensImg,
    const float* pd_sensScaling, float threshold, bool synchronize)
{
	ASSERT_MSG(updateImg->getParams().isSameDimensionsAs(getParams()),
	           "Image dimensions mismatch");
	ASSERT_MSG(sensImg->getParams().isSameDimensionsAs(getParams()),
	           "Image dimensions mismatch");
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		if (pd_sensScaling == nullptr)
		{
			updateEMDynamic_kernel<false>
			    <<<m_launchParams.gridSize, m_launchParams.blockSize, 0,
			       *mp_stream>>>(
			        updateImg->getDevicePointer(), getDevicePointer(),
			        sensImg->getDevicePointer(), getParams().nx, getParams().ny,
			        getParams().nz, getParams().nt, nullptr, threshold);
		}
		else
		{
			updateEMDynamic_kernel<true>
			    <<<m_launchParams.gridSize, m_launchParams.blockSize, 0,
			       *mp_stream>>>(
			        updateImg->getDevicePointer(), getDevicePointer(),
			        sensImg->getDevicePointer(), getParams().nx, getParams().ny,
			        getParams().nz, getParams().nt, pd_sensScaling, threshold);
		}
	}
	else
	{
		if (pd_sensScaling == nullptr)
		{
			updateEMDynamic_kernel<false>
			    <<<m_launchParams.gridSize, m_launchParams.blockSize>>>(
			        updateImg->getDevicePointer(), getDevicePointer(),
			        sensImg->getDevicePointer(), getParams().nx, getParams().ny,
			        getParams().nz, getParams().nt, nullptr, threshold);
		}
		else
		{
			updateEMDynamic_kernel<true>
			    <<<m_launchParams.gridSize, m_launchParams.blockSize>>>(
			        updateImg->getDevicePointer(), getDevicePointer(),
			        sensImg->getDevicePointer(), getParams().nx, getParams().ny,
			        getParams().nz, getParams().nt, pd_sensScaling, threshold);
		}
	}
	synchronizeIfNeeded({mp_stream, synchronize});
	ASSERT(cudaCheckError());
}

void ImageDevice::updateEMThresholdDynamicWith4DSens(ImageDevice* updateImg,
                                                     const ImageDevice* sensImg,
                                                     float threshold,
                                                     bool synchronize)
{
	ASSERT_MSG(updateImg->getParams().isSameDimensionsAs(getParams()),
	           "Image dimensions mismatch");
	ASSERT_MSG(sensImg->getParams().isSameDimensionsAs(getParams()),
	           "Image dimensions mismatch");
	ASSERT_MSG(updateImg->getParams().isSameNumFramesAs(getParams()),
	           "Size mismatch with update image in time dimension");
	ASSERT_MSG(sensImg->getParams().isSameNumFramesAs(getParams()),
	           "Size mismatch with sensitivity image in time dimension");
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		updateEM_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize, 0,
		                  *mp_stream>>>(
		    updateImg->getDevicePointer(), getDevicePointer(),
		    sensImg->getDevicePointer(), getParams().nx, getParams().ny,
		    getParams().nz, getParams().nt, threshold);
	}
	else
	{
		updateEM_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize>>>(
		    updateImg->getDevicePointer(), getDevicePointer(),
		    sensImg->getDevicePointer(), getParams().nx, getParams().ny,
		    getParams().nz, getParams().nt, threshold);
	}
	synchronizeIfNeeded({mp_stream, synchronize});
	ASSERT(cudaCheckError());
}

void ImageDevice::updateEMThresholdDynamic(ImageBase* updateImg,
                                           const ImageBase* sensImg,
                                           float threshold)
{
	auto* updateImg_ImageDevice = dynamic_cast<ImageDevice*>(updateImg);
	const auto* sensImg_ImageDevice = dynamic_cast<const ImageDevice*>(sensImg);

	ASSERT_MSG(updateImg_ImageDevice != nullptr,
	           "updateImg is not ImageDevice");
	ASSERT_MSG(sensImg_ImageDevice != nullptr, "sensImg is not ImageDevice");

	updateEMThresholdDynamicDevice(updateImg_ImageDevice, sensImg_ImageDevice,
	                               threshold, true);
}

void ImageDevice::updateEMThresholdDynamic(
    ImageBase* updateImg, const ImageBase* sensImg,
    const std::vector<float>& sensScaling, float threshold)
{
	auto* updateImg_ImageDevice = dynamic_cast<ImageDevice*>(updateImg);
	const auto* sensImg_ImageDevice = dynamic_cast<const ImageDevice*>(sensImg);

	ASSERT_MSG(updateImg_ImageDevice != nullptr,
	           "updateImg is not ImageDevice");
	ASSERT_MSG(sensImg_ImageDevice != nullptr, "sensImg is not ImageDevice");

	// Create temporary device-side array to store the sensScaling
	DeviceArray<float> sensScalingDevice(sensScaling.size());
	sensScalingDevice.copyFromHost(sensScaling.data(), sensScaling.size(), {});

	updateEMThresholdDynamicWithScaling(
	    updateImg_ImageDevice, sensImg_ImageDevice,
	    sensScalingDevice.getDevicePointer(), threshold, true);
}

void ImageDevice::applyThreshold(const ImageBase* maskImg, float threshold,
                                 float val_le_scale, float val_le_off,
                                 float val_gt_scale, float val_gt_off)
{
	const auto maskImg_ImageDevice = dynamic_cast<const ImageDevice*>(maskImg);
	ASSERT_MSG(maskImg_ImageDevice != nullptr,
	           "Input image has the wrong type");
	applyThresholdDevice(maskImg_ImageDevice, threshold, val_le_scale,
	                     val_le_off, val_gt_scale, val_gt_off, true);
}

void ImageDevice::writeToFile(const std::string& image_fname) const
{
	auto tmpImage = std::make_unique<ImageOwned>(getParams());
	tmpImage->allocate();
	transferToHostMemory(tmpImage.get(), true);
	tmpImage->writeToFile(image_fname);
}

void ImageDevice::copyFromHostImage(const Image* imSrc, bool synchronize)
{
	ASSERT(imSrc != nullptr);
	transferToDeviceMemory(imSrc, synchronize);
}

void ImageDevice::copyFromDeviceImage(const ImageDevice* imSrc,
                                      bool p_synchronize)
{
	ASSERT(imSrc != nullptr);
	const float* pd_src = imSrc->getDevicePointer();
	float* pd_dest = getDevicePointer();
	ASSERT(pd_src != nullptr);
	ASSERT(pd_dest != nullptr);
	util::copyDeviceToDevice(pd_dest, pd_src, m_imgSize,
	                         {mp_stream, p_synchronize});
}

void ImageDevice::fillDevice(float initValue, bool synchronize)
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		fill_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize, 0,
		              *mp_stream>>>(getDevicePointer(), initValue,
		                            getParams().nx, getParams().ny,
		                            getParams().nz, getParams().nt);
	}
	else
	{
		fill_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize>>>(
		    getDevicePointer(), initValue, getParams().nx, getParams().ny,
		    getParams().nz, getParams().nt);
	}
	synchronizeIfNeeded({mp_stream, synchronize});
	ASSERT(cudaCheckError());
}

void ImageDevice::updateEMThresholdStaticDevice(ImageDevice* updateImg,
                                                const ImageDevice* sensImg,
                                                float threshold,
                                                bool synchronize)
{
	ASSERT_MSG(updateImg->getParams().isSameDimensionsAs(getParams()),
	           "Image dimensions mismatch");
	ASSERT_MSG(sensImg->getParams().isSameDimensionsAs(getParams()),
	           "Image dimensions mismatch");
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		updateEM_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize, 0,
		                  *mp_stream>>>(
		    updateImg->getDevicePointer(), getDevicePointer(),
		    sensImg->getDevicePointer(), getParams().nx, getParams().ny,
		    getParams().nz, getParams().nt, threshold);
	}
	else
	{
		updateEM_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize>>>(
		    updateImg->getDevicePointer(), getDevicePointer(),
		    sensImg->getDevicePointer(), getParams().nx, getParams().ny,
		    getParams().nz, getParams().nt, threshold);
	}
	synchronizeIfNeeded({mp_stream, synchronize});
	ASSERT(cudaCheckError());
}

void ImageDevice::addFirstImageToSecondDevice(ImageDevice* imgOut,
                                              bool synchronize) const
{
	ASSERT_MSG(imgOut->getParams().isSameDimensionsAs(getParams()),
	           "Image dimensions mismatch");
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (imgOut->getParams().nt > getParams().nt)
	{
		if (mp_stream != nullptr)
		{
			addFirstImage3DToSecond4D_kernel<<<m_launchParams.gridSize,
			                                   m_launchParams.blockSize, 0,
			                                   *mp_stream>>>(
			    getDevicePointer(), imgOut->getDevicePointer(), getParams().nx,
			    getParams().ny, getParams().nz, imgOut->getParams().nt);
		}
		else
		{
			addFirstImage3DToSecond4D_kernel<<<m_launchParams.gridSize,
			                                   m_launchParams.blockSize>>>(
			    getDevicePointer(), imgOut->getDevicePointer(), getParams().nx,
			    getParams().ny, getParams().nz, imgOut->getParams().nt);
		}
	}
	else
	{
		if (mp_stream != nullptr)
		{
			addFirstImageToSecond_kernel<<<m_launchParams.gridSize,
			                               m_launchParams.blockSize, 0,
			                               *mp_stream>>>(
			    getDevicePointer(), imgOut->getDevicePointer(), getParams().nx,
			    getParams().ny, getParams().nz);
		}
		else
		{
			addFirstImageToSecond_kernel<<<m_launchParams.gridSize,
			                               m_launchParams.blockSize>>>(
			    getDevicePointer(), imgOut->getDevicePointer(), getParams().nx,
			    getParams().ny, getParams().nz);
		}
	}
	synchronizeIfNeeded({mp_stream, synchronize});
	ASSERT(cudaCheckError());
}

void ImageDevice::updateEMThresholdStatic(ImageBase* updateImg,
                                          const ImageBase* sensImg,
                                          float threshold)
{
	auto* updateImg_ImageDevice = dynamic_cast<ImageDevice*>(updateImg);
	const auto* sensImg_ImageDevice = dynamic_cast<const ImageDevice*>(sensImg);

	ASSERT_MSG(updateImg_ImageDevice != nullptr,
	           "updateImg is not ImageDevice");
	ASSERT_MSG(sensImg_ImageDevice != nullptr, "sensImg is not ImageDevice");

	updateEMThresholdStaticDevice(updateImg_ImageDevice, sensImg_ImageDevice,
	                              threshold, true);
}

void ImageDevice::addFirstImageToSecond(ImageBase* second) const
{
	auto* second_ImageDevice = dynamic_cast<ImageDevice*>(second);

	ASSERT_MSG(second_ImageDevice != nullptr, "'second' is not an ImageDevice");

	addFirstImageToSecondDevice(second_ImageDevice, true);
}

void ImageDevice::fill(float initValue)
{
	fillDevice(initValue, true);
}

void ImageDevice::multWithScalar(float scalar, bool synchronize)
{
	float* devicePointer = getDevicePointer();
	ASSERT_MSG(devicePointer != nullptr, "Device Image not allocated yet");

	const ImageParams& params = getParams();

	if (mp_stream != nullptr)
	{
		multWithScalar_kernel<<<m_launchParams.gridSize,
		                        m_launchParams.blockSize, 0, *mp_stream>>>(
		    devicePointer, scalar, params.nx, params.ny, params.nz, params.nt);
	}
	else
	{
		multWithScalar_kernel<<<m_launchParams.gridSize,
		                        m_launchParams.blockSize>>>(
		    devicePointer, scalar, params.nx, params.ny, params.nz, params.nt);
	}

	synchronizeIfNeeded({mp_stream, synchronize});
}

void ImageDevice::copyFromImage(const ImageBase* imSrc)
{
	const auto imSrc_host = dynamic_cast<const Image*>(imSrc);
	if (imSrc_host != nullptr)
	{
		// Input image is in host
		copyFromHostImage(imSrc_host, true);
	}
	else
	{
		const auto imSrc_dev = dynamic_cast<const ImageDevice*>(imSrc);
		copyFromDeviceImage(imSrc_dev, true);
	}
}


ImageDeviceOwned::ImageDeviceOwned(const ImageParams& imgParams,
                                   const cudaStream_t* stream_ptr)
    : ImageDevice(imgParams, stream_ptr), mpd_devicePointer(nullptr)
{
}

ImageDeviceOwned::ImageDeviceOwned(const std::string& filename,
                                   const cudaStream_t* stream_ptr)
    : ImageDevice(stream_ptr), mpd_devicePointer(nullptr)
{
	readFromFile(filename);
}

ImageDeviceOwned::ImageDeviceOwned(const ImageParams& imgParams,
                                   const std::string& filename,
                                   const cudaStream_t* stream_ptr)
    : ImageDevice(imgParams, stream_ptr), mpd_devicePointer(nullptr)
{
	readFromFile(getParams(), filename);
}

ImageDeviceOwned::ImageDeviceOwned(const Image* img_ptr,
                                   const cudaStream_t* stream_ptr)
    : ImageDevice(img_ptr->getParams(), stream_ptr), mpd_devicePointer(nullptr)
{
	allocate(false);
	transferToDeviceMemory(img_ptr, true);
}

ImageDeviceOwned::~ImageDeviceOwned()
{
	if (mpd_devicePointer != nullptr)
	{
		std::cout << "Freeing image device buffer..." << std::endl;
		util::deallocateDevice(mpd_devicePointer, {nullptr, true});
	}
}

void ImageDeviceOwned::allocate(bool synchronize, bool initializeToZero)
{
	const auto& params = getParams();
	std::cout << "Allocating device memory for an image of dimensions "
	          << "[" << params.nt << ", " << params.nz << ", " << params.ny
	          << ", " << params.nx << "]..." << std::endl;

	util::allocateDevice(&mpd_devicePointer, m_imgSize, {mp_stream, false});
	if (initializeToZero)
	{
		util::memsetDevice(mpd_devicePointer, 0, m_imgSize,
		                   {mp_stream, synchronize});
	}
}

void ImageDeviceOwned::readFromFile(const ImageParams& params,
                                    const std::string& filename)
{
	// Create temporary Image
	const auto img = std::make_unique<ImageOwned>(params, filename);
	allocate(false);
	transferToDeviceMemory(img.get(), true);
}

void ImageDeviceOwned::readFromFile(const std::string& filename)
{
	// Create temporary Image
	const auto img = std::make_unique<ImageOwned>(filename);
	setParams(img->getParams());
	setDeviceParams(getParams());
	allocate(false);
	transferToDeviceMemory(img.get(), true);
}

float* ImageDeviceOwned::getDevicePointer()
{
	return mpd_devicePointer;
}

const float* ImageDeviceOwned::getDevicePointer() const
{
	return mpd_devicePointer;
}


ImageDeviceAlias::ImageDeviceAlias(const ImageParams& imgParams,
                                   const cudaStream_t* stream_ptr)
    : ImageDevice(imgParams, stream_ptr), mpd_devicePointer(nullptr)
{
}

float* ImageDeviceAlias::getDevicePointer()
{
	return mpd_devicePointer;
}

const float* ImageDeviceAlias::getDevicePointer() const
{
	return mpd_devicePointer;
}

size_t ImageDeviceAlias::getDevicePointerInULL() const
{
	return reinterpret_cast<size_t>(mpd_devicePointer);
}

void ImageDeviceAlias::setDevicePointer(float* ppd_devicePointer)
{
	mpd_devicePointer = ppd_devicePointer;
}

void ImageDeviceAlias::setDevicePointer(size_t ppd_pointerInULL)
{
	setDevicePointer(reinterpret_cast<float*>(ppd_pointerInULL));
}

bool ImageDeviceAlias::isDevicePointerSet() const
{
	return mpd_devicePointer != nullptr;
}
}  // namespace yrt
