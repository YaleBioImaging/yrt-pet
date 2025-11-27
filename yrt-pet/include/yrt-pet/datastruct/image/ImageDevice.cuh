/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageBase.hpp"
#include "yrt-pet/utils/GPUTypes.cuh"

#include <cuda_runtime_api.h>

namespace yrt
{
class Image;

class ImageDevice : public ImageBase
{
public:
	virtual float* getDevicePointer() = 0;
	virtual const float* getDevicePointer() const = 0;
	bool isMemoryValid() const;
	size_t getImageSize() const;
	const cudaStream_t* getStream() const;
	void transferToDeviceMemory(const float* ph_img_ptr,
	                            bool p_synchronize = true);
	void transferToDeviceMemory(const Image* ph_img_ptr,
	                            bool p_synchronize = true);
	void transferToHostMemory(float* ph_img_ptr,
	                          bool p_synchronize = true) const;
	void transferToHostMemory(Image* ph_img_ptr,
	                          bool p_synchronize = true) const;
	GPULaunchParams3D getLaunchParams() const;
	void setValue(float initValue) override;
	void copyFromImage(const ImageBase* imSrc) override;
	void addFirstImageToSecond(ImageBase* imgOut) const override;
	void applyThreshold(const ImageBase* maskImg, float threshold,
	                    float val_le_scale, float val_le_off,
	                    float val_gt_scale, float val_gt_off) override;
	void applyThresholdBroadcast(const ImageBase* maskImg, float threshold,
	                             float val_le_scale, float val_le_off,
	                             float val_gt_scale, float val_gt_off) override;
	void updateEMThreshold(ImageBase* updateImg, const ImageBase* normImg,
	                       float threshold) override;
	void updateEMThresholdRankScaled(ImageBase* updateImg, const ImageBase* normImg,
	                                 const float* c_r,
	                                 float threshold) override;
	void writeToFile(const std::string& image_fname) const override;

	void copyFromHostImage(const Image* imSrc, bool synchronize);
	void copyFromDeviceImage(const ImageDevice* imSrc, bool p_synchronize);

	void setValueDevice(float initValue, bool synchronize);
	void updateEMThresholdDevice(ImageDevice* updateImg,
	                             const ImageDevice* normImg, float threshold,
	                             bool synchronize);
	void addFirstImageToSecondDevice(ImageDevice* imgOut,
	                                 bool synchronize) const;
	void applyThresholdDevice(const ImageDevice* maskImg, float threshold,
	                          float val_le_scale, float val_le_off,
	                          float val_gt_scale, float val_gt_off,
	                          bool synchronize);

protected:
	explicit ImageDevice(const cudaStream_t* stream_ptr = nullptr);
	explicit ImageDevice(const ImageParams& imgParams,
	                     const cudaStream_t* stream_ptr = nullptr);
	void setDeviceParams(const ImageParams& params);

	size_t m_imgSize;
	const cudaStream_t* mp_stream;

private:
	GPULaunchParams3D m_launchParams;
};

class ImageDeviceOwned : public ImageDevice
{
public:
	explicit ImageDeviceOwned(const ImageParams& imgParams,
	                          const cudaStream_t* stream_ptr = nullptr);
	explicit ImageDeviceOwned(const Image* img_ptr,
	                          const cudaStream_t* stream_ptr = nullptr);
	explicit ImageDeviceOwned(const std::string& filename,
	                          const cudaStream_t* stream_ptr = nullptr);
	ImageDeviceOwned(const ImageParams& imgParams, const std::string& filename,
	                 const cudaStream_t* stream_ptr = nullptr);
	~ImageDeviceOwned() override;
	void allocate(bool synchronize = true, bool initializeToZero = true);
	void readFromFile(const ImageParams& params, const std::string& filename);
	void readFromFile(const std::string& filename);
	float* getDevicePointer() override;
	const float* getDevicePointer() const override;

private:
	float* mpd_devicePointer;  // Device data
};

class ImageDeviceAlias : public ImageDevice
{
public:
	ImageDeviceAlias(const ImageParams& imgParams,
	                 const cudaStream_t* stream_ptr = nullptr);
	float* getDevicePointer() override;
	const float* getDevicePointer() const override;
	size_t getDevicePointerInULL() const;

	void setDevicePointer(float* ppd_devicePointer);
	void setDevicePointer(size_t ppd_pointerInULL);
	bool isDevicePointerSet() const;

private:
	float* mpd_devicePointer;
};
}  // namespace yrt
