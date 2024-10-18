/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/OperatorDevice.cuh"
#include "operators/OperatorPsf.hpp"
#include "utils/DeviceArray.cuh"

class OperatorPsfDevice : public OperatorDevice, public OperatorPsf
{
public:
	OperatorPsfDevice();
	explicit OperatorPsfDevice(const std::string& imageSpacePsf_fname,
	                           const cudaStream_t* pp_stream = nullptr);

	void readFromFile(const std::string& imageSpacePsf_fname) override;
	void readFromFile(const std::string& imageSpacePsf_fname,
	                  const cudaStream_t* pp_stream = nullptr);

	void copyToDevice(const cudaStream_t* pp_stream = nullptr);

	template <bool Transpose>
	void apply(const Variable* in, Variable* out) const;
	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	void convolve(const ImageDevice& inputImageDevice,
	              ImageDevice& outputImageDevice,
	              const std::vector<float>& kernelX,
	              const std::vector<float>& kernelY,
	              const std::vector<float>& kernelZ) const;

	void convolve(const Image* in, Image* out,
	              const std::vector<float>& kernelX,
	              const std::vector<float>& kernelY,
	              const std::vector<float>& kernelZ) const override;

protected:
	void initDeviceArraysIfNeeded();
	void allocateDeviceArrays(const cudaStream_t* stream = nullptr,
	                          bool synchronize = true);

	template <bool Flipped>
	void convolveDevice(const ImageDevice& inputImage,
	                    ImageDevice& outputImage) const;

	static void convolveDevice(const ImageDevice& inputImage,
	                           ImageDevice& outputImage,
	                           const DeviceArray<float>& kernelX,
	                           const DeviceArray<float>& kernelY,
	                           const DeviceArray<float>& kernelZ,
	                           const cudaStream_t* stream = nullptr,
	                           bool synchronize = true);

	std::unique_ptr<DeviceArray<float>> mpd_kernelX;
	std::unique_ptr<DeviceArray<float>> mpd_kernelY;
	std::unique_ptr<DeviceArray<float>> mpd_kernelZ;
	std::unique_ptr<DeviceArray<float>> mpd_kernelX_flipped;
	std::unique_ptr<DeviceArray<float>> mpd_kernelY_flipped;
	std::unique_ptr<DeviceArray<float>> mpd_kernelZ_flipped;

private:
	void readFromFileInternal(const std::string& imageSpacePsf_fname,
	                          const cudaStream_t* pp_stream);
	static void initDeviceArrayIfNeeded(
	    std::unique_ptr<DeviceArray<float>>& ppd_kernel);
	static void allocateDeviceArray(DeviceArray<float>& prd_kernel,
	                                size_t newSize,
	                                const cudaStream_t* stream = nullptr,
	                                bool synchronize = true);
};
