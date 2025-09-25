/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "OperatorPsfDevice.cuh"
#include "datastruct/image/ImageDevice.cuh"
#include "operators/DeviceSynchronized.cuh"
#include "operators/OperatorVarPsf.hpp"
#include "utils/DeviceArray.cuh"


namespace yrt
{

class OperatorVarPsfDevice : public DeviceSynchronized, public OperatorPsfDevice, public OperatorVarPsf
{
public:
	explicit OperatorVarPsfDevice(const cudaStream_t* pp_stream = nullptr, const ImageParams& p_imageParams);
	explicit OperatorVarPsfDevice(const std::string& pr_imagePsf_fname,
							   const cudaStream_t* pp_stream = nullptr, const ImageParams& p_imageParams);

	void readFromFile(const std::string& pr_imagePsf_fname) override;
	void readFromFile(const std::string& pr_imagePsf_fname, bool p_synchronize);

	void allocateTemporaryDeviceImageIfNeeded(const ImageParams& params,
	                                          GPULaunchConfig config) const;

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;
	void applyA(const Variable* in, Variable* out, bool synchronize) const;
	void applyAH(const Variable* in, Variable* out, bool synchronize) const;

protected:
	void initDeviceArraysIfNeeded();
	void allocateDeviceArrays(bool synchronize);
	template <bool Transpose>
	void apply(const Variable* in, Variable* out, bool synchronize) const;

	mutable std::unique_ptr<ImageDeviceOwned> mpd_intermediaryImage;

private:
	std::vector<std::unique_ptr<DeviceArray<float>>> mpd_kernelLUT;
	void readFromFileInternal(const std::string& pr_imagePsf_fname,
	                          bool p_synchronize);
	static void initDeviceArrayIfNeeded(
	    std::unique_ptr<DeviceArray<float>>& ppd_kernel);
	void allocateDeviceArray(DeviceArray<float>& prd_kernel, size_t newSize,
	                         bool synchronize);
};
}  // namespace yrt