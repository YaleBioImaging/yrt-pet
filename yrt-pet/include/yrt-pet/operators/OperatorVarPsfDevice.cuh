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
	struct DeviceVarPsf
	{
		float* kernels;     // flatten kernel data
		int*   offsets;     // The starting position of each kernel in the kernels array
		int*   sizes;       // The size of each kernel (number of elements)
		int    numKernels;  // kernel number
	};

	explicit OperatorVarPsfDevice(const cudaStream_t* pp_stream = nullptr, const ImageParams& p_imageParams);
	explicit OperatorVarPsfDevice(const std::string& pr_imagePsf_fname,
							   const cudaStream_t* pp_stream = nullptr, const ImageParams& p_imageParams);

	void copyVarPsfToDevice(bool synchronize = true);// copy kernel LUT to device: allocate and upload
	//void readFromFile(const std::string& pr_imagePsf_fname) override;
	//void readFromFile(const std::string& pr_imagePsf_fname, bool p_synchronize);

	void allocateTemporaryDeviceImageIfNeeded(const ImageParams& params,
	                                          GPULaunchConfig config) const;

	//void applyA(const Variable* in, Variable* out) override;
	//void applyAH(const Variable* in, Variable* out) override;
	//void applyA(const Variable* in, Variable* out, bool synchronize) const;
	//void applyAH(const Variable* in, Variable* out, bool synchronize) const;

protected:
	void initDeviceArraysIfNeeded();
	void allocateDeviceArrays(bool synchronize);
	template <bool Transpose>
	void apply(const Variable* in, Variable* out, bool synchronize) const;

	mutable std::unique_ptr<ImageDeviceOwned> mpd_intermediaryImage;

private:
	// flattened GPU storage
	std::unique_ptr<DeviceArray<float>> mpd_kernelsFlat;
	std::unique_ptr<DeviceArray<int>> mpd_kernelOffsets;
	std::unique_ptr<DeviceArray<int>> mpd_kernelDims;       // triples: kx,ky,kz per kernel
	std::unique_ptr<DeviceArray<int>> mpd_kernelHalfSizes; // triples: hx,hy,hz per kernel
	std::unique_ptr<DeviceArray<int>> mpd_kernelLUT;

	static void initDeviceArrayIfNeeded(
		std::unique_ptr<DeviceArray<float>>& ppd_kernel);

	//std::vector<std::unique_ptr<DeviceArray<float>>> mpd_kernelLUT;
	void readFromFileInternal(const std::string& pr_imagePsf_fname,
	                          bool p_synchronize);
	//static void initDeviceArrayIfNeeded(
	 //   std::unique_ptr<DeviceArray<float>>& ppd_kernel);
	void flattenKernelsToDevice(bool synchronize);
	void allocateDeviceArraysVarPsf(size_t nKernels, size_t totalKernelSize, bool synchronize);

	// scalar parameters copied to kernels as args
	int lut_x_dim = 0, lut_y_dim = 0, lut_z_dim = 0;
	float d_xGap = 0.f, d_yGap = 0.f, d_zGap = 0.f;
	float d_xCenter = 0.f, d_yCenter = 0.f, d_zCenter = 0.f;
};
}  // namespace yrt