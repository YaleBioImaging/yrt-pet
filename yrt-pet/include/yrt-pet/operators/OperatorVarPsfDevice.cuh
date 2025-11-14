/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/operators/DeviceSynchronized.cuh"
#include "yrt-pet/operators/OperatorVarPsf.hpp"
#include "yrt-pet/utils/DeviceArray.cuh"


namespace yrt
{

class OperatorVarPsfDevice : public DeviceSynchronized, public OperatorVarPsf
{
public:
	struct DeviceVarPsf
	{
		float* kernels;     // flatten kernel data
		int*   offsets;     // The starting position of each kernel in the kernels array
		int*   sizes;       // The size of each kernel (number of elements)
		int    numKernels;  // kernel number
	};

	explicit OperatorVarPsfDevice(const ImageParams& p_imageParams, const cudaStream_t* pp_stream = nullptr);
	explicit OperatorVarPsfDevice(const std::string& pr_imagePsf_fname,
							   const ImageParams& p_imageParams, const cudaStream_t* pp_stream = nullptr);

	void copyVarPsfToDevice(bool synchronize = true);// copy kernel LUT to device: allocate and upload

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;
	void applyA(const Variable* in, Variable* out, bool synchronize) const;
	void applyAH(const Variable* in, Variable* out, bool synchronize) const;




protected:
	void initDeviceArraysIfNeeded();
	void allocateDeviceArraysVarPsf(size_t nKernels, size_t totalKernelSize,
										bool synchronize);
	template <bool Transpose>
	void apply(const Variable* in, Variable* out, bool synchronize) const;
	template <bool IS_FWD>
	void varconvolveDevice(const ImageDevice& inputImage, ImageDevice& outputImage,
						   bool synchronize) const;

	mutable std::unique_ptr<ImageDeviceOwned> mpd_intermediaryImage;

private:
	// flattened GPU storage
	std::unique_ptr<DeviceArray<float>> mpd_kernelsFlat;
	std::unique_ptr<DeviceArray<int>> mpd_kernelOffsets;
	std::unique_ptr<DeviceArray<int>> mpd_kernelDims;       // triples: kx,ky,kz per kernel
	std::unique_ptr<DeviceArray<int>> mpd_kernelHalfSizes; // triples: hx,hy,hz per kernel
	//std::unique_ptr<DeviceArray<int>> mpd_kernelLUT;

	int lut_x_dim = 0;
	int lut_y_dim = 0;
	int lut_z_dim = 0;
	float d_xGap = 0.f;
	float d_yGap = 0.f;
	float d_zGap = 0.f;

	//static void initDeviceArrayIfNeeded(
		//std::unique_ptr<DeviceArray<float>>& ppd_kernel);

	//void readFromFileInternal(const std::string& pr_imagePsf_fname,
	  //                        bool p_synchronize);
	//static void initDeviceArrayIfNeeded(
	 //   std::unique_ptr<DeviceArray<float>>& ppd_kernel);
	//void flattenKernelsToDevice(bool synchronize);

	// scalar parameters copied to kernels as args
	//float d_xCenter = 0.f, d_yCenter = 0.f, d_zCenter = 0.f;
};
}  // namespace yrt