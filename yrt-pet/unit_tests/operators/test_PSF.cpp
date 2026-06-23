/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../unit_tests/test_utils.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"
#include "yrt-pet/operators/OperatorVarPsf.hpp"
#include "yrt-pet/utils/Assert.hpp"

#if BUILD_CUDA
#include "yrt-pet/operators/OperatorPsfDevice.cuh"
#include "yrt-pet/operators/OperatorVarPsfDevice.cuh"
#endif

#include "catch.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <tuple>

namespace yrt::util::test
{
std::vector<float> generateSymmetricGaussianKernel(int size, float sigma)
{
	std::vector<float> kernel(size);
	int halfSize = size / 2;
	float sum = 0.0f;

	for (int i = 0; i < size; ++i)
	{
		int x = i - halfSize;
		kernel[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
		sum += kernel[i];
	}

	// Normalize to make the sum = 1
	for (auto& v : kernel)
	{
		v /= sum;
	}

	return kernel;
}

template <typename T>
std::vector<float>
    convolve(const std::vector<T>& data, const std::vector<int64_t>& dims = {},
             const std::vector<float>& voxels = {},
             const std::vector<float>& sigma = {}, const bool T_flag = false,
             const int kernel_size_x = 2, const int kernel_size_y = 2,
             const int kernel_size_z = 1)
{
	//  Ensure the dimensions are provided and valid
	if (dims.size() != 3)
	{
		throw std::invalid_argument(
		    "dims must contain exactly 3 elements representing the dimensions "
		    "of the 3D volume.");
	}
	int x_dim = dims[0];
	int y_dim = dims[1];
	int z_dim = dims[2];
	float sigmax = sigma[0];
	float sigmay = sigma[1];
	float sigmaz = sigma[2];
	float xoffset, yoffset, zoffset;
	int index1, index2;
	std::vector<float> Img_PSF(data.size(), 0.0f);
	float kernel_sum = 0.0f;

	// calculate kernel coefficient in advance
	std::vector<std::vector<std::vector<float>>> psf_kernel(
	    kernel_size_x * 2 + 1,
	    std::vector<std::vector<float>>(
	        kernel_size_y * 2 + 1,
	        std::vector<float>(kernel_size_z * 2 + 1, 0.0f)));
	float inv_2_sigmax2 = 1.0f / (2 * sigmax * sigmax);
	float inv_2_sigmay2 = 1.0f / (2 * sigmay * sigmay);
	float inv_2_sigmaz2 = 1.0f / (2 * sigmaz * sigmaz);
	for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x; x_diff++)
		for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; y_diff++)
			for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z; z_diff++)
			{
				xoffset = x_diff * voxels[0];
				yoffset = y_diff * voxels[1];
				zoffset = z_diff * voxels[2];
				float temp;
				temp = (-xoffset * xoffset * inv_2_sigmax2) +
				       (-yoffset * yoffset * inv_2_sigmay2) +
				       (-zoffset * zoffset * inv_2_sigmaz2);
				psf_kernel[x_diff + kernel_size_x][y_diff + kernel_size_y]
				          [z_diff + kernel_size_z] = exp(temp);
				kernel_sum += exp(temp);
			}

	for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x; x_diff++)
		for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; y_diff++)
			for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z; z_diff++)
			{
				psf_kernel[x_diff + kernel_size_x][y_diff + kernel_size_y]
				          [z_diff + kernel_size_z] /= kernel_sum;
			}

	for (int i = 0; i < x_dim; i++)
	{
		for (int j = 0; j < y_dim; j++)
		{
			for (int k = 0; k < z_dim; k++)
			{
				for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z;
				     z_diff++)
				{
					for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y;
					     y_diff++)
					{
						for (int x_diff = -kernel_size_x;
						     x_diff <= kernel_size_x; x_diff++)
						{
							int ii = (i + x_diff + x_dim) % x_dim;
							int jj = (j + y_diff + y_dim) % y_dim;
							int kk = (k + z_diff + z_dim) % z_dim;
							xoffset = x_diff * voxels[0];
							yoffset = y_diff * voxels[1];
							zoffset = z_diff * voxels[2];
							index1 = i + j * x_dim + k * x_dim * y_dim;
							index2 = ii + jj * x_dim + kk * x_dim * y_dim;
							if (T_flag == 0)
							{
								Img_PSF[index1] +=
								    data[index2] *
								    psf_kernel[x_diff + kernel_size_x]
								              [y_diff + kernel_size_y]
								              [z_diff + kernel_size_z];
							}
							else
							{
								Img_PSF[index1] +=
								    data[index2] *
								    psf_kernel[-x_diff + kernel_size_x]
								              [-y_diff + kernel_size_y]
								              [-z_diff + kernel_size_z];
							}
						}
					}
				}
			}
		}
	}
	return Img_PSF;
}

template <typename T>
std::vector<float> convolveGaussianMixture(
    const std::vector<T>& data, const std::vector<int64_t>& dims,
    const std::vector<float>& voxels, const std::vector<float>& sigma1,
    const std::vector<float>& sigma2, float weight2, const bool T_flag,
    const int kernel_size_x, const int kernel_size_y, const int kernel_size_z)
{
	if (dims.size() != 3)
	{
		throw std::invalid_argument(
		    "dims must contain exactly 3 elements representing the dimensions "
		    "of the 3D volume.");
	}
	int x_dim = dims[0];
	int y_dim = dims[1];
	int z_dim = dims[2];
	float xoffset, yoffset, zoffset;
	int index1, index2;
	std::vector<float> Img_PSF(data.size(), 0.0f);
	float kernel1_sum = 0.0f;
	float kernel2_sum = 0.0f;
	const float weight1 = 1.0f - weight2;

	std::vector<std::vector<std::vector<float>>> psf_kernel(
	    kernel_size_x * 2 + 1,
	    std::vector<std::vector<float>>(
	        kernel_size_y * 2 + 1,
	        std::vector<float>(kernel_size_z * 2 + 1, 0.0f)));
	const float inv_2_sigmax1_2 = 1.0f / (2 * sigma1[0] * sigma1[0]);
	const float inv_2_sigmay1_2 = 1.0f / (2 * sigma1[1] * sigma1[1]);
	const float inv_2_sigmaz1_2 = 1.0f / (2 * sigma1[2] * sigma1[2]);
	const float inv_2_sigmax2_2 = 1.0f / (2 * sigma2[0] * sigma2[0]);
	const float inv_2_sigmay2_2 = 1.0f / (2 * sigma2[1] * sigma2[1]);
	const float inv_2_sigmaz2_2 = 1.0f / (2 * sigma2[2] * sigma2[2]);
	for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x; x_diff++)
		for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; y_diff++)
			for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z; z_diff++)
			{
				xoffset = x_diff * voxels[0];
				yoffset = y_diff * voxels[1];
				zoffset = z_diff * voxels[2];
				const float exp1 =
				    (-xoffset * xoffset * inv_2_sigmax1_2) +
				    (-yoffset * yoffset * inv_2_sigmay1_2) +
				    (-zoffset * zoffset * inv_2_sigmaz1_2);
				const float exp2 =
				    (-xoffset * xoffset * inv_2_sigmax2_2) +
				    (-yoffset * yoffset * inv_2_sigmay2_2) +
				    (-zoffset * zoffset * inv_2_sigmaz2_2);
				const float value1 = exp(exp1);
				const float value2 = exp(exp2);
				psf_kernel[x_diff + kernel_size_x][y_diff + kernel_size_y]
				          [z_diff + kernel_size_z] = value1;
				kernel1_sum += value1;
				kernel2_sum += value2;
			}

	for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x; x_diff++)
		for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; y_diff++)
			for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z; z_diff++)
			{
				xoffset = x_diff * voxels[0];
				yoffset = y_diff * voxels[1];
				zoffset = z_diff * voxels[2];
				const float exp2 =
				    (-xoffset * xoffset * inv_2_sigmax2_2) +
				    (-yoffset * yoffset * inv_2_sigmay2_2) +
				    (-zoffset * zoffset * inv_2_sigmaz2_2);
				psf_kernel[x_diff + kernel_size_x][y_diff + kernel_size_y]
				          [z_diff + kernel_size_z] =
				    weight1 *
				        psf_kernel[x_diff + kernel_size_x]
				                  [y_diff + kernel_size_y]
				                  [z_diff + kernel_size_z] /
				        kernel1_sum +
				    weight2 * exp(exp2) / kernel2_sum;
			}

	for (int i = 0; i < x_dim; i++)
	{
		for (int j = 0; j < y_dim; j++)
		{
			for (int k = 0; k < z_dim; k++)
			{
				for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z;
				     z_diff++)
				{
					for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y;
					     y_diff++)
					{
						for (int x_diff = -kernel_size_x;
						     x_diff <= kernel_size_x; x_diff++)
						{
							int ii = (i + x_diff + x_dim) % x_dim;
							int jj = (j + y_diff + y_dim) % y_dim;
							int kk = (k + z_diff + z_dim) % z_dim;
							index1 = i + j * x_dim + k * x_dim * y_dim;
							index2 = ii + jj * x_dim + kk * x_dim * y_dim;
							if (T_flag == 0)
							{
								Img_PSF[index1] +=
								    data[index2] *
								    psf_kernel[x_diff + kernel_size_x]
								              [y_diff + kernel_size_y]
								              [z_diff + kernel_size_z];
							}
							else
							{
								Img_PSF[index1] +=
								    data[index2] *
								    psf_kernel[-x_diff + kernel_size_x]
								              [-y_diff + kernel_size_y]
								              [-z_diff + kernel_size_z];
							}
						}
					}
				}
			}
		}
	}
	return Img_PSF;
}

void copyVarPsfToDeviceIfNeeded(OperatorVarPsf&) {}

#if BUILD_CUDA
void copyVarPsfToDeviceIfNeeded(OperatorVarPsfDevice& op)
{
	op.copyVarPsfToDevice(true);
}
#endif

template <typename OperatorType>
void runVarPsfTest(std::default_random_engine& engine,
                   const std::string& suffix, bool useTwoGaussian)
{
	for (int i = 0; i < 100; ++i)
	{
		ImageParams imgParams{100,    100,  51,   400.0f, 401.0f,
		                      421.0f, 0.0f, 0.0f, 0.0f};
		auto image = makeImageWithRandomPrism(imgParams, &engine);

		std::uniform_real_distribution<float> sigma_dist1_low(0.5f, 1.0f);
		std::uniform_real_distribution<float> sigma_dist1_high(1.1f, 1.5f);
		std::uniform_real_distribution<float> sigma_dist2_low(1.6f, 2.0f);
		std::uniform_real_distribution<float> sigma_dist2_high(2.1f, 2.5f);
		std::uniform_real_distribution<float> weight_dist(0.2f, 0.6f);

		const std::vector<float> sigma1Low = {sigma_dist1_low(engine),
		                                      sigma_dist1_low(engine),
		                                      sigma_dist1_low(engine)};
		const std::vector<float> sigma1High = {sigma_dist1_high(engine),
		                                       sigma_dist1_high(engine),
		                                       sigma_dist1_high(engine)};
		const std::vector<float> sigma2Low = {sigma_dist2_low(engine),
		                                      sigma_dist2_low(engine),
		                                      sigma_dist2_low(engine)};
		const std::vector<float> sigma2High = {sigma_dist2_high(engine),
		                                       sigma_dist2_high(engine),
		                                       sigma_dist2_high(engine)};
		const float weight2 = weight_dist(engine);
		const std::string modeName = useTwoGaussian ? "_2g" : "";

		std::vector<std::tuple<float, float, float>> positions;
		const std::vector<float> xvals = {5.1f, 54.9f, 105.3f, 155.1f,
		                                  205.5f};
		const std::vector<float> yvals = {5.1f, 54.9f, 105.3f, 155.1f,
		                                  204.9f};
		const std::vector<float> zvals = {4.8f, 54.4f, 104.8f, 154.4f,
		                                  204.8f};
		for (float z : zvals)
		{
			for (float y : yvals)
			{
				for (float x : xvals)
				{
					positions.emplace_back(x, y, z);
				}
			}
		}

		OperatorType op_var(imgParams);
		op_var.setRangeAndGap(200, 50, 200, 50, 200, 50);

		const float threshold = 100.0f;
		const float nstdx = 4.0f;
		const float nstdy = 4.0f;
		const float nstdz = 4.0f;
		OperatorVarPsf::ConvolutionKernelCollection kernels;
		for (size_t ii = 0; ii < positions.size(); ++ii)
		{
			float tempx, tempy, tempz;
			std::tie(tempx, tempy, tempz) = positions[ii];
			const std::vector<float> sigmas1 = {
			    (tempx > threshold) ? sigma1High[0] : sigma1Low[0],
			    (tempy > threshold) ? sigma1High[1] : sigma1Low[1],
			    (tempz > threshold) ? sigma1High[2] : sigma1Low[2]};
			const std::vector<float> sigmas2 = {
			    (tempx > threshold) ? sigma2High[0] : sigma2Low[0],
			    (tempy > threshold) ? sigma2High[1] : sigma2Low[1],
			    (tempz > threshold) ? sigma2High[2] : sigma2Low[2]};
			std::unique_ptr<ConvolutionKernel> kernel;
			if (useTwoGaussian)
			{
				kernel = std::make_unique<ConvolutionKernelGaussianMixture>(
				    sigmas1[0], sigmas1[1], sigmas1[2], sigmas2[0],
				    sigmas2[1], sigmas2[2], weight2, nstdx, nstdy, nstdz,
				    imgParams);
			}
			else
			{
				kernel = std::make_unique<ConvolutionKernelGaussian>(
				    sigmas1[0], sigmas1[1], sigmas1[2], nstdx, nstdy,
				    nstdz, imgParams);
			}
			kernels.push_back(std::move(kernel));
		}
		op_var.setKernelCollection(kernels);
		copyVarPsfToDeviceIfNeeded(op_var);

		auto img_out = std::make_unique<ImageOwned>(imgParams);
		img_out->allocate();

		const std::vector<int64_t> dims = {imgParams.nx, imgParams.ny,
		                                   imgParams.nz};
		const std::vector<float> voxels = {imgParams.vx, imgParams.vy,
		                                   imgParams.vz};
		std::vector<float> inputData(image->getData().getSizeTotal());
		float* outputPtr = img_out->getRawPointer();
		std::memcpy(inputData.data(), image->getRawPointer(),
		            image->getData().getSizeTotal() * sizeof(float));

		const auto kernelSize = [&imgParams, nstdx, nstdy, nstdz](
		                            const std::vector<float>& sigmas1,
		                            const std::vector<float>& sigmas2,
		                            bool useTwoGaussian)
		{
			const float sigmaX =
			    useTwoGaussian ? std::max(sigmas1[0], sigmas2[0]) : sigmas1[0];
			const float sigmaY =
			    useTwoGaussian ? std::max(sigmas1[1], sigmas2[1]) : sigmas1[1];
			const float sigmaZ =
			    useTwoGaussian ? std::max(sigmas1[2], sigmas2[2]) : sigmas1[2];
			return std::array<int, 3>{
			    std::max(
			        1,
			        static_cast<int>(
			            std::floor((sigmaX * nstdx) / imgParams.vx)) -
			            1),
			    std::max(
			        1,
			        static_cast<int>(
			            std::floor((sigmaY * nstdy) / imgParams.vy)) -
			            1),
			    std::max(
			        1,
			        static_cast<int>(
			            std::floor((sigmaZ * nstdz) / imgParams.vz)) -
			            1)};
		};
		const auto makeExpected =
		    [&](const std::vector<float>& sigmas1,
		        const std::vector<float>& sigmas2, bool transpose)
		{
			const auto ksize = kernelSize(sigmas1, sigmas2, useTwoGaussian);
			if (useTwoGaussian)
			{
				return convolveGaussianMixture(
				    inputData, dims, voxels, sigmas1, sigmas2, weight2,
				    transpose, ksize[0], ksize[1], ksize[2]);
			}
			return convolve(inputData, dims, voxels, sigmas1, transpose,
			                ksize[0], ksize[1], ksize[2]);
		};

		const Vector3D center_pt = {-imgParams.vx / 2, -imgParams.vy / 2,
		                            -imgParams.vz / 2};
		const Vector3D test_pt1 = {threshold + imgParams.vx, 0, 0};
		const Vector3D test_pt2 = {threshold + imgParams.vx,
		                           -threshold - imgParams.vy,
		                           threshold + imgParams.vz};
		const Vector3D test_pt3 = {(imgParams.nx - 1) * imgParams.vx / 2,
		                           (imgParams.ny - 1) * imgParams.vy / 2,
		                           -(imgParams.nz - 1) * imgParams.vz / 2};
		ssize_t center_x, center_y, center_z;
		ssize_t tp1_x, tp1_y, tp1_z;
		ssize_t tp2_x, tp2_y, tp2_z;
		ssize_t tp3_x, tp3_y, tp3_z;
		image->getNearestNeighborIdx(center_pt, &center_x, &center_y,
		                             &center_z);
		image->getNearestNeighborIdx(test_pt1, &tp1_x, &tp1_y, &tp1_z);
		image->getNearestNeighborIdx(test_pt2, &tp2_x, &tp2_y, &tp2_z);
		image->getNearestNeighborIdx(test_pt3, &tp3_x, &tp3_y, &tp3_z);

		const std::vector<float> sigmas1LLL = {sigma1Low[0], sigma1Low[1],
		                                       sigma1Low[2]};
		const std::vector<float> sigmas2LLL = {sigma2Low[0], sigma2Low[1],
		                                       sigma2Low[2]};
		const std::vector<float> sigmas1HLL = {sigma1High[0], sigma1Low[1],
		                                       sigma1Low[2]};
		const std::vector<float> sigmas2HLL = {sigma2High[0], sigma2Low[1],
		                                       sigma2Low[2]};
		const std::vector<float> sigmas1HHH = {sigma1High[0], sigma1High[1],
		                                       sigma1High[2]};
		const std::vector<float> sigmas2HHH = {sigma2High[0], sigma2High[1],
		                                       sigma2High[2]};

		SECTION("forward_varpsf" + modeName + suffix)
		{
			op_var.applyA(image.get(), img_out.get());
			const std::vector<float> expectedLLL =
			    makeExpected(sigmas1LLL, sigmas2LLL, false);
			const std::vector<float> expectedHLL =
			    makeExpected(sigmas1HLL, sigmas2HLL, false);
			const std::vector<float> expectedHHH =
			    makeExpected(sigmas1HHH, sigmas2HHH, false);

			size_t idx =
			    center_x + imgParams.nx * (center_y + imgParams.ny * center_z);
			CHECK(outputPtr[idx] == Approx(expectedLLL[idx]).epsilon(1e-3));
			idx = tp1_x + imgParams.nx * (tp1_y + imgParams.ny * tp1_z);
			CHECK(outputPtr[idx] == Approx(expectedHLL[idx]).epsilon(1e-3));
			idx = tp2_x + imgParams.nx * (tp2_y + imgParams.ny * tp2_z);
			CHECK(outputPtr[idx] == Approx(expectedHHH[idx]).epsilon(1e-3));
			idx = tp3_x + imgParams.nx * (tp3_y + imgParams.ny * tp3_z);
			CHECK(outputPtr[idx] == Approx(expectedHHH[idx]).epsilon(1e-3));
		}

		SECTION("transpose_varpsf" + modeName + suffix)
		{
			op_var.applyAH(image.get(), img_out.get());
			const std::vector<float> expectedLLL =
			    makeExpected(sigmas1LLL, sigmas2LLL, true);
			const std::vector<float> expectedHLL =
			    makeExpected(sigmas1HLL, sigmas2HLL, true);
			const std::vector<float> expectedHHH =
			    makeExpected(sigmas1HHH, sigmas2HHH, true);
			const auto ksizeHLL =
			    kernelSize(sigmas1HLL, sigmas2HLL, useTwoGaussian);
			const auto ksizeHHH =
			    kernelSize(sigmas1HHH, sigmas2HHH, useTwoGaussian);

			size_t idx =
			    center_x + imgParams.nx * (center_y + imgParams.ny * center_z);
			CHECK(outputPtr[idx] == Approx(expectedLLL[idx]).epsilon(1e-3));
			idx = tp1_x - ksizeHLL[0] +
			      imgParams.nx * (tp1_y + imgParams.ny * tp1_z);
			CHECK(outputPtr[idx] == Approx(expectedHLL[idx]).epsilon(1e-3));
			idx = tp2_x + ksizeHHH[0] +
			      imgParams.nx * (tp2_y - ksizeHHH[1] +
			                      imgParams.ny * (tp2_z + ksizeHHH[2]));
			CHECK(outputPtr[idx] == Approx(expectedHHH[idx]).epsilon(1e-3));
			idx = tp3_x + imgParams.nx * (tp3_y + imgParams.ny * tp3_z);
			CHECK(outputPtr[idx] == Approx(expectedHHH[idx]).epsilon(1e-3));
		}

		SECTION("adjoint_test_varpsf" + modeName + suffix)
		{
			auto img_out1 = std::make_unique<ImageOwned>(imgParams);
			img_out1->allocate();
			op_var.applyA(image.get(), img_out1.get());

			auto image2 = makeImageWithRandomPrism(imgParams);
			auto img_out2 = std::make_unique<ImageOwned>(imgParams);
			img_out2->allocate();
			op_var.applyAH(image2.get(), img_out2.get());

			const float lhs = img_out1->dotProduct(*image2);
			const float rhs = image->dotProduct(*img_out2);
			CHECK(lhs == Approx(rhs).epsilon(1e-3));
		}
	}
}

TEST_CASE("PSF", "[psf]")
{
#if BUILD_CUDA
	int numModes = 2;
#else
	int numModes = 1;
#endif
	for (int mode = 1; mode <= numModes; ++mode)
	{
		bool isGPU = (mode == 2);
		std::string modeName = isGPU ? "GPU" : "CPU";
		ImageParams imgParams{isGPU ? 50ll : 30ll,
		                      isGPU ? 50ll : 30ll,
		                      isGPU ? 25ll : 15ll,
		                      isGPU ? 60.0f : 30.0f,
		                      isGPU ? 59.0f : 31.0f,
		                      isGPU ? 23.0f : 15.0f,
		                      0.0f,
		                      0.0f,
		                      0.0f};
		auto image = makeImageWithRandomPrism(imgParams);

		// Generate random sigma and Gaussian kernels
		std::mt19937 gen(static_cast<unsigned int>(std::time(0)));
		std::uniform_real_distribution<float> sigma_dist(0.5f, 2.0f);
		float sigmaX = sigma_dist(gen);
		float sigmaY = sigma_dist(gen);
		float sigmaZ = sigma_dist(gen);
		std::vector<float> kernelX =
		    generateSymmetricGaussianKernel(5, sigmaX / imgParams.vx);
		std::vector<float> kernelY =
		    generateSymmetricGaussianKernel(5, sigmaY / imgParams.vy);
		std::vector<float> kernelZ =
		    generateSymmetricGaussianKernel(3, sigmaZ / imgParams.vz);
		std::unique_ptr<OperatorPsf> op;

		if (isGPU)
		{
#if BUILD_CUDA
			op = std::make_unique<OperatorPsfDevice>(kernelX, kernelY, kernelZ);
#else
			ASSERT_MSG(false, "Attempting GPU run with non-CUDA build");
#endif
		}
		else
		{
			op = std::make_unique<OperatorPsf>(kernelX, kernelY, kernelZ);
		}

		auto img_out = std::make_unique<ImageOwned>(imgParams);
		img_out->allocate();

		std::vector<int64_t> dims = {static_cast<int64_t>(imgParams.nx),
		                             static_cast<int64_t>(imgParams.ny),
		                             static_cast<int64_t>(imgParams.nz)};
		std::vector<float> voxels = {imgParams.vx, imgParams.vy, imgParams.vz};
		std::vector<float> sigmas = {sigmaX, sigmaY, sigmaZ};

		std::vector<float> inputData(image->getData().getSizeTotal());
		std::memcpy(inputData.data(), image->getRawPointer(),
		            inputData.size() * sizeof(float));
		float* outputPtr = img_out->getRawPointer();

		SECTION("forward_psf_" + modeName)
		{
			op->applyA(image.get(), img_out.get());
			std::vector<float> expected =
			    convolve(inputData, dims, voxels, sigmas, false);
			for (size_t i = 0; i < expected.size(); ++i)
			{
				CHECK(outputPtr[i] == Approx(expected[i]).epsilon(1e-3));
			}
		}

		SECTION("transpose_psf_" + modeName)
		{
			op->applyAH(image.get(), img_out.get());
			std::vector<float> expected =
			    convolve(inputData, dims, voxels, sigmas, true);
			for (size_t i = 0; i < expected.size(); ++i)
			{
				CHECK(outputPtr[i] == Approx(expected[i]).epsilon(1e-3));
			}
		}

		SECTION("adjoint_test_psf_" + modeName)
		{
			auto img_out1 = std::make_unique<ImageOwned>(imgParams);
			img_out1->allocate();
			op->applyA(image.get(), img_out1.get());

			auto image2 = makeImageWithRandomPrism(imgParams);
			auto img_out2 = std::make_unique<ImageOwned>(imgParams);
			img_out2->allocate();
			op->applyAH(image2.get(), img_out2.get());

			float lhs = img_out1->dotProduct(*image2);  // <Ax, y>
			float rhs = image->dotProduct(*img_out2);   // <x, Aty>
			CHECK(lhs == Approx(rhs).epsilon(1e-3));
		}
	}
}

TEST_CASE("PSF_4D", "[psf]")
{
#if BUILD_CUDA
	int numModes = 2;
#else
	int numModes = 1;
#endif
	for (int useGPU = 0; useGPU < numModes; useGPU++)
	{
		ImageParams imgParams4D(30, 30, 15, 30.0f, 30.0f, 15.0f, 0.0f, 0.0f,
		                        0.0f, 2);

		ImageParams imgParams3D{imgParams4D.nx,       imgParams4D.ny,
		                        imgParams4D.nz,       imgParams4D.length_x,
		                        imgParams4D.length_y, imgParams4D.length_z,
		                        imgParams4D.off_x,    imgParams4D.off_y,
		                        imgParams4D.off_z,    1};
		auto image3DFrame0 = makeImageWithRandomPrism(imgParams3D);
		auto image3DFrame1 = makeImageWithRandomPrism(imgParams3D);
		auto image4D = std::make_unique<ImageOwned>(imgParams4D);
		image4D->allocate();

		const size_t frameSize = static_cast<size_t>(imgParams4D.nx) *
		                         imgParams4D.ny * imgParams4D.nz;
		for (size_t i = 0; i < frameSize; i++)
		{
			image4D->getRawPointer()[i] = image3DFrame0->getRawPointer()[i];
			image4D->getRawPointer()[frameSize + i] =
			    image3DFrame1->getRawPointer()[i];
		}

		std::mt19937 gen(static_cast<unsigned int>(std::time(0)));
		std::uniform_real_distribution<float> sigmaDist(0.5f, 2.0f);
		float sigmaX = sigmaDist(gen);
		float sigmaY = sigmaDist(gen);
		float sigmaZ = sigmaDist(gen);
		std::vector<float> kernelX =
		    generateSymmetricGaussianKernel(5, sigmaX / imgParams4D.vx);
		std::vector<float> kernelY =
		    generateSymmetricGaussianKernel(5, sigmaY / imgParams4D.vy);
		std::vector<float> kernelZ =
		    generateSymmetricGaussianKernel(3, sigmaZ / imgParams4D.vz);

		std::unique_ptr<Operator> op;
		if (useGPU)
		{
#if BUILD_CUDA
			op = std::make_unique<OperatorPsfDevice>(kernelX, kernelY, kernelZ);
#endif
		}
		else
		{
			op = std::make_unique<OperatorPsf>(kernelX, kernelY, kernelZ);
		}

		auto img_out4D = std::make_unique<ImageOwned>(imgParams4D);
		img_out4D->allocate();
		auto img_out3D_frame0 = std::make_unique<ImageOwned>(imgParams3D);
		img_out3D_frame0->allocate();
		auto img_out3D_frame1 = std::make_unique<ImageOwned>(imgParams3D);
		img_out3D_frame1->allocate();

		op->applyA(image4D.get(), img_out4D.get());
		op->applyA(image3DFrame0.get(), img_out3D_frame0.get());
		op->applyA(image3DFrame1.get(), img_out3D_frame1.get());

		for (size_t i = 0; i < frameSize; i++)
		{
			CHECK(img_out4D->getRawPointer()[i] ==
			      Approx(img_out3D_frame0->getRawPointer()[i]).epsilon(1e-4));
		}
		for (size_t i = 0; i < frameSize; i++)
		{
			CHECK(img_out4D->getRawPointer()[frameSize + i] ==
			      Approx(img_out3D_frame1->getRawPointer()[i]).epsilon(1e-4));
		}
	}
}

TEST_CASE("VarPSF", "[varpsf]")
{
	const unsigned int randomSeed =
	    static_cast<unsigned int>(std::time(nullptr));
	std::default_random_engine engine(randomSeed);
	runVarPsfTest<OperatorVarPsf>(engine, "", false);
}
TEST_CASE("VarPSF_readFromFile_auto_detect", "[varpsf]")
{
	const ImageParams imgParams{4, 4, 3, 8.0f, 8.0f, 6.0f, 0.0f, 0.0f,
	                            0.0f};
	const std::string fname1g = "tmp_varpsf_auto_1g.csv";
	const std::string fname2g = "tmp_varpsf_auto_2g.csv";
	{
		std::ofstream file(fname1g);
		file << "4,4,4\n"
		     << "4,4,4\n"
		     << "4,4,4\n"
		     << "0.5,0.5,0.5\n";
	}
	{
		std::ofstream file(fname2g);
		file << "4,4,4,0,0,0,0\n"
		     << "4,4,4,0,0,0,0\n"
		     << "4,4,4,0,0,0,0\n"
		     << "0.5,0.5,0.5,1.6,1.6,1.6,0.3\n";
	}

	OperatorVarPsf op1g(fname1g, imgParams);
	OperatorVarPsf op2g(fname2g, imgParams);

	std::remove(fname1g.c_str());
	std::remove(fname2g.c_str());
}

TEST_CASE("VarPSF_2G", "[varpsf_2g]")
{
	const unsigned int randomSeed =
	    static_cast<unsigned int>(std::time(nullptr));
	std::default_random_engine engine(randomSeed);
	runVarPsfTest<OperatorVarPsf>(engine, "", true);
}
#if BUILD_CUDA
TEST_CASE("VarPSF_GPU", "[varpsf_gpu]")
{
	const unsigned int randomSeed =
	    static_cast<unsigned int>(std::time(nullptr));
	std::default_random_engine engine(randomSeed);
	runVarPsfTest<OperatorVarPsfDevice>(engine, "_gpu", false);
}
TEST_CASE("VarPSF_2G_GPU", "[varpsf_2g_gpu]")
{
	const unsigned int randomSeed =
	    static_cast<unsigned int>(std::time(nullptr));
	std::default_random_engine engine(randomSeed);
	runVarPsfTest<OperatorVarPsfDevice>(engine, "_gpu", true);
}
#endif  // BUILD_CUDA
}  // namespace yrt::util::test
