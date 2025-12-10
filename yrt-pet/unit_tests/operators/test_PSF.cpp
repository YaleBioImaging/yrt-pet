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
#include <cmath>
#include <cstring>
#include <ctime>
#include <random>

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
		ImageParams imgParams{isGPU ? 50 : 30,
		                      isGPU ? 50 : 30,
		                      isGPU ? 25 : 15,
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

		std::vector<int64_t> dims = {imgParams.nx, imgParams.ny, imgParams.nz};
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

TEST_CASE("VarPSF", "[varpsf]")
{
	// Random sigma generator
	const unsigned int randomSeed =
	    static_cast<unsigned int>(std::time(nullptr));

	std::default_random_engine engine(randomSeed);

	for (int i = 0; i < 100; ++i)
	{
		ImageParams imgParams{100,    100,  51,   400.0f, 401.0f,
		                      421.0f, 0.0f, 0.0f, 0.0f};
		auto image = makeImageWithRandomPrism(imgParams, &engine);
		std::uniform_real_distribution<float> sigma_dist1(0.5f, 1.0f);

		float sigmaX1 = sigma_dist1(engine);
		float sigmaY1 = sigma_dist1(engine);
		float sigmaZ1 = sigma_dist1(engine);

		std::uniform_real_distribution<float> sigma_dist2(1.5f, 2.0f);
		float sigmaX2 = sigma_dist2(engine);
		float sigmaY2 = sigma_dist2(engine);
		float sigmaZ2 = sigma_dist2(engine);

		// Generate sigma_lookup on the fly using fixed (x,y,z) values
		OperatorVarPsf::ConvolutionKernelCollection random_sigma_lookup;
		std::vector<std::tuple<float, float, float>> positions;
		std::vector<float> xvals = {5.1f, 54.9f, 105.3f, 155.1f, 205.5f};
		std::vector<float> yvals = {5.1f, 54.9f, 105.3f, 155.1f, 204.9f};
		std::vector<float> zvals = {4.8f, 54.4f, 104.8f, 154.4f, 204.8f};
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
		OperatorVarPsf op_var(imgParams);
		op_var.setRangeAndGap(200, 50, 200, 50, 200, 50);
		float threshold = 100.0f;
		float tempx, tempy, tempz;
		float sigmax, sigmay, sigmaz;
		float nstdx = 4.0, nstdy = 4.0, nstdz = 4.0;
		OperatorVarPsf::ConvolutionKernelCollection kernels;
		for (size_t i = 0; i < positions.size(); ++i)
		{
			std::tie(tempx, tempy, tempz) = positions[i];
			sigmax = (tempx > threshold) ? sigmaX2 : sigmaX1;
			sigmay = (tempy > threshold) ? sigmaY2 : sigmaY1;
			sigmaz = (tempz > threshold) ? sigmaZ2 : sigmaZ1;
			auto kernel = std::make_unique<ConvolutionKernelGaussian>(
			    sigmax, sigmay, sigmaz, nstdx, nstdy, nstdz, imgParams);
			kernels.push_back(std::move(kernel));
		}
		op_var.setKernelCollection(kernels);

		auto img_out = std::make_unique<ImageOwned>(imgParams);
		img_out->allocate();
		std::vector<int64_t> dims = {imgParams.nx, imgParams.ny, imgParams.nz};
		std::vector<float> voxels = {imgParams.vx, imgParams.vy, imgParams.vz};

		std::vector<float> sigmas1 = {sigmaX1, sigmaY1, sigmaZ1};
		int kernel_size_x1 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas1[0] * nstdx) / imgParams.vx)) -
		        1);
		int kernel_size_y1 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas1[1] * nstdy) / imgParams.vy)) -
		        1);
		int kernel_size_z1 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas1[2] * nstdz) / imgParams.vz)) -
		        1);
		std::vector<float> sigmas2 = {sigmaX2, sigmaY2, sigmaZ2};
		int kernel_size_x2 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas2[0] * nstdx) / imgParams.vx)) -
		        1);
		int kernel_size_y2 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas2[1] * nstdy) / imgParams.vy)) -
		        1);
		int kernel_size_z2 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas2[2] * nstdz) / imgParams.vz)) -
		        1);
		std::vector<float> inputData;
		inputData.resize(image->getData().getSizeTotal());
		float* inputPtr = image->getRawPointer();
		float* outputPtr = img_out->getRawPointer();
		std::memcpy(inputData.data(), inputPtr,
		            image->getData().getSizeTotal() * sizeof(float));

		threshold = threshold - 25;
		Vector3D center_pt = {-imgParams.vx / 2, -imgParams.vy / 2,
		                      -imgParams.vz / 2};
		Vector3D test_pt1 = {threshold - imgParams.vx, 0, 0};
		Vector3D test_pt2 = {threshold + imgParams.vx,
		                     -threshold - imgParams.vy,
		                     threshold + imgParams.vz};
		Vector3D test_pt3 = {(imgParams.nx - 1) * imgParams.vx / 2,
		                     (imgParams.ny - 1) * imgParams.vy / 2,
		                     -(imgParams.nz - 1) * imgParams.vz / 2};
		int center_x, center_y, center_z;
		int tp1_x, tp1_y, tp1_z;
		int tp2_x, tp2_y, tp2_z;
		int tp3_x, tp3_y, tp3_z;

		image->getNearestNeighborIdx(center_pt, &center_x, &center_y,
		                             &center_z);
		image->getNearestNeighborIdx(test_pt1, &tp1_x, &tp1_y, &tp1_z);
		image->getNearestNeighborIdx(test_pt2, &tp2_x, &tp2_y, &tp2_z);
		image->getNearestNeighborIdx(test_pt3, &tp3_x, &tp3_y, &tp3_z);

		SECTION("forward_varpsf")
		{
			op_var.applyA(image.get(), img_out.get());
			std::vector<float> expected1 =
			    convolve(inputData, dims, voxels, sigmas1, false,
			             kernel_size_x1, kernel_size_y1, kernel_size_z1);

			std::vector<float> expected2 =
			    convolve(inputData, dims, voxels, sigmas2, false,
			             kernel_size_x2, kernel_size_y2, kernel_size_z2);

			size_t idx =
			    center_x + imgParams.nx * (center_y + imgParams.ny * center_z);
			CHECK(outputPtr[idx] == Approx(expected1[idx]).epsilon(1e-3));
			idx = tp1_x + imgParams.nx * (tp1_y + imgParams.ny * tp1_z);
			CHECK(outputPtr[idx] == Approx(expected1[idx]).epsilon(1e-3));
			idx = tp2_x + imgParams.nx * (tp2_y + imgParams.ny * tp2_z);
			CHECK(outputPtr[idx] == Approx(expected2[idx]).epsilon(1e-3));
			idx = tp3_x + imgParams.nx * (tp3_y + imgParams.ny * tp3_z);
			CHECK(outputPtr[idx] == Approx(expected2[idx]).epsilon(1e-3));
		}

		SECTION("transpose_varpsf")
		{
			op_var.applyAH(image.get(), img_out.get());
			std::vector<float> expected1 =
			    convolve(inputData, dims, voxels, sigmas1, true, kernel_size_x1,
			             kernel_size_y1, kernel_size_z1);
			std::vector<float> expected2 =
			    convolve(inputData, dims, voxels, sigmas2, true, kernel_size_x2,
			             kernel_size_y2, kernel_size_z2);

			size_t idx =
			    center_x + imgParams.nx * (center_y + imgParams.ny * center_z);
			CHECK(outputPtr[idx] == Approx(expected1[idx]).epsilon(1e-3));
			idx = tp1_x - kernel_size_x2 +
			      imgParams.nx * (tp1_y + imgParams.ny * tp1_z);
			CHECK(outputPtr[idx] == Approx(expected1[idx]).epsilon(1e-3));
			idx = tp2_x + kernel_size_x2 +
			      imgParams.nx * (tp2_y - kernel_size_y2 +
			                      imgParams.ny * (tp2_z + kernel_size_z2));
			CHECK(outputPtr[idx] == Approx(expected2[idx]).epsilon(1e-3));
			idx = tp3_x + imgParams.nx * (tp3_y + imgParams.ny * tp3_z);
			CHECK(outputPtr[idx] == Approx(expected2[idx]).epsilon(1e-3));
		}

		SECTION("adjoint_test_varpsf")
		{
			//<Ax,y> =? <x,Aty>
			auto img_out1 = std::make_unique<ImageOwned>(imgParams);
			img_out1->allocate();
			op_var.applyA(image.get(), img_out1.get());

			auto image2 = makeImageWithRandomPrism(imgParams);
			auto img_out2 = std::make_unique<ImageOwned>(imgParams);
			img_out2->allocate();
			op_var.applyAH(image2.get(), img_out2.get());

			// Compute dot products
			float lhs = img_out1->dotProduct(*image2);  // <Ax, y>
			float rhs = image->dotProduct(*img_out2);   // <x, Aty>
			CHECK(lhs == Approx(rhs).epsilon(1e-3));
		}
	}
}

#if BUILD_CUDA
TEST_CASE("VarPSF_GPU", "[varpsf_gpu]")
{
	// Random sigma generator
	const unsigned int randomSeed =
	    static_cast<unsigned int>(std::time(nullptr));

	std::default_random_engine engine(randomSeed);

	for (int i = 0; i < 100; ++i)
	{
		ImageParams imgParams{100,    100,  51,   400.0f, 401.0f,
		                      421.0f, 0.0f, 0.0f, 0.0f};
		auto image = makeImageWithRandomPrism(imgParams, &engine);

		std::uniform_real_distribution<float> sigma_dist1(0.5f, 1.0f);
		float sigmaX1 = sigma_dist1(engine);
		float sigmaY1 = sigma_dist1(engine);
		float sigmaZ1 = sigma_dist1(engine);

		std::uniform_real_distribution<float> sigma_dist2(1.5f, 2.0f);
		float sigmaX2 = sigma_dist2(engine);
		float sigmaY2 = sigma_dist2(engine);
		float sigmaZ2 = sigma_dist2(engine);

		// Generate sigma_lookup on the fly using fixed (x,y,z) values
		OperatorVarPsf::ConvolutionKernelCollection random_sigma_lookup;
		std::vector<std::tuple<float, float, float>> positions;
		std::vector<float> xvals = {5.1f, 54.9f, 105.3f, 155.1f, 205.5f};
		std::vector<float> yvals = {5.1f, 54.9f, 105.3f, 155.1f, 204.9f};
		std::vector<float> zvals = {4.8f, 54.4f, 104.8f, 154.4f, 204.8f};
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

		OperatorVarPsfDevice op_var(imgParams);
		op_var.setRangeAndGap(200, 50, 200, 50, 200, 50);

		float threshold = 100.0f;
		float tempx, tempy, tempz;
		float sigmax, sigmay, sigmaz;
		float nstdx = 4.0, nstdy = 4.0, nstdz = 4.0;
		OperatorVarPsf::ConvolutionKernelCollection kernels;
		for (size_t i = 0; i < positions.size(); ++i)
		{
			std::tie(tempx, tempy, tempz) = positions[i];
			sigmax = (tempx > threshold) ? sigmaX2 : sigmaX1;
			sigmay = (tempy > threshold) ? sigmaY2 : sigmaY1;
			sigmaz = (tempz > threshold) ? sigmaZ2 : sigmaZ1;
			auto kernel = std::make_unique<ConvolutionKernelGaussian>(
			    sigmax, sigmay, sigmaz, nstdx, nstdy, nstdz, imgParams);
			kernels.push_back(std::move(kernel));
		}
		op_var.setKernelCollection(kernels);
		op_var.copyVarPsfToDevice(true);

		auto img_out = std::make_unique<ImageOwned>(imgParams);
		img_out->allocate();

		std::vector<int64_t> dims = {imgParams.nx, imgParams.ny, imgParams.nz};
		std::vector<float> voxels = {imgParams.vx, imgParams.vy, imgParams.vz};

		std::vector<float> sigmas1 = {sigmaX1, sigmaY1, sigmaZ1};
		int kernel_size_x1 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas1[0] * nstdx) / imgParams.vx)) -
		        1);
		int kernel_size_y1 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas1[1] * nstdy) / imgParams.vy)) -
		        1);
		int kernel_size_z1 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas1[2] * nstdz) / imgParams.vz)) -
		        1);
		std::vector<float> sigmas2 = {sigmaX2, sigmaY2, sigmaZ2};
		int kernel_size_x2 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas2[0] * nstdx) / imgParams.vx)) -
		        1);
		int kernel_size_y2 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas2[1] * nstdy) / imgParams.vy)) -
		        1);
		int kernel_size_z2 = std::max(
		    1,
		    static_cast<int>(std::floor((sigmas2[2] * nstdz) / imgParams.vz)) -
		        1);

		std::vector<float> inputData;
		inputData.resize(image->getData().getSizeTotal());
		float* inputPtr = image->getRawPointer();
		float* outputPtr = img_out->getRawPointer();
		std::memcpy(inputData.data(), inputPtr,
		            image->getData().getSizeTotal() * sizeof(float));

		threshold = threshold - 25;
		Vector3D center_pt = {-imgParams.vx / 2, -imgParams.vy / 2,
		                      -imgParams.vz / 2};
		Vector3D test_pt1 = {threshold - imgParams.vx, 0, 0};
		Vector3D test_pt2 = {threshold + imgParams.vx,
		                     -threshold - imgParams.vy,
		                     threshold + imgParams.vz};
		Vector3D test_pt3 = {(imgParams.nx - 1) * imgParams.vx / 2,
		                     (imgParams.ny - 1) * imgParams.vy / 2,
		                     -(imgParams.nz - 1) * imgParams.vz / 2};
		int center_x, center_y, center_z;
		int tp1_x, tp1_y, tp1_z;
		int tp2_x, tp2_y, tp2_z;
		int tp3_x, tp3_y, tp3_z;

		image->getNearestNeighborIdx(center_pt, &center_x, &center_y,
		                             &center_z);
		image->getNearestNeighborIdx(test_pt1, &tp1_x, &tp1_y, &tp1_z);
		image->getNearestNeighborIdx(test_pt2, &tp2_x, &tp2_y, &tp2_z);
		image->getNearestNeighborIdx(test_pt3, &tp3_x, &tp3_y, &tp3_z);

		SECTION("forward_varpsf_gpu")
		{
			op_var.applyA(image.get(), img_out.get());
			std::vector<float> expected1 =
			    convolve(inputData, dims, voxels, sigmas1, false,
			             kernel_size_x1, kernel_size_y1, kernel_size_z1);

			std::vector<float> expected2 =
			    convolve(inputData, dims, voxels, sigmas2, false,
			             kernel_size_x2, kernel_size_y2, kernel_size_z2);

			size_t idx =
			    center_x + imgParams.nx * (center_y + imgParams.ny * center_z);
			CHECK(outputPtr[idx] == Approx(expected1[idx]).epsilon(1e-3));
			idx = tp1_x + imgParams.nx * (tp1_y + imgParams.ny * tp1_z);
			CHECK(outputPtr[idx] == Approx(expected1[idx]).epsilon(1e-3));
			idx = tp2_x + imgParams.nx * (tp2_y + imgParams.ny * tp2_z);
			CHECK(outputPtr[idx] == Approx(expected2[idx]).epsilon(1e-3));
			idx = tp3_x + imgParams.nx * (tp3_y + imgParams.ny * tp3_z);
			CHECK(outputPtr[idx] == Approx(expected2[idx]).epsilon(1e-3));
		}

		SECTION("transpose_varpsf_gpu")
		{
			op_var.applyAH(image.get(), img_out.get());
			std::vector<float> expected1 =
			    convolve(inputData, dims, voxels, sigmas1, true, kernel_size_x1,
			             kernel_size_y1, kernel_size_z1);
			std::vector<float> expected2 =
			    convolve(inputData, dims, voxels, sigmas2, true, kernel_size_x2,
			             kernel_size_y2, kernel_size_z2);

			size_t idx =
			    center_x + imgParams.nx * (center_y + imgParams.ny * center_z);
			CHECK(outputPtr[idx] == Approx(expected1[idx]).epsilon(1e-3));
			idx = tp1_x - kernel_size_x2 +
			      imgParams.nx * (tp1_y + imgParams.ny * tp1_z);
			CHECK(outputPtr[idx] == Approx(expected1[idx]).epsilon(1e-3));
			idx = tp2_x + kernel_size_x2 +
			      imgParams.nx * (tp2_y - kernel_size_y2 +
			                      imgParams.ny * (tp2_z + kernel_size_z2));
			CHECK(outputPtr[idx] == Approx(expected2[idx]).epsilon(1e-3));
			idx = tp3_x + imgParams.nx * (tp3_y + imgParams.ny * tp3_z);
			CHECK(outputPtr[idx] == Approx(expected2[idx]).epsilon(1e-3));
		}

		SECTION("adjoint_test_varpsf_gpu")
		{
			// <Ax, y> =? <x, A^T y>
			auto img_out1 = std::make_unique<ImageOwned>(imgParams);
			img_out1->allocate();
			op_var.applyA(image.get(), img_out1.get());

			auto image2 = makeImageWithRandomPrism(imgParams);
			auto img_out2 = std::make_unique<ImageOwned>(imgParams);
			img_out2->allocate();
			op_var.applyAH(image2.get(), img_out2.get());

			float lhs = img_out1->dotProduct(*image2);  // <Ax, y>
			float rhs = image->dotProduct(*img_out2);   // <x, A^T y>
			CHECK(lhs == Approx(rhs).epsilon(1e-3));
		}
	}
}
#endif  // BUILD_CUDA
}  // namespace yrt::util::test
