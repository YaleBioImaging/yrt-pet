/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorPsf.hpp"
#include "operators/OperatorPsfDevice.cuh"
#include "../unit_tests/test_utils.hpp"

#include "catch.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <ctime>
#include <cstring>

using namespace TestUtils;

std::vector<float> generateSymmetricGaussianKernel(int size, float sigma) {
    std::vector<float> kernel(size);
    int halfSize = size / 2;
    float sum = 0.0f;

    for (int i = 0; i < size; ++i) {
        int x = i - halfSize;
        kernel[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize to make the sum = 1
    for (auto& v : kernel) {
        v /= sum;
    }

    return kernel;
}

template <typename T>
std::vector<float> convolve(const std::vector<T> &data,  const std::vector<int64_t> &dims = {}, const std:: vector<float> &voxels = {}, 
	const std:: vector<float> &sigma = {},const int T_flag = 0) {
    //const float PI = 3.1415926f;
	// Ensure the dimensions are provided and valid
    if (dims.size() != 3) {
        throw std::invalid_argument("dims must contain exactly 3 elements representing the dimensions of the 3D volume.");
    }
    int x_dim = dims[0];
    int y_dim = dims[1];
    int z_dim = dims[2];
    float sigmax = sigma[0];
    float sigmay = sigma[1];
    float sigmaz = sigma[2];
    int kernel_size_x,kernel_size_y,kernel_size_z;
    
    kernel_size_x = 2;
    kernel_size_y = 2;
    kernel_size_z = 1;
    float xoffset,yoffset,zoffset;
    int index1,index2;
    std::vector<float> Img_PSF(data.size(),0.0f);
	float kernel_sum = 0.0f;

    //calculate kernel coefficient in advance
    std::vector<std::vector<std::vector<float>>> psf_kernel(kernel_size_x*2+1, std::vector<std::vector<float>>(kernel_size_y*2+1, std::vector<float>(kernel_size_z*2+1, 0.0f)));
    float inv_2_sigmax2 = 1.0f / (2 * sigmax * sigmax);
    float inv_2_sigmay2 = 1.0f / (2 * sigmay * sigmay);
    float inv_2_sigmaz2 = 1.0f / (2 * sigmaz * sigmaz);
    for (int x_diff = -kernel_size_x;x_diff<=kernel_size_x;x_diff++)
    for (int y_diff = -kernel_size_y;y_diff<=kernel_size_y;y_diff++)
    for (int z_diff = -kernel_size_z;z_diff<=kernel_size_z;z_diff++)
    {
        xoffset = x_diff*voxels[0];
        yoffset = y_diff*voxels[1];
        zoffset = z_diff*voxels[2];
        float temp;
        temp = (-xoffset*xoffset*inv_2_sigmax2)+(-yoffset*yoffset*inv_2_sigmay2)+(-zoffset*zoffset*inv_2_sigmaz2);
        psf_kernel[x_diff + kernel_size_x][y_diff + kernel_size_y][z_diff + kernel_size_z] = exp(temp);
		kernel_sum += exp(temp);
    }

	for (int x_diff = -kernel_size_x;x_diff<=kernel_size_x;x_diff++)
    for (int y_diff = -kernel_size_y;y_diff<=kernel_size_y;y_diff++)
    for (int z_diff = -kernel_size_z;z_diff<=kernel_size_z;z_diff++)
    {
        psf_kernel[x_diff + kernel_size_x][y_diff + kernel_size_y][z_diff + kernel_size_z] /= kernel_sum;
    }

    //padding 0
    for (int i=0; i<x_dim;i++)
    for (int j=0; j<y_dim;j++)
    for (int k=0; k<z_dim;k++)
    {
		for (int x_diff = -kernel_size_x;x_diff<=kernel_size_x;x_diff++)
		for (int y_diff = -kernel_size_y;y_diff<=kernel_size_y;y_diff++)
		for (int z_diff = -kernel_size_z;z_diff<=kernel_size_z;z_diff++)
		{
			int ii = (i + x_diff + x_dim) % x_dim;
			int jj = (j + y_diff + y_dim) % y_dim;
			int kk = (k + z_diff + z_dim) % z_dim;
			xoffset = x_diff*voxels[0];
			yoffset = y_diff*voxels[1];
			zoffset = z_diff*voxels[2];
			index1 = i+j*x_dim+k*x_dim*y_dim;
			index2 = ii+jj*x_dim+kk*x_dim*y_dim;
			if (T_flag == 0)
			{
				Img_PSF[index1] += data[index2]*psf_kernel[x_diff+kernel_size_x][y_diff+kernel_size_y][z_diff+kernel_size_z]; 
			}
			else
			{
				Img_PSF[index1] += data[index2]*psf_kernel[-x_diff+kernel_size_x][-y_diff+kernel_size_y][-z_diff+kernel_size_z]; 
			}
		}
    }
    return Img_PSF;
}

TEST_CASE("PSF", "[psf]")
{
	ImageParams imgParams{30, 30, 15, 30.0f, 31.0f, 15.0f, 0.0f, 0.0f, 0.0f};
	auto image = TestUtils::makeImageWithRandomPrism(imgParams);

	// Random sigma generator
    std::mt19937 gen(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> sigma_dist(0.5f, 2.0f);

	float sigmaX = sigma_dist(gen);
    float sigmaY = sigma_dist(gen);
    float sigmaZ = sigma_dist(gen);

	// Manually generate random PSF kernels
    std::vector<float> kernelX = generateSymmetricGaussianKernel(5,sigmaX/imgParams.vx);
    std::vector<float> kernelY = generateSymmetricGaussianKernel(5,sigmaY/imgParams.vy);
    std::vector<float> kernelZ = generateSymmetricGaussianKernel(3,sigmaZ/imgParams.vz);

	OperatorPsf op(kernelX, kernelY, kernelZ);
	auto img_out = std::make_unique<ImageOwned>(imgParams);
	img_out->allocate();
	std::vector<int64_t> dims = {imgParams.nx, imgParams.ny, imgParams.nz};
	std::vector<float> voxels = {imgParams.vx, imgParams.vy, imgParams.vz};
	std::vector<float> sigmas = {sigmaX, sigmaY, sigmaZ};
	std::vector<float> inputData;
	inputData.resize(image->getData().getSizeTotal());
	float* inputPtr = image->getRawPointer();
	float* outputPtr = img_out->getRawPointer();
	std::memcpy(inputData.data(), inputPtr, image->getData().getSizeTotal()*sizeof(float));

	SECTION("forward_psf") 
	{
        op.applyA(image.get(), img_out.get());
		std::vector<float> expected = convolve(inputData, dims, voxels, sigmas, 0);

		// Compare to output of op.applyA
		for (size_t i = 0; i < expected.size(); ++i) {
			CHECK(outputPtr[i] == Approx(expected[i]).epsilon(1e-3));
		}
    }

	SECTION("transpose_psf")
	{
		op.applyAH(image.get(), img_out.get());
		std::vector<float> expected = convolve(inputData, dims, voxels, sigmas, 1);

		// Compare to output of op.applyAH
		for (size_t i = 0; i < expected.size(); ++i) {
			CHECK(outputPtr[i] == Approx(expected[i]).epsilon(1e-3));
		}
	}

	SECTION("adjoint_psf")
	{
		//<Ax,y> =? <x,Aty>
		auto img_out1 = std::make_unique<ImageOwned>(imgParams);
		img_out1->allocate();
        op.applyA(image.get(), img_out1.get());

		auto image2 = TestUtils::makeImageWithRandomPrism(imgParams);
		auto img_out2 = std::make_unique<ImageOwned>(imgParams);
		img_out2->allocate();
        op.applyAH(image2.get(), img_out2.get());

		// Compute dot products
		float lhs = img_out1->dotProduct(*image2);  // <Ax, y>
		float rhs = image->dotProduct(*img_out2);   // <x, Aty>
		CHECK(lhs == Approx(rhs).epsilon(1e-3));
	}
}

TEST_CASE("PSF_GPU", "[psfgpu]")
{
	ImageParams imgParams{50, 50, 25, 60.0f, 59.0f, 23.0f, 0.0f, 0.0f, 0.0f};
	auto image = TestUtils::makeImageWithRandomPrism(imgParams);

	// Random sigma generator
    std::mt19937 gen(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> sigma_dist(0.5f, 2.0f);

	float sigmaX = sigma_dist(gen);
    float sigmaY = sigma_dist(gen);
    float sigmaZ = sigma_dist(gen);

	// Manually generate random PSF kernels
    std::vector<float> kernelX = generateSymmetricGaussianKernel(5,sigmaX/imgParams.vx);
    std::vector<float> kernelY = generateSymmetricGaussianKernel(5,sigmaY/imgParams.vy);
    std::vector<float> kernelZ = generateSymmetricGaussianKernel(3,sigmaZ/imgParams.vz);

	OperatorPsfDevice op_gpu(kernelX, kernelY, kernelZ);
	auto img_out = std::make_unique<ImageOwned>(imgParams);
	img_out->allocate();
	std::vector<int64_t> dims = {imgParams.nx, imgParams.ny, imgParams.nz};
	std::vector<float> voxels = {imgParams.vx, imgParams.vy, imgParams.vz};
	std::vector<float> sigmas = {sigmaX, sigmaY, sigmaZ};
	std::vector<float> inputData;
	inputData.resize(image->getData().getSizeTotal());
	float* inputPtr = image->getRawPointer();
	float* outputPtr = img_out->getRawPointer();
	std::memcpy(inputData.data(), inputPtr, image->getData().getSizeTotal()*sizeof(float));

	SECTION("forward_psf_gpu")
	{
        op_gpu.applyA(image.get(), img_out.get());
		std::vector<float> expected = convolve(inputData, dims, voxels, sigmas, 0);

		// Compare to output of op.applyA
		//for (size_t i = 0; i < expected.size(); ++i) {
		//	CHECK(outputPtr[i] == Approx(expected[i]).epsilon(1e-3));
		//}

		bool all_close = true;
		for (size_t i = 0; i < expected.size(); ++i) {
			if (!(Approx(expected[i]).epsilon(1e-3) == outputPtr[i])) {
				all_close = false;
				CHECK(outputPtr[i] == Approx(expected[i]).epsilon(1e-3));  // still log failure
			}
		}

		if (all_close) {
			std::cout << "PSF forward GPU test passed!" << std::endl;
		}
	}

	SECTION("transpose_psf_gpu")
	{
		op_gpu.applyAH(image.get(), img_out.get());
		std::vector<float> expected = convolve(inputData, dims, voxels, sigmas, 1);

		// Compare to output of op.applyAH
		//for (size_t i = 0; i < expected.size(); ++i) {
		//	CHECK(outputPtr[i] == Approx(expected[i]).epsilon(1e-3));
		//}

		bool all_close = true;
		for (size_t i = 0; i < expected.size(); ++i) {
			if (!(Approx(expected[i]).epsilon(1e-3) == outputPtr[i])) {
				all_close = false;
				CHECK(outputPtr[i] == Approx(expected[i]).epsilon(1e-3));
			}
		}

		if (all_close) {
			std::cout << "PSF Transposed GPU test passed!" << std::endl;
		}
	}

	SECTION("adjoint_psf_gpu")
	{
		//<Ax,y> =? <x,Aty>
		auto img_out1 = std::make_unique<ImageOwned>(imgParams);
		img_out1->allocate();
        op_gpu.applyA(image.get(), img_out1.get());

		auto image2 = TestUtils::makeImageWithRandomPrism(imgParams);
		auto img_out2 = std::make_unique<ImageOwned>(imgParams);
		img_out2->allocate();
        op_gpu.applyAH(image2.get(), img_out2.get());

		// Compute dot products
		float lhs = img_out1->dotProduct(*image2);  // <Ax, y>
		float rhs = image->dotProduct(*img_out2);   // <x, Aty>
		//CHECK(lhs == Approx(rhs).epsilon(1e-3));
		bool pass = Approx(lhs).epsilon(1e-3) == rhs;
		CHECK(pass);
		if (pass) {
			std::cout << "PSF adjoint GPU test passed: <Ax, y> = <x, Aty>" << std::endl;
		}
	}
}