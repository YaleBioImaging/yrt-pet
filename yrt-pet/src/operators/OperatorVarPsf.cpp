/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorVarPsf.hpp"

#include "yrt-pet/datastruct/image/ImageBase.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Tools.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_operatorvarpsf(py::module& m)
{
	auto c = py::class_<OperatorVarPsf, Operator>(m, "OperatorVarPsf");
	c.def(py::init<const ImageParams&>());
	c.def(py::init<const std::string&, const ImageParams&>());
	c.def("readFromFile", &OperatorVarPsf::readFromFile, py::arg("fname"),
	      "Read the variant PSF from CSV LUT");
	c.def(
	    "applyA", [](OperatorVarPsf& self, const Image* img_in, Image* img_out)
	    { self.applyA(img_in, img_out); }, py::arg("img_in"),
	    py::arg("img_out"));
	c.def(
	    "applyAH", [](OperatorVarPsf& self, const Image* img_in, Image* img_out)
	    { self.applyAH(img_in, img_out); }, py::arg("img_in"),
	    py::arg("img_out"));
}
}  // namespace yrt
#endif

namespace yrt
{
size_t ConvolutionKernel::getHalfSizeX() const
{
	return (psfKernel.getSize(2) - 1) / 2;
}

size_t ConvolutionKernel::getHalfSizeY() const
{
	return (psfKernel.getSize(1) - 1) / 2;
}

size_t ConvolutionKernel::getHalfSizeZ() const
{
	return (psfKernel.getSize(0) - 1) / 2;
}

ConvolutionKernel::ConvolutionKernel(const ConvolutionKernel& p_kernel)
{
	auto& p_array = p_kernel.getArray();
	auto dims = p_array.getDims();
	psfKernel.allocate(dims[0], dims[1], dims[2]);
	psfKernel.copy(p_kernel.getArray());
}

const ConvolutionKernel::KernelArray& ConvolutionKernel::getArray() const
{
	return psfKernel;
}

ConvolutionKernelGaussian::ConvolutionKernelGaussian(
    float p_sigmaX, float p_sigmaY, float p_sigmaZ, float p_nStdX,
    float p_nStdY, float p_nStdZ, const ImageParams& pr_imageParams)
    : ConvolutionKernel()
{
	setSigmas(p_sigmaX, p_sigmaY, p_sigmaZ, p_nStdX, p_nStdY, p_nStdZ,
	          pr_imageParams);
}

void ConvolutionKernelGaussian::setSigmas(float p_sigmaX, float p_sigmaY,
                                          float p_sigmaZ, float p_nStdX,
                                          float p_nStdY, float p_nStdZ,
                                          const ImageParams& pr_imageParams)
{
	m_sigmaX = p_sigmaX;
	m_sigmaY = p_sigmaY;
	m_sigmaZ = p_sigmaZ;
	m_nStdX = p_nStdX;
	m_nStdY = p_nStdY;
	m_nStdZ = p_nStdZ;

	int kernel_size_x = std::max(
	    1,
	    static_cast<int>(std::floor((m_sigmaX * m_nStdX) / pr_imageParams.vx)) -
	        1);
	int kernel_size_y = std::max(
	    1,
	    static_cast<int>(std::floor((m_sigmaY * m_nStdY) / pr_imageParams.vy)) -
	        1);
	int kernel_size_z = std::max(
	    1,
	    static_cast<int>(std::floor((m_sigmaZ * m_nStdZ) / pr_imageParams.vz)) -
	        1);

	const int kx_len = kernel_size_x * 2 + 1;
	const int ky_len = kernel_size_y * 2 + 1;
	const int kz_len = kernel_size_z * 2 + 1;
	psfKernel.allocate(kz_len, ky_len, kx_len);

	float inv_2_sigmax2 = 1.0f / (2 * m_sigmaX * m_sigmaX);
	float inv_2_sigmay2 = 1.0f / (2 * m_sigmaY * m_sigmaY);
	float inv_2_sigmaz2 = 1.0f / (2 * m_sigmaZ * m_sigmaZ);
	float kernel_sum = 0.0f;
	int idx = 0;

	for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z; ++z_diff)
	{
		for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; ++y_diff)
		{
			for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x;
			     ++x_diff, ++idx)
			{
				float xoffset = x_diff * pr_imageParams.vx;
				float yoffset = y_diff * pr_imageParams.vy;
				float zoffset = z_diff * pr_imageParams.vz;
				float temp = -(xoffset * xoffset * inv_2_sigmax2 +
				               yoffset * yoffset * inv_2_sigmay2 +
				               zoffset * zoffset * inv_2_sigmaz2);
				psfKernel.setFlat(idx, std::exp(temp));
				kernel_sum += psfKernel.getFlat(idx);
			}
		}
	}

	for (size_t i = 0; i < psfKernel.getSizeTotal(); ++i)
	{
		psfKernel.getFlat(i) /= kernel_sum;
	}
}

OperatorVarPsf::OperatorVarPsf(const ImageParams& pr_imageParams)
    : Operator{}, m_imageParams(pr_imageParams)
{
}

OperatorVarPsf::OperatorVarPsf(const std::string& imageVarPsf_fname,
                               const ImageParams& pr_imageParams)
    : OperatorVarPsf{pr_imageParams}
{
	readFromFile(imageVarPsf_fname);
}

const ConvolutionKernel& OperatorVarPsf::findNearestKernel(float x, float y,
                                                           float z) const
{
	int x_dim = static_cast<int>(std::floor(m_xRange / m_xGap)) + 1;
	int y_dim = static_cast<int>(std::floor(m_yRange / m_yGap)) + 1;
	int z_dim = static_cast<int>(std::floor(m_zRange / m_zGap)) + 1;

	int i = static_cast<int>(std::round(abs(x) / m_xGap));
	int j = static_cast<int>(std::round(abs(y) / m_yGap));
	int k = static_cast<int>(std::round(abs(z) / m_zGap));

	if (i >= x_dim)
	{
		i = x_dim - 1;
	}
	if (j >= y_dim)
	{
		j = y_dim - 1;
	}
	if (k >= z_dim)
	{
		k = z_dim - 1;
	}

	int index = IDX3(i, j, k, x_dim, y_dim);
	return *m_kernelLUT[index];
}

void OperatorVarPsf::readFromFile(const std::string& imageVarPsf_fname)
{
	std::cout << "Reading image space Variant PSF sigma lookup table file..."
	          << std::endl;
	Array2D<float> data;
	util::readCSV<float>(imageVarPsf_fname, data);
	size_t dims[2];
	data.getDims(dims);

	ASSERT_MSG(dims[0] > 3, "CSV file format error: At least 4 rows expected");
	ASSERT_MSG(dims[1] == 3, "CSV file format error: 3 columns expected");

	m_xRange = data[0][0];
	m_yRange = data[0][1];
	m_zRange = data[0][2];
	m_xGap = data[1][0];
	m_yGap = data[1][1];
	m_zGap = data[1][2];
	float nStdX = data[2][0];
	float nStdY = data[2][1];
	float nStdZ = data[2][2];

	for (size_t i = 3; i < dims[0]; ++i)
	{
		auto kernel = std::make_unique<ConvolutionKernelGaussian>(
		    data[i][0], data[i][1], data[i][2], nStdX, nStdY, nStdZ,
		    m_imageParams);
		m_kernelLUT.push_back(std::move(kernel));
	}
}

void OperatorVarPsf::setKernelCollection(
    const ConvolutionKernelCollection& p_kernelLUT)
{
	m_kernelLUT.clear();
	for (auto& kernel : p_kernelLUT)
	{
		m_kernelLUT.push_back(std::make_unique<ConvolutionKernel>(*kernel));
	}
}

void OperatorVarPsf::applyA(const Variable* in, Variable* out)
{
	auto start = std::chrono::high_resolution_clock::now();
	const Image* img_in = dynamic_cast<const Image*>(in);
	Image* img_out = dynamic_cast<Image*>(out);
	ASSERT_MSG(img_in != nullptr && img_out != nullptr,
	           "Input parameters must be images");
	varconvolve<true>(img_in, img_out);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << "Var PSF execution time: " << duration.count() << " seconds"
	          << std::endl;
}

void OperatorVarPsf::applyAH(const Variable* in, Variable* out)
{
	auto start = std::chrono::high_resolution_clock::now();
	const Image* img_in = dynamic_cast<const Image*>(in);
	Image* img_out = dynamic_cast<Image*>(out);
	ASSERT_MSG(img_in != nullptr && img_out != nullptr,
	           "Input parameters must be images");
	varconvolve<false>(img_in, img_out);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << "Var Transposed PSF execution time: " << duration.count()
	          << " seconds" << std::endl;
}

template <bool IS_FWD>
void OperatorVarPsf::varconvolve(const Image* in, Image* out) const
{
	const ImageParams& params = in->getParams();
	ASSERT_MSG(params.isSameDimensionsAs(out->getParams()),
	           "Dimensions mismatch between the two images");
	ASSERT_MSG(m_kernelLUT.size() > 0, "LUT not defined");
	const float* inPtr = in->getRawPointer();
	float* outPtr = out->getRawPointer();
	const int nx = params.nx;
	const int ny = params.ny;
	const int nz = params.nz;
	const float vx = params.vx;
	const float vy = params.vy;
	const float vz = params.vz;
	float x_center = nx * vx / 2.0f;
	float y_center = ny * vy / 2.0f;
	float z_center = nz * vz / 2.0f;
	float temp_x, temp_y, temp_z;
	int i, j, k, ii, jj, kk;

#pragma omp parallel for private(temp_x, temp_y, temp_z, i, j, k, ii, jj, kk) \
    shared(outPtr)
	for (int pp = 0; pp < nx * ny * nz; pp++)
	{
		i = pp % nx;
		j = (pp / nx) % ny;
		k = pp / (nx * ny);
		temp_x = std::abs((i + 0.5) * vx - x_center);
		temp_y = std::abs((j + 0.5) * vy - y_center);
		temp_z = std::abs((k + 0.5) * vz - z_center);
		auto& kernel = findNearestKernel(temp_x, temp_y, temp_z);

		int kernel_size_x = kernel.getHalfSizeX();
		int kernel_size_y = kernel.getHalfSizeY();
		int kernel_size_z = kernel.getHalfSizeZ();

		auto& psf_kernel = kernel.getArray();
		int idx = 0;
		float temp1 = inPtr[IDX3(i, j, k, nx, ny)];

		for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z; ++z_diff)
		{
			for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; ++y_diff)
			{
				for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x;
				     ++x_diff, ++idx)
				{
					ii = util::circular(nx, i + x_diff);
					jj = util::circular(ny, j + y_diff);
					kk = util::circular(nz, k + z_diff);

					if constexpr (IS_FWD)
					{
						outPtr[IDX3(i, j, k, nx, ny)] +=
						    inPtr[IDX3(ii, jj, kk, nx, ny)] *
						    psf_kernel.getFlat(idx);
					}
					else
					{
#pragma omp atomic
						outPtr[IDX3(ii, jj, kk, nx, ny)] +=
						    temp1 * psf_kernel.getFlat(idx);
					}
				}
			}
		}
	}
}

void OperatorVarPsf::setRangeAndGap(float xRange, float xGap, float yRange,
                                    float yGap, float zRange, float zGap)
{
	m_xRange = xRange;
	m_xGap = xGap;
	m_yRange = yRange;
	m_yGap = yGap;
	m_zRange = zRange;
	m_zGap = zGap;
}

}  // namespace yrt
