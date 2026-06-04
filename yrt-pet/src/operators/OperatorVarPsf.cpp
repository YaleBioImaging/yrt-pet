/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorVarPsf.hpp"

#include "yrt-pet/datastruct/image/ImageBase.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Tools.hpp"
#include <algorithm>
#include <memory>

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
	c.def(py::init<const std::string&, const ImageParams&, bool>(),
	      py::arg("fname"), py::arg("image_params"),
	      py::arg("use_two_gaussian") = false);
	c.def("readFromFile", &OperatorVarPsf::readFromFile, py::arg("fname"),
	      py::arg("use_two_gaussian") = false,
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

ConvolutionKernelGaussianMixture::ConvolutionKernelGaussianMixture(
    float p_sigmaX1, float p_sigmaY1, float p_sigmaZ1, float p_sigmaX2,
    float p_sigmaY2, float p_sigmaZ2, float p_weight2, float p_nStdX,
    float p_nStdY, float p_nStdZ, const ImageParams& pr_imageParams)
    : ConvolutionKernel()
{
	setSigmas(p_sigmaX1, p_sigmaY1, p_sigmaZ1, p_sigmaX2, p_sigmaY2,
	          p_sigmaZ2, p_weight2, p_nStdX, p_nStdY, p_nStdZ,
	          pr_imageParams);
}

void ConvolutionKernelGaussianMixture::setSigmas(
    float p_sigmaX1, float p_sigmaY1, float p_sigmaZ1, float p_sigmaX2,
    float p_sigmaY2, float p_sigmaZ2, float p_weight2, float p_nStdX,
    float p_nStdY, float p_nStdZ, const ImageParams& pr_imageParams)
{
	ASSERT_MSG(p_sigmaX1 > 0.0f && p_sigmaY1 > 0.0f && p_sigmaZ1 > 0.0f,
	           "First Gaussian sigmas must be positive");
	ASSERT_MSG(p_sigmaX2 > 0.0f && p_sigmaY2 > 0.0f && p_sigmaZ2 > 0.0f,
	           "Second Gaussian sigmas must be positive");
	ASSERT_MSG(p_weight2 >= 0.0f && p_weight2 <= 1.0f,
	           "Second Gaussian weight must be in [0, 1]");

	m_sigmaX1 = p_sigmaX1;
	m_sigmaY1 = p_sigmaY1;
	m_sigmaZ1 = p_sigmaZ1;
	m_sigmaX2 = p_sigmaX2;
	m_sigmaY2 = p_sigmaY2;
	m_sigmaZ2 = p_sigmaZ2;
	m_weight2 = p_weight2;
	m_nStdX = p_nStdX;
	m_nStdY = p_nStdY;
	m_nStdZ = p_nStdZ;

	const float sigmaXMax = std::max(m_sigmaX1, m_sigmaX2);
	const float sigmaYMax = std::max(m_sigmaY1, m_sigmaY2);
	const float sigmaZMax = std::max(m_sigmaZ1, m_sigmaZ2);

	int kernel_size_x = std::max(
	    1,
	    static_cast<int>(std::floor((sigmaXMax * m_nStdX) / pr_imageParams.vx)) -
	        1);
	int kernel_size_y = std::max(
	    1,
	    static_cast<int>(std::floor((sigmaYMax * m_nStdY) / pr_imageParams.vy)) -
	        1);
	int kernel_size_z = std::max(
	    1,
	    static_cast<int>(std::floor((sigmaZMax * m_nStdZ) / pr_imageParams.vz)) -
	        1);

	const int kx_len = kernel_size_x * 2 + 1;
	const int ky_len = kernel_size_y * 2 + 1;
	const int kz_len = kernel_size_z * 2 + 1;
	psfKernel.allocate(kz_len, ky_len, kx_len);

	const float weight1 = 1.0f - m_weight2;
	const float inv_2_sigmax1_2 = 1.0f / (2 * m_sigmaX1 * m_sigmaX1);
	const float inv_2_sigmay1_2 = 1.0f / (2 * m_sigmaY1 * m_sigmaY1);
	const float inv_2_sigmaz1_2 = 1.0f / (2 * m_sigmaZ1 * m_sigmaZ1);
	const float inv_2_sigmax2_2 = 1.0f / (2 * m_sigmaX2 * m_sigmaX2);
	const float inv_2_sigmay2_2 = 1.0f / (2 * m_sigmaY2 * m_sigmaY2);
	const float inv_2_sigmaz2_2 = 1.0f / (2 * m_sigmaZ2 * m_sigmaZ2);
	float kernel1_sum = 0.0f;
	float kernel2_sum = 0.0f;
	int idx = 0;

	for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z; ++z_diff)
	{
		for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; ++y_diff)
		{
			for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x;
			     ++x_diff, ++idx)
			{
				const float xoffset = x_diff * pr_imageParams.vx;
				const float yoffset = y_diff * pr_imageParams.vy;
				const float zoffset = z_diff * pr_imageParams.vz;
				const float exp1 =
				    -(xoffset * xoffset * inv_2_sigmax1_2 +
				      yoffset * yoffset * inv_2_sigmay1_2 +
				      zoffset * zoffset * inv_2_sigmaz1_2);
				const float exp2 =
				    -(xoffset * xoffset * inv_2_sigmax2_2 +
				      yoffset * yoffset * inv_2_sigmay2_2 +
				      zoffset * zoffset * inv_2_sigmaz2_2);
				const float value1 = std::exp(exp1);
				const float value2 = std::exp(exp2);
				psfKernel.setFlat(idx, value1);
				kernel1_sum += value1;
				kernel2_sum += value2;
			}
		}
	}
	ASSERT_MSG(kernel1_sum > 0.0f && kernel2_sum > 0.0f,
	           "2-Gaussian kernel sums must be positive");

	idx = 0;
	for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z; ++z_diff)
	{
		for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; ++y_diff)
		{
			for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x;
			     ++x_diff, ++idx)
			{
				const float xoffset = x_diff * pr_imageParams.vx;
				const float yoffset = y_diff * pr_imageParams.vy;
				const float zoffset = z_diff * pr_imageParams.vz;
				const float exp2 =
				    -(xoffset * xoffset * inv_2_sigmax2_2 +
				      yoffset * yoffset * inv_2_sigmay2_2 +
				      zoffset * zoffset * inv_2_sigmaz2_2);
				const float value =
				    weight1 * psfKernel.getFlat(idx) / kernel1_sum +
				    m_weight2 * std::exp(exp2) / kernel2_sum;
				psfKernel.setFlat(idx, value);
			}
		}
	}

	for (size_t i = 0; i < psfKernel.getSizeTotal(); ++i)
	{
		ASSERT_MSG(psfKernel.getFlat(i) >= 0.0f,
		           "2-Gaussian kernel value cannot be negative");
	}
}

OperatorVarPsf::OperatorVarPsf(const ImageParams& pr_imageParams)
    : Operator{}, m_imageParams(pr_imageParams)
{
}

OperatorVarPsf::OperatorVarPsf(const std::string& imageVarPsf_fname,
                               const ImageParams& pr_imageParams,
                               bool p_useTwoGaussian)
    : OperatorVarPsf{pr_imageParams}
{
	readFromFile(imageVarPsf_fname, p_useTwoGaussian);
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

void OperatorVarPsf::readFromFile(const std::string& imageVarPsf_fname,
                                  bool p_useTwoGaussian)
{
	Array2DOwned<float> data;
	util::readCSV<float>(imageVarPsf_fname, data);
	const std::array<size_t, 2> dims = data.getDims();

	ASSERT_MSG(dims[0] > 3, "CSV file format error: At least 4 rows expected");
	if (p_useTwoGaussian)
	{
		ASSERT_MSG(dims[1] == 7,
		           "CSV file format error: 7 columns expected for 2-Gaussian "
		           "VarPSF");
	}
	else
	{
		ASSERT_MSG(dims[1] == 3, "CSV file format error: 3 columns expected");
	}

	m_xRange = data[0][0];
	m_yRange = data[0][1];
	m_zRange = data[0][2];
	m_xGap = data[1][0];
	m_yGap = data[1][1];
	m_zGap = data[1][2];
	float nStdX = data[2][0];
	float nStdY = data[2][1];
	float nStdZ = data[2][2];

	m_kernelLUT.clear();
	for (size_t i = 3; i < dims[0]; ++i)
	{
		std::unique_ptr<ConvolutionKernel> kernel;
		if (p_useTwoGaussian)
		{
			kernel = std::make_unique<ConvolutionKernelGaussianMixture>(
			    data[i][0], data[i][1], data[i][2], data[i][3], data[i][4],
			    data[i][5], data[i][6], nStdX, nStdY, nStdZ, m_imageParams);
		}
		else
		{
			kernel = std::make_unique<ConvolutionKernelGaussian>(
			    data[i][0], data[i][1], data[i][2], nStdX, nStdY, nStdZ,
			    m_imageParams);
		}
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
	const Image* img_in = dynamic_cast<const Image*>(in);
	Image* img_out = dynamic_cast<Image*>(out);
	ASSERT_MSG(img_in != nullptr && img_out != nullptr,
	           "Input parameters must be images");
	varconvolve<true>(img_in, img_out);
}

void OperatorVarPsf::applyAH(const Variable* in, Variable* out)
{
	const Image* img_in = dynamic_cast<const Image*>(in);
	Image* img_out = dynamic_cast<Image*>(out);
	ASSERT_MSG(img_in != nullptr && img_out != nullptr,
	           "Input parameters must be images");
	varconvolve<false>(img_in, img_out);
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
	const ssize_t nx = params.nx;
	const ssize_t ny = params.ny;
	const ssize_t nz = params.nz;
	const float vx = params.vx;
	const float vy = params.vy;
	const float vz = params.vz;
	float x_center = nx * vx / 2.0f;
	float y_center = ny * vy / 2.0f;
	float z_center = nz * vz / 2.0f;

	util::parallelForChunked(
	    nx * ny * nz, globals::getNumThreads(),
	    [nx, ny, nz, vx, vy, vz, x_center, y_center, z_center, inPtr, outPtr,
	     this](size_t pp, size_t /*tid*/)
	    {
		    const ssize_t k = pp / (nx * ny);
		    const ssize_t i = pp % nx;
		    const ssize_t j = (pp / nx) % ny;
		    const float temp_x = std::abs((i + 0.5) * vx - x_center);
		    const float temp_y = std::abs((j + 0.5) * vy - y_center);
		    const float temp_z = std::abs((k + 0.5) * vz - z_center);
		    auto& kernel = findNearestKernel(temp_x, temp_y, temp_z);

		    const ssize_t kernelSize_x = kernel.getHalfSizeX();
		    const ssize_t kernelSize_y = kernel.getHalfSizeY();
		    const ssize_t kernelSize_z = kernel.getHalfSizeZ();

		    auto& psf_kernel = kernel.getArray();
		    ssize_t idx = 0;
		    const float temp1 = inPtr[IDX3(i, j, k, nx, ny)];

		    for (ssize_t z_diff = -kernelSize_z; z_diff <= kernelSize_z; ++z_diff)
		    {
			    for (ssize_t y_diff = -kernelSize_y; y_diff <= kernelSize_y;
			         ++y_diff)
			    {
				    for (ssize_t x_diff = -kernelSize_x; x_diff <= kernelSize_x;
				         ++x_diff, ++idx)
				    {
					    const ssize_t ii = util::circular(nx, i + x_diff);
					    const ssize_t jj = util::circular(ny, j + y_diff);
					    const ssize_t kk = util::circular(nz, k + z_diff);

					    if constexpr (IS_FWD)
					    {
						    outPtr[IDX3(i, j, k, nx, ny)] +=
						        inPtr[IDX3(ii, jj, kk, nx, ny)] *
						        psf_kernel.getFlat(idx);
					    }
					    else
					    {
						    std::atomic_ref<float> atomic_elem(
						        outPtr[IDX3(ii, jj, kk, nx, ny)]);
						    atomic_elem.fetch_add(temp1 *
						                          psf_kernel.getFlat(idx));
					    }
				    }
			    }
		    }
	    });
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
