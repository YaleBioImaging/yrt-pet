/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorVarPsf.hpp"

#include "utils/Assert.hpp"
#include "utils/Tools.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void py_setup_operatorvarpsf(py::module& m)
{
	auto c = py::class_<OperatorVarPsf, Operator>(m, "OperatorVarPsf");
	c.def(py::init<>());
	c.def(py::init<const std::string&>());
	c.def("readFromFile", &OperatorVarPsf::readFromFile,
	      "Read the variant PSF from CSV LUT");
	// c.def("varconvolve", static_cast<void (OperatorVarPsf::*)(const Image*,
	// Image*) const>(&OperatorVarPsf::varconvolve<true>));
	// c.def("varconvolve_transposed", static_cast<void
	// (OperatorVarPsf::*)(const Image*, Image*)
	// const>(&OperatorVarPsf::varconvolve<false>));
	c.def(
	    "applyA",
	    [](OperatorVarPsf& self, const Image* img_in, Image* img_out)
	    { self.applyA(img_in, img_out); },
	    py::arg("img_in"), py::arg("img_out"));
	c.def(
	    "applyAH",
	    [](OperatorVarPsf& self, const Image* img_in, Image* img_out)
	    { self.applyAH(img_in, img_out); },
	    py::arg("img_in"), py::arg("img_out"));
}
#endif

OperatorVarPsf::OperatorVarPsf() : Operator{} {}

OperatorVarPsf::OperatorVarPsf(const std::string& imageVarPsf_fname)
    : OperatorVarPsf{}
{
	readFromFileInternal(imageVarPsf_fname);
}

void OperatorVarPsf::readFromFile(const std::string& imageVarPsf_fname)
{
	readFromFileInternal(imageVarPsf_fname);
}
Sigma OperatorVarPsf::find_nearest_sigma(const std::vector<Sigma>& sigma_lookup,
                                         float x, float y, float z) const
{
	int x_dim = static_cast<int>(std::floor(x_range / x_gap)) + 1;
	int y_dim = static_cast<int>(std::floor(y_range / y_gap)) + 1;
	int z_dim = static_cast<int>(std::floor(z_range / z_gap)) + 1;

	int i = static_cast<int>(std::round(abs(x - 5) / x_gap));
	int j = static_cast<int>(std::round(abs(y - 5) / y_gap));
	int k = static_cast<int>(std::round(abs(z - 5) / z_gap));

	if (i >= x_dim)
		i = x_dim - 1;
	if (j >= y_dim)
		j = y_dim - 1;
	if (k >= z_dim)
		k = z_dim - 1;

	int index = IDX3(i, j, k, x_dim, y_dim);
	Sigma nearest_sigma = sigma_lookup[index];

	return nearest_sigma;
}
void OperatorVarPsf::readFromFileInternal(const std::string& imageVarPsf_fname)
{
	std::cout << "Reading image space Variant PSF sigma lookup table file..."
	          << std::endl;
	Array2D<float> data;
	Util::readCSV<float>(imageVarPsf_fname, data);

	size_t dims[2];
	data.getDims(dims);

	for (size_t i = 0; i < dims[0]; ++i)
	{
		if (dims[1] < 6)
		{
			throw std::runtime_error(
			    "CSV file format error: Not enough columns");
		}
		Sigma s;
		s.x = data[i][0];
		s.y = data[i][1];
		s.z = data[i][2];
		s.sigmax = data[i][3];
		s.sigmay = data[i][4];
		s.sigmaz = data[i][5];
		sigma_lookup.push_back(s);
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
	ASSERT_MSG(sigma_lookup.size() > 0, "LUT not defined");
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

	const size_t sizeBuffer = std::max(std::max(nx, ny), nz);
	m_buffer_tmp.resize(sizeBuffer);

	float xoffset, yoffset, zoffset, temp_x, temp_y, temp_z;
	int ii, jj, kk;
	int i, j, k;

#pragma omp parallel for private(temp_x, temp_y, temp_z, i, j, k, ii, jj, kk, \
                                     xoffset, yoffset, zoffset) shared(outPtr)
	for (int pp = 0; pp < nx * ny * nz; pp++)
	{
		i = pp % nx;
		j = (pp / nx) % ny;
		k = pp / (nx * ny);
		temp_x = std::abs((i + 0.5) * vx - x_center);
		temp_y = std::abs((j + 0.5) * vy - y_center);
		temp_z = std::abs((k + 0.5) * vz - z_center);
		Sigma s = find_nearest_sigma(sigma_lookup, temp_x, temp_y, temp_z);
		int kernel_size_x = std::min(
		    4, std::max(1, static_cast<int>(std::floor(
		                       (s.sigmax * kernel_width_control) / vx)) -
		                       1));
		int kernel_size_y = std::min(
		    4, std::max(1, static_cast<int>(std::floor(
		                       (s.sigmay * kernel_width_control) / vy)) -
		                       1));
		int kernel_size_z = std::min(
		    4, std::max(1, static_cast<int>(std::floor(
		                       (s.sigmaz * kernel_width_control) / vz)) -
		                       1));

		const int kx_len = kernel_size_x * 2 + 1;
		const int ky_len = kernel_size_y * 2 + 1;
		const int kz_len = kernel_size_z * 2 + 1;
		std::vector<float> psf_kernel(kx_len * ky_len * kz_len, 0.0f);
		float inv_2_sigmax2 = 1.0f / (2 * s.sigmax * s.sigmax);
		float inv_2_sigmay2 = 1.0f / (2 * s.sigmay * s.sigmay);
		float inv_2_sigmaz2 = 1.0f / (2 * s.sigmaz * s.sigmaz);
		float kernel_sum = 0.0f;
		int idx = 0;

		for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x; ++x_diff)
			for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; ++y_diff)
				for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z;
				     ++z_diff, ++idx)
				{
					xoffset = x_diff * vx;
					yoffset = y_diff * vy;
					zoffset = z_diff * vz;
					float temp = -(xoffset * xoffset * inv_2_sigmax2 +
					               yoffset * yoffset * inv_2_sigmay2 +
					               zoffset * zoffset * inv_2_sigmaz2);
					psf_kernel[idx] = exp(temp);
					kernel_sum += psf_kernel[idx];
				}
		for (auto& val : psf_kernel)
			val /= kernel_sum;

		float temp1 = inPtr[IDX3(i, j, k, nx, ny)];
		idx = 0;
		for (int x_diff = -kernel_size_x; x_diff <= kernel_size_x; ++x_diff)
			for (int y_diff = -kernel_size_y; y_diff <= kernel_size_y; ++y_diff)
				for (int z_diff = -kernel_size_z; z_diff <= kernel_size_z;
				     ++z_diff, ++idx)
				{
					ii = Util::circular(nx, i + x_diff);
					jj = Util::circular(ny, j + y_diff);
					kk = Util::circular(nz, k + z_diff);

					if constexpr (IS_FWD)
					{
#pragma omp atomic
						outPtr[IDX3(i, j, k, nx, ny)] +=
						    inPtr[IDX3(ii, jj, kk, nx, ny)] * psf_kernel[idx];
					}
					else
					{
#pragma omp atomic
						outPtr[IDX3(ii, jj, kk, nx, ny)] +=
						    temp1 * psf_kernel[idx];
					}
				}
	}
}
