/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorPsf.hpp"

#include "utils/Assert.hpp"
#include "utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void py_setup_operatorpsf(py::module& m)
{
	auto c = py::class_<OperatorPsf, Operator>(m, "OperatorPsf");
	c.def(py::init<const ImageParams&>());
	c.def(py::init<const ImageParams&, const std::string&>());
	c.def("readFromFile", &OperatorPsf::readFromFile);
	c.def("convolve", &OperatorPsf::convolve);
	c.def(
	    "applyA",
	    [](OperatorPsf& self, const Image* img_in, Image* img_out)
	    { self.applyA(img_in, img_out); },
	    py::arg("img_in"), py::arg("img_out"));
	c.def(
	    "applyAH",
	    [](OperatorPsf& self, const Image* img_in, Image* img_out)
	    { self.applyAH(img_in, img_out); },
	    py::arg("img_in"), py::arg("img_out"));
}
#endif

OperatorPsf::OperatorPsf(const ImageParams& img_params)
    : Operator(), m_params(img_params)
{
	m_nx = m_params.nx;
	m_ny = m_params.ny;
	m_nz = m_params.nz;

	const size_t sizeBuffer = std::max(std::max(m_nx, m_ny), m_nz);
	m_buffer_tmp.resize(sizeBuffer);
}

OperatorPsf::OperatorPsf(const ImageParams& img_params,
                         const std::string& image_space_psf_filename)
    : OperatorPsf(img_params)
{
	readFromFile(image_space_psf_filename);
}

void OperatorPsf::readFromFile(const std::string& image_space_psf_filename)
{
	Array2D<float> kernelsArray2D;
	std::cout << "Reading image space PSF kernel csv file" << std::endl;
	Util::readCSV<float>(image_space_psf_filename, kernelsArray2D);
	std::cout << "Done reading image space PSF kernel csv file" << std::endl;

	// X
	{
		m_kerSize.push_back(kernelsArray2D[3][0]);
		for (int i = 0; i < m_kerSize[0]; i++)
		{
			m_KernelX.push_back(kernelsArray2D[0][i]);
			m_KernelX_flipped.push_back(
			    kernelsArray2D[0][m_kerSize[0] - 1 - i]);
		}
	}
	// Y
	{
		m_kerSize.push_back(kernelsArray2D[3][1]);
		for (int i = 0; i < m_kerSize[1]; i++)
		{
			m_KernelY.push_back(kernelsArray2D[1][i]);
			m_KernelY_flipped.push_back(
			    kernelsArray2D[1][m_kerSize[1] - 1 - i]);
		}
	}

	// Z
	{
		m_kerSize.push_back(kernelsArray2D[3][2]);
		for (int i = 0; i < m_kerSize[2]; i++)
		{
			m_KernelZ.push_back(kernelsArray2D[2][i]);
			m_KernelZ_flipped.push_back(
			    kernelsArray2D[2][m_kerSize[2] - 1 - i]);
		}
	}
}


OperatorPsf::~OperatorPsf() {}

void OperatorPsf::applyA(const Variable* in, Variable* out)
{
	const Image* img_in = dynamic_cast<const Image*>(in);
	Image* img_out = dynamic_cast<Image*>(out);
	ASSERT(img_in != nullptr && img_out != nullptr);
	convolve(img_in, img_out, m_KernelX, m_KernelY, m_KernelZ);
}

void OperatorPsf::applyAH(const Variable* in, Variable* out)
{
	const Image* img_in = dynamic_cast<const Image*>(in);
	Image* img_out = dynamic_cast<Image*>(out);

	convolve(img_in, img_out, m_KernelX_flipped, m_KernelY_flipped,
	         m_KernelZ_flipped);
}

void OperatorPsf::convolve(const Image* in, Image* out,
                           const std::vector<float>& KernelX,
                           const std::vector<float>& KernelY,
                           const std::vector<float>& KernelZ) const
{
	// kernel size must always be an odd number and must have same size in all 3
	// dimensions
	int kerIndexCenteredX = m_kerSize[0] / 2;
	int kerIndexCenteredY = m_kerSize[1] / 2;
	int kerIndexCenteredZ = m_kerSize[2] / 2;
	const double* inPtr = in->getData().getRawPointer();
	double* outPtr = out->getData().getRawPointer();

	for (int k = 0; k < m_nz; k++)
	{
		for (int j = 0; j < m_ny; j++)
		{
			for (int i = 0; i < m_nx; i++)
			{
				m_buffer_tmp[i] = inPtr[IDX3(i, j, k, m_nx, m_ny)];
			}
			for (int i = 0; i < m_nx; i++)
			{
				float sum = 0.0;
				for (int kk = -kerIndexCenteredX; kk <= kerIndexCenteredX; kk++)
				{
					int r = Util::circular(m_nx, i - kk);
					sum += KernelX[kk + kerIndexCenteredX] * m_buffer_tmp[r];
				}
				outPtr[IDX3(i, j, k, m_nx, m_ny)] = sum;
			}
		}
	}

	for (int k = 0; k < m_nz; k++)
	{
		for (int i = 0; i < m_nx; i++)
		{
			for (int j = 0; j < m_ny; j++)
			{
				m_buffer_tmp[j] = outPtr[IDX3(i, j, k, m_nx, m_ny)];
			}
			for (int j = 0; j < m_ny; j++)
			{
				float sum = 0.0;
				for (int kk = -kerIndexCenteredY; kk <= kerIndexCenteredY; kk++)
				{
					int r = Util::circular(m_ny, j - kk);
					sum += KernelY[kk + kerIndexCenteredY] * m_buffer_tmp[r];
				}
				outPtr[IDX3(i, j, k, m_nx, m_ny)] = sum;
			}
		}
	}

	for (int i = 0; i < m_nx; i++)
	{
		for (int j = 0; j < m_ny; j++)
		{
			for (int k = 0; k < m_nz; k++)
			{
				m_buffer_tmp[k] = outPtr[IDX3(i, j, k, m_nx, m_ny)];
			}
			for (int k = 0; k < m_nz; k++)
			{
				float sum = 0.0;
				for (int kk = -kerIndexCenteredZ; kk <= kerIndexCenteredZ; kk++)
				{
					int r = Util::circular(m_nz, k - kk);
					sum += KernelZ[kk + kerIndexCenteredZ] * m_buffer_tmp[r];
				}
				outPtr[IDX3(i, j, k, m_nx, m_ny)] = sum;
			}
		}
	}
}
