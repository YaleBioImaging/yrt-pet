/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorPsf.hpp"

#include "utils/Assert.hpp"
#include "utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void py_setup_operatorpsf(py::module& m)
{
	auto c = py::class_<OperatorPsf, Operator>(m, "OperatorPsf");
	c.def(py::init<>());
	c.def(py::init<const std::string&>());
	c.def("readFromFile", &OperatorPsf::readFromFile);
	c.def("convolve", &OperatorPsf::convolve);
	c.def(
	    "applyA", [](OperatorPsf& self, const Image* img_in, Image* img_out)
	    { self.applyA(img_in, img_out); }, py::arg("img_in"),
	    py::arg("img_out"));
	c.def(
	    "applyAH", [](OperatorPsf& self, const Image* img_in, Image* img_out)
	    { self.applyAH(img_in, img_out); }, py::arg("img_in"),
	    py::arg("img_out"));

	c.def("getKernelX",
	      [](const OperatorPsf& self) -> py::array_t<float>
	      {
		      auto arr = self.getKernelX();
		      const auto buf_info =
		          py::buffer_info(arr.data(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {arr.size()}, {sizeof(float)});
		      return pybind11::array_t<float>(buf_info);
	      });
	c.def("getKernelY",
	      [](const OperatorPsf& self) -> py::array_t<float>
	      {
		      auto arr = self.getKernelY();
		      const auto buf_info =
		          py::buffer_info(arr.data(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {arr.size()}, {sizeof(float)});
		      return pybind11::array_t<float>(buf_info);
	      });
	c.def("getKernelZ",
	      [](const OperatorPsf& self) -> py::array_t<float>
	      {
		      auto arr = self.getKernelZ();
		      const auto buf_info =
		          py::buffer_info(arr.data(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {arr.size()}, {sizeof(float)});
		      return pybind11::array_t<float>(buf_info);
	      });
	c.def("getKernelXFlipped",
	      [](const OperatorPsf& self) -> py::array_t<float>
	      {
		      auto arr = self.getKernelXFlipped();
		      const auto buf_info =
		          py::buffer_info(arr.data(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {arr.size()}, {sizeof(float)});
		      return pybind11::array_t<float>(buf_info);
	      });
	c.def("getKernelYFlipped",
	      [](const OperatorPsf& self) -> py::array_t<float>
	      {
		      auto arr = self.getKernelYFlipped();
		      const auto buf_info =
		          py::buffer_info(arr.data(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {arr.size()}, {sizeof(float)});
		      return pybind11::array_t<float>(buf_info);
	      });
	c.def("getKernelZFlipped",
	      [](const OperatorPsf& self) -> py::array_t<float>
	      {
		      auto arr = self.getKernelZFlipped();
		      const auto buf_info =
		          py::buffer_info(arr.data(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {arr.size()}, {sizeof(float)});
		      return pybind11::array_t<float>(buf_info);
	      });
}
#endif

OperatorPsf::OperatorPsf() : Operator{} {}

OperatorPsf::OperatorPsf(const std::string& imagePsf_fname) : OperatorPsf{}
{
	readFromFileInternal(imagePsf_fname);
}

void OperatorPsf::readFromFile(const std::string& imagePsf_fname)
{
	readFromFileInternal(imagePsf_fname);
}

void OperatorPsf::readFromFileInternal(const std::string& imagePsf_fname)
{
	Array2D<float> kernelsArray2D;
	std::cout << "Reading image space PSF kernel csv file..." << std::endl;
	Util::readCSV<float>(imagePsf_fname, kernelsArray2D);

	std::array<int, 3> kerSize;
	kerSize[0] = kernelsArray2D[3][0];
	kerSize[1] = kernelsArray2D[3][1];
	kerSize[2] = kernelsArray2D[3][2];
	ASSERT_MSG(kerSize[0] % 2 != 0, "Kernel size must be odd");
	ASSERT_MSG(kerSize[1] % 2 != 0, "Kernel size must be odd");
	ASSERT_MSG(kerSize[2] % 2 != 0, "Kernel size must be odd");

	// X
	{
		m_kernelX.reserve(kerSize[0]);
		m_kernelX_flipped.reserve(kerSize[0]);
		for (int i = 0; i < kerSize[0]; i++)
		{
			m_kernelX.push_back(kernelsArray2D[0][i]);
			m_kernelX_flipped.push_back(kernelsArray2D[0][kerSize[0] - 1 - i]);
		}
	}
	// Y
	{
		m_kernelY.reserve(kerSize[1]);
		m_kernelY_flipped.reserve(kerSize[1]);
		for (int i = 0; i < kerSize[1]; i++)
		{
			m_kernelY.push_back(kernelsArray2D[1][i]);
			m_kernelY_flipped.push_back(kernelsArray2D[1][kerSize[1] - 1 - i]);
		}
	}

	// Z
	{
		m_kernelZ.reserve(kerSize[2]);
		m_kernelZ_flipped.reserve(kerSize[2]);
		for (int i = 0; i < kerSize[2]; i++)
		{
			m_kernelZ.push_back(kernelsArray2D[2][i]);
			m_kernelZ_flipped.push_back(kernelsArray2D[2][kerSize[2] - 1 - i]);
		}
	}
}

void OperatorPsf::applyA(const Variable* in, Variable* out)
{
	const Image* img_in = dynamic_cast<const Image*>(in);
	Image* img_out = dynamic_cast<Image*>(out);
	ASSERT_MSG(img_in != nullptr && img_out != nullptr,
	           "Input parameters must be images");

	convolve(img_in, img_out, m_kernelX, m_kernelY, m_kernelZ);
}

void OperatorPsf::applyAH(const Variable* in, Variable* out)
{
	const Image* img_in = dynamic_cast<const Image*>(in);
	Image* img_out = dynamic_cast<Image*>(out);
	ASSERT_MSG(img_in != nullptr && img_out != nullptr,
	           "Input parameters must be images");

	convolve(img_in, img_out, m_kernelX_flipped, m_kernelY_flipped,
	         m_kernelZ_flipped);
}

void OperatorPsf::convolve(const Image* in, Image* out,
                           const std::vector<float>& kernelX,
                           const std::vector<float>& kernelY,
                           const std::vector<float>& kernelZ) const
{
	const ImageParams& params = in->getParams();
	ASSERT_MSG(params.isSameDimensionsAs(out->getParams()),
	           "Dimensions mismatch between the two images");
	const int nx = params.nx;
	const int ny = params.ny;
	const int nz = params.nz;

	const size_t sizeBuffer = std::max(std::max(nx, ny), nz);
	m_buffer_tmp.resize(sizeBuffer);

	const std::array<int, 3> kerSize{static_cast<int>(kernelX.size()),
	                                 static_cast<int>(kernelY.size()),
	                                 static_cast<int>(kernelZ.size())};
	ASSERT_MSG(kerSize[0] % 2 != 0, "Kernel size must be odd");
	ASSERT_MSG(kerSize[1] % 2 != 0, "Kernel size must be odd");
	ASSERT_MSG(kerSize[2] % 2 != 0, "Kernel size must be odd");

	// kernel size must always be an odd number and must have same size in all 3
	// dimensions
	const int kerIndexCenteredX = kerSize[0] / 2;
	const int kerIndexCenteredY = kerSize[1] / 2;
	const int kerIndexCenteredZ = kerSize[2] / 2;
	const float* inPtr = in->getRawPointer();
	float* outPtr = out->getRawPointer();

	for (int k = 0; k < nz; k++)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				m_buffer_tmp[i] = inPtr[IDX3(i, j, k, nx, ny)];
			}
			for (int i = 0; i < nx; i++)
			{
				float sum = 0.0;
				for (int kk = -kerIndexCenteredX; kk <= kerIndexCenteredX; kk++)
				{
					const int r = Util::circular(nx, i - kk);
					sum += kernelX[kk + kerIndexCenteredX] * m_buffer_tmp[r];
				}
				outPtr[IDX3(i, j, k, nx, ny)] = sum;
			}
		}
	}

	for (int k = 0; k < nz; k++)
	{
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				m_buffer_tmp[j] = outPtr[IDX3(i, j, k, nx, ny)];
			}
			for (int j = 0; j < ny; j++)
			{
				float sum = 0.0;
				for (int kk = -kerIndexCenteredY; kk <= kerIndexCenteredY; kk++)
				{
					const int r = Util::circular(ny, j - kk);
					sum += kernelY[kk + kerIndexCenteredY] * m_buffer_tmp[r];
				}
				outPtr[IDX3(i, j, k, nx, ny)] = sum;
			}
		}
	}

	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int k = 0; k < nz; k++)
			{
				m_buffer_tmp[k] = outPtr[IDX3(i, j, k, nx, ny)];
			}
			for (int k = 0; k < nz; k++)
			{
				float sum = 0.0;
				for (int kk = -kerIndexCenteredZ; kk <= kerIndexCenteredZ; kk++)
				{
					const int r = Util::circular(nz, k - kk);
					sum += kernelZ[kk + kerIndexCenteredZ] * m_buffer_tmp[r];
				}
				outPtr[IDX3(i, j, k, nx, ny)] = sum;
			}
		}
	}
}

const std::vector<float>& OperatorPsf::getKernelX() const
{
	return m_kernelX;
}

const std::vector<float>& OperatorPsf::getKernelY() const
{
	return m_kernelX;
}

const std::vector<float>& OperatorPsf::getKernelZ() const
{
	return m_kernelZ;
}

const std::vector<float>& OperatorPsf::getKernelXFlipped() const
{
	return m_kernelX_flipped;
}

const std::vector<float>& OperatorPsf::getKernelYFlipped() const
{
	return m_kernelY_flipped;
}

const std::vector<float>& OperatorPsf::getKernelZFlipped() const
{
	return m_kernelZ_flipped;
}
