/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Tools.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace yrt
{
namespace util
{

template <typename T>
void readCSV(const std::string& filename, Array2D<T>& output)
{
	std::ifstream file(filename);
	std::string line = "";
	size_t numLines = 0;
	size_t numCols = 0;
	T val;
	// Iterate through each line and split the content using delimiter
	while (std::getline(file, line))
	{
		size_t numColsLine = 0;
		// Count columns on first line
		std::stringstream ss(line);
		// Extract each integer
		while (ss >> val)
		{
			// Store value
			numColsLine++;
			// If the next token is a comma, ignore it and move on
			if (ss.peek() == ',')
			{
				ss.ignore();
			}
		}
		if (numColsLine > numCols)
		{
			numCols = numColsLine;
		}
		++numLines;
	}
	output.allocate(numLines, numCols);
	if (numLines * numCols == 0)
	{
		throw std::runtime_error("The CSV file seems empty");
	}

	file.clear();
	file.seekg(0, file.beg);

	size_t tidx = 0;
	while (std::getline(file, line))
	{
		// Create a stringstream of the current line
		std::stringstream ss(line);

		// Keep track of the current column index
		int colIdx = 0;

		// Extract each integer
		while (ss >> val)
		{
			// Store value
			output[tidx][colIdx] = val;
			// If the next token is a comma, ignore it and move on
			if (ss.peek() == ',')
			{
				ss.ignore();
			}

			// Increment the column index
			colIdx++;
		}
		tidx++;
	}
	// Close the File
	file.close();
}

template void readCSV(const std::string& filename, Array2D<int>& output);
template void readCSV(const std::string& filename, Array2D<float>& output);
template void readCSV(const std::string& filename, Array2D<double>& output);

template <typename TFloat>
TFloat erfc(TFloat x)
{
	TFloat t, z, ans;

	z = std::abs(x);
	t = 1.0 / (1.0 + 0.5 * z);
	ans =
	    t * exp(-z * z - 1.26551223 +
	            t * (1.00002368 +
	                 t * (0.37409196 +
	                      t * (0.09678418 +
	                           t * (-0.18628806 +
	                                t * (0.27886807 +
	                                     t * (-1.13520398 +
	                                          t * (1.48851587 +
	                                               t * (-0.82215223 +
	                                                    t * 0.17087277)))))))));
	return x >= 0.0 ? ans : 2.0 - ans;
}
template float erfc<float>(float);
template double erfc<double>(double);

template <typename T>
T periodicDiff(const T& a, const T& b, const T& period)
{
	if constexpr (std::is_floating_point<T>::value)
	{
		T halfPeriod = period / static_cast<T>(2.f);
		return halfPeriod -
		       std::abs(std::fmod(std::fabs(a - b), period) - halfPeriod);
	}
	else if constexpr (std::is_integral<T>::value)
	{
		T diff = (a > b) ? (a - b) : (b - a);
		T mod_diff = diff % period;
		return std::min(mod_diff, period - mod_diff);
	}
	else
	{
		throw std::invalid_argument(
		    "Only floating-point or integer types supported");
	}
}
template float periodicDiff(const float&, const float&, const float&);
template double periodicDiff(const double&, const double&, const double&);
template size_t periodicDiff(const size_t&, const size_t&, const size_t&);
template int periodicDiff(const int&, const int&, const int&);

int reflect(int M, int x)
{
	if (x < 0)
	{
		return -x - 1;
	}
	if (x >= M)
	{
		return 2 * M - x - 1;
	}
	return x;
}

int circular(int M, int x)
{
	if (x < 0)
		return x + M;
	if (x >= M)
		return x - M;
	return x;
}

std::string addBeforeExtension(const std::string& fname,
                               const std::string& addition)
{
	int fnameSize = fname.size();
	int pos = fnameSize - 1;
	char lastTwoChars[2] = {0, 0};
	int extensionPosition = -1;

	// Insert before extension, except when the extension is .gz, then wait
	// for the next extension
	while (pos >= 0)
	{
		const char currentChar = fname[pos];
		if (currentChar == '.')
		{
			if (!(lastTwoChars[0] == 'g' && lastTwoChars[1] == 'z'))
			{
				extensionPosition = pos;
				break;
			}
		}
		lastTwoChars[1] = lastTwoChars[0];
		lastTwoChars[0] = currentChar;
		pos--;
	}
	std::string fnameInserted(fname);

	const size_t posWhereInsert =
	    extensionPosition >= 0 ? extensionPosition : fnameSize;

	fnameInserted = fnameInserted.insert(posWhereInsert, addition);
	return fnameInserted;
}

bool endsWith(const std::string& str, const std::string& suffix)
{
	return str.size() >= suffix.size() &&
	       str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

template <typename T>
void conv3D_separable(const Array3DBase<T>& src, const Array1DBase<T>& kernelX,
                      const Array1DBase<T>& kernelY,
                      const Array1DBase<T>& kernelZ, Array3DBase<T>& dst)
{
	size_t kerSize = kernelX.getSize(0);
	int kerIndexCentered =
	    std::floor(kerSize / 2);  // kernel size must always be an odd number
	// and must have same size in all 3 dimensions

	size_t nz = dst.getSize(0);
	size_t ny = dst.getSize(1);
	size_t nx = dst.getSize(2);

	if (nz != src.getSize(0) || ny != src.getSize(1) || nx != src.getSize(2))
	{
		throw std::invalid_argument("Error in convolution, size mismatch "
		                            "between image and destination");
	}
	if (kernelX.getSize(0) != kernelY.getSize(0) ||
	    kernelX.getSize(0) != kernelZ.getSize(0))
	{
		throw std::invalid_argument("Error in convolution, size mismatch "
		                            "between separated kernels");
	}
	if (kernelX.getSize(0) % 2 != 1)
	{
		throw std::logic_error("Kernel size must be an odd number");
	}

	auto dst1 = std::make_unique<Array3D<T>>();
	auto dst2 = std::make_unique<Array3D<T>>();
	dst1->allocate(src.getSize(0), src.getSize(1), src.getSize(2));
	dst2->allocate(src.getSize(0), src.getSize(1), src.getSize(2));

	for (size_t k = 0; k < nz; k++)
	{
		for (size_t j = 0; j < ny; j++)
		{
			for (size_t i = 0; i < nx; i++)
			{
				T sum = 0;
				size_t r = 0;
				for (int kk = -kerIndexCentered; kk <= kerIndexCentered; kk++)
				{
					r = circular(nx, i - kk);
					sum += kernelX.getFlat(kk + kerIndexCentered) *
					       src.get({k, j, r});
				}
				dst1->set({k, j, i}, sum);
			}
		}
	}

	for (size_t k = 0; k < nz; k++)
	{
		for (size_t i = 0; i < nx; i++)
		{
			for (size_t j = 0; j < ny; j++)
			{
				T sum = 0;
				size_t r = 0;
				for (int kk = -kerIndexCentered; kk <= kerIndexCentered; kk++)
				{
					r = circular(ny, j - kk);
					sum += kernelY.getFlat(kk + kerIndexCentered) *
					       dst1->get({k, r, i});
				}
				dst2->set({k, j, i}, sum);
			}
		}
	}

	for (size_t i = 0; i < nx; i++)
	{
		for (size_t j = 0; j < ny; j++)
		{
			for (size_t k = 0; k < nz; k++)
			{
				T sum = 0;
				size_t r = 0;
				for (int kk = -kerIndexCentered; kk <= kerIndexCentered; kk++)
				{
					r = circular(nz, k - kk);
					sum += kernelZ.getFlat(kk + kerIndexCentered) *
					       dst2->get({r, j, i});
				}
				dst.set({k, j, i}, sum);
			}
		}
	}
}

template void conv3D_separable(const Array3DBase<float>& src,
                               const Array1DBase<float>& kernelX,
                               const Array1DBase<float>& kernelY,
                               const Array1DBase<float>& kernelZ,
                               Array3DBase<float>& dst);

template void conv3D_separable(const Array3DBase<double>& src,
                               const Array1DBase<double>& kernelX,
                               const Array1DBase<double>& kernelY,
                               const Array1DBase<double>& kernelZ,
                               Array3DBase<double>& dst);

template <typename T>
void conv3D(const Array3DBase<T>& image, const Array3DBase<T>& kernel,
            Array3DBase<T>& newImage)
{
	size_t i_nz = image.getSize(0);
	size_t i_ny = image.getSize(1);
	size_t i_nx = image.getSize(2);
	size_t k_nz = kernel.getSize(0);
	size_t k_ny = kernel.getSize(1);
	size_t k_nx = kernel.getSize(2);

	// Iterate over image

	for (size_t iz = 0; iz < i_nz - k_nz + 1; iz++)
	{
		for (size_t iy = 0; iy < i_ny - k_ny + 1; iy++)
		{
			for (size_t ix = 0; ix < i_nx - k_nx + 1; ix++)
			{
				// Iterate over kernel
				auto ipos = std::array<size_t, 3>({iz, iy, ix});
				T sum = 0;
				for (size_t kz = 0; kz < k_nz; kz++)
				{
					for (size_t ky = 0; ky < k_ny; ky++)
					{
						for (size_t kx = 0; kx < k_nx; kx++)
						{
							sum += kernel.get({kz, ky, kx}) *
							       image.get({kz + iz, ky + iy, kx + ix});
						}
					}
				}
				newImage.set(ipos, sum);
			}
		}
	}
}

template void conv3D(const Array3DBase<float>& image,
                     const Array3DBase<float>& kernel,
                     Array3DBase<float>& newImage);
template void conv3D(const Array3DBase<double>& image,
                     const Array3DBase<double>& kernel,
                     Array3DBase<double>& newImage);

template <typename T>
void gauss1DKernelFill(Array1DBase<T>& kernel)
{
	size_t kerSize = kernel.getSize(0);
	for (size_t i = 0; i < kerSize; ++i)
	{
		double gauss = 1 / ((double)kerSize) *
		               std::exp(-0.5 * std::pow(i - kerSize / 2.0, 2.0));
		kernel.setFlat(i, gauss);
	}
}

template void gauss1DKernelFill(Array1DBase<float>& kernel);
template void gauss1DKernelFill(Array1DBase<double>& kernel);

template <typename T>
void fillBox(Array3DBase<T>& arr, size_t z1, size_t z2, size_t y1, size_t y2,
             size_t x1, size_t x2)
{
	size_t Nx = x2 - x1 + 1;  // Number of elements
	size_t Ny = y2 - y1 + 1;
	size_t Nz = z2 - z1 + 1;

	T Dx = static_cast<T>(x2 - x1);  // "Distance"
	T Dy = static_cast<T>(y2 - y1);
	T Dz = static_cast<T>(z2 - z1);

	T v000 = arr.get({z1, y1, x1});
	T v001 = arr.get({z1, y1, x2});
	T v010 = arr.get({z1, y2, x1});
	T v011 = arr.get({z1, y2, x2});
	T v100 = arr.get({z2, y1, x1});
	T v101 = arr.get({z2, y1, x2});
	T v110 = arr.get({z2, y2, x1});
	T v111 = arr.get({z2, y2, x2});

	T vp00 = v000;
	T vp10 = v100;
	T vp01 = v010;
	T vp11 = v110;
	T dvp00 = (Dx == 0) ? 0 : (v001 - v000) / Dx;
	T dvp10 = (Dx == 0) ? 0 : (v101 - v100) / Dx;
	T dvp01 = (Dx == 0) ? 0 : (v011 - v010) / Dx;
	T dvp11 = (Dx == 0) ? 0 : (v111 - v110) / Dx;

	for (size_t px = 0; px < Nx; px++)
	{
		T vp0 = vp00;
		T vp1 = vp10;
		T dvp0 = (Dy == 0) ? 0 : (vp01 - vp00) / Dy;
		T dvp1 = (Dy == 0) ? 0 : (vp11 - vp10) / Dy;
		for (size_t py = 0; py < Ny; py++)
		{
			T vp = vp0;
			T dvp = (Dz == 0) ? 0 : (vp1 - vp0) / Dz;
			for (size_t pz = 0; pz < Nz; pz++)
			{
				arr.set({z1 + pz, y1 + py, x1 + px}, vp);
				vp += dvp;
			}
			vp0 += dvp0;
			vp1 += dvp1;
		}
		vp00 += dvp00;
		vp01 += dvp01;
		vp10 += dvp10;
		vp11 += dvp11;
	}
}

template void fillBox(Array3DBase<float>& arr, size_t z1, size_t z2, size_t y1,
                      size_t y2, size_t x1, size_t x2);
template void fillBox(Array3DBase<double>& arr, size_t z1, size_t z2, size_t y1,
                      size_t y2, size_t x1, size_t x2);

int numberOfDigits(int n)
{
	if (n == 0)
	{
		return 1;
	}
	return static_cast<int>(std::ceil(std::log10(n + 1)));
}

std::string padZeros(int number, int numDigits)
{
	const int numberOfZerosToPad = numDigits - numberOfDigits(number);
	if (numberOfZerosToPad < 0)
	{
		throw std::invalid_argument("The number given in padZeros "
		                            "exceeds the number of digits allowed");
	}
	std::string outString;
	outString.append("00000000000000000000000000000", numberOfZerosToPad);
	outString.append(std::to_string(number));
	return outString;
}

template float getAttenuationCoefficientFactor(float proj, float unitFactor);
template double getAttenuationCoefficientFactor(double proj, double unitFactor);

}  // namespace util
}  // namespace yrt
