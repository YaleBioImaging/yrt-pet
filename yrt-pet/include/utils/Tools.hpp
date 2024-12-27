/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "utils/Array.hpp"

#include <string>

#define IDX2(x, y, Nx) ((y) * (Nx) + (x))
#define IDX3(x, y, z, Nx, Ny) ((z) * (Ny) * (Nx) + (y) * (Nx) + (x))
#define IDX4(x, y, z, t, Nx, Ny, Nz) \
	((t) * (Nz) * (Ny) * (Nx) + (z) * (Ny) * (Nx) + (y) * (Nx) + (x))


namespace Util
{
	// CSV Reader
	template <typename T>
	void readCSV(const std::string& filename, Array2D<T>& output);

	template <typename T>
	void conv3D(const Array3DBase<T>& image, const Array3DBase<T>& kernel,
	            Array3DBase<T>& newImage);

	int reflect(int M, int x);
	int circular(int M, int x);

	template <typename T>
	void conv3D_separable(const Array3DBase<T>& src,
	                      const Array1DBase<T>& kernelX,
	                      const Array1DBase<T>& kernelY,
	                      const Array1DBase<T>& kernelZ, Array3DBase<T>& dst);

	template <typename T>
	void gauss1DKernelFill(Array1DBase<T>& kernel);

	// Complementary error function
	template<typename TFloat>
	TFloat erfc(TFloat x);

	std::string addBeforeExtension(const std::string& fname,
	                               const std::string& addition);

	/**
	 * Fills a given box by trilinear interpolation. It fills from the values at
	 * the coordinates given : (z1,y1,x1), (z1,y1,x2),(z1,y2,x1),(z1,y2,x2)
	 * (z2,y1,x1), (z2,y1,x2), (z2,y2,x1), (z2,y2,x2)
	 **/
	template <typename T>
	void fillBox(Array3DBase<T>& arr, size_t z1, size_t z2, size_t y1,
	             size_t y2, size_t x1, size_t x2);

	/**
	 * @brief Return a string version of an iterable STL container.
	 * @param container Iterable container.
	 * @return The string representation of the container.
	 */
	template <typename T>
	std::string iterableToCompactString(const T& container,
	                                    bool includeBrackets = true,
	                                    std::string delimiter = ",")
	{
		std::string s = (includeBrackets) ? "[" : "";
		size_t i = 0;
		for (const auto& item : container)
		{
			s += std::to_string(item);
			if ((i + 1) < container.size())
			{
				s += delimiter;
			}
			i++;
		}
		s += (includeBrackets) ? "]" : "";
		return s;
	}

	int numberOfDigits(int n);
	std::string padZeros(int number, int numDigits);

	template <typename T>
	T getAttenuationCoefficientFactor(T proj, T unitFactor = 0.1)
	{
		return exp(-proj * unitFactor);
	}

}  // namespace Util
