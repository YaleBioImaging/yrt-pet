/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Array.hpp"

namespace yrt
{

class Scanner;

class DetectorMask
{
public:
	explicit DetectorMask(const std::string& pr_fname);
	explicit DetectorMask(const Array1DBase<bool>& pr_data);
	DetectorMask(const DetectorMask& other); // Copy constructor

	// Legacy format
	explicit DetectorMask(const Array3DBase<float>& pr_data);

	void readFromFile(const std::string& fname);

	Array1D<bool>& getData();
	const Array1D<bool>& getData() const;

	bool checkAgainstScanner(const Scanner& scanner) const;
	size_t getNumDets() const;
	bool checkDetector(size_t detId) const;

	void writeToFile(const std::string& fname) const;

private:
	std::unique_ptr<Array1D<bool>> mp_data;
};

}  // namespace yrt
