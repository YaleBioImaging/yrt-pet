/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Array.hpp"
#include "yrt-pet/utils/Types.hpp"

namespace yrt
{

class Scanner;

/*
 * This class stores an array of boolean values for the purposes of enabling or
 * disabling detectors. The array is stored in RAWD format (from the Array.hpp
 * class).
 * true -> detector is enabled
 * false -> detector is disabled
 */
class DetectorMask
{
public:
	explicit DetectorMask(size_t numDets);
	explicit DetectorMask(const std::string& pr_fname);
	explicit DetectorMask(const Array1DBase<bool>& pr_data);
	DetectorMask(const DetectorMask& other);  // Copy constructor

	// Legacy format
	explicit DetectorMask(const Array3DBase<float>& pr_data);

	void readFromFile(const std::string& fname);

	Array1D<bool>& getData();
	const Array1D<bool>& getData() const;

	void enableAllDetectors();
	void disableAllDetectors();
	void enableDetector(det_id_t detId);
	void disableDetector(det_id_t detId);

	bool checkAgainstScanner(const Scanner& scanner) const;
	size_t getNumDets() const;
	bool checkDetector(det_id_t detId) const;
	bool isDetectorEnabled(det_id_t detId) const;

	void writeToFile(const std::string& fname) const;

	void logicalAndWithOther(const DetectorMask& other);
	void logicalOrWithOther(const DetectorMask& other);
	void logicalXorWithOther(const DetectorMask& other);
	void logicalNandWithOther(const DetectorMask& other);

	size_t countEnabledDetectors() const;

private:
	enum class BinaryOperations
	{
		AND,
		OR,
		XOR,
		NAND
	};
	template <BinaryOperations Oper>
	void logicalOperWithOther(const DetectorMask& other);

	void setDetectorEnabled(det_id_t detId, bool enabled);

	// true -> enabled
	// false -> disabled
	std::unique_ptr<Array1D<bool>> mp_data;
};

}  // namespace yrt
