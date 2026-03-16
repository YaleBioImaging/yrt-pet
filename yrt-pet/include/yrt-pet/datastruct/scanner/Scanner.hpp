/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/Constraints.hpp"
#include "yrt-pet/datastruct/scanner/DetectorSetup.hpp"
#include "yrt-pet/geometry/Vector3D.hpp"

#include <filesystem>
#include <string>
#include <memory>

namespace fs = std::filesystem;

namespace yrt
{
class Scanner
{
public:
	static constexpr float SCANNER_FILE_VERSION = 3.2;

	// Create a scanner while using the parameters to generate a regular
	//  structure. The structure can still later be overridden by a LUT
	Scanner(std::string pr_scannerName, float p_axialFOV, float p_crystalSize_z,
	        float p_crystalSize_trans, float p_crystalDepth,
	        float p_scannerRadius, size_t p_detsPerRing, size_t p_numRings,
	        size_t p_numDOI, size_t p_maxRingDiff, size_t p_minAngDiff,
	        size_t p_detsPerBlock);

	// Create a scanner using a given JSON file
	explicit Scanner(const std::string& p_definitionFile);

	void readFromFile(const std::string& p_definitionFile);
	void readFromString(const std::string& fileContents);

	std::string getScannerPath() const;
	size_t getNumDets() const;
	size_t getExpectedNumDets() const;
	Vector3D getDetectorPos(det_id_t id) const;
	Vector3D getDetectorOrient(det_id_t id) const;
	std::shared_ptr<DetectorSetup> getDetectorSetup() const;
	bool isValid() const;
	bool isDetectorAllowed(det_id_t det) const;
	bool hasMask() const;

	// Allocate and fill array with detector positions
	void createLUT(Array2DOwned<float>& lut) const;
	void setDetectorSetup(const std::shared_ptr<DetectorSetup>& pp_detectors);

	void collectConstraints(
	    std::vector<std::unique_ptr<Constraint>>& constraints) const;

	void addMask(const std::string& mask_fname);
	void addMask(const DetectorMask& mask);

public:
	std::string scannerName;
	float axialFOV, crystalSize_z, crystalSize_trans, crystalDepth,
	    scannerRadius;
	float collimatorRadius, fwhm, energyLLD;  // Optional, for scatter only

	// detsPerRing : Number of detectors per ring (not counting DOI)
	// numRings : Number of rings in total (not counting DOI)
	// numDOI : Number of DOI layers
	// maxRingDiff : Maximum ring difference (number of rings)
	// minAngDiff : Minimum angular difference, in terms of detector indices
	size_t detsPerRing, numRings, numDOI, maxRingDiff, minAngDiff;
	size_t detsPerBlock; // optional

protected:
	fs::path m_scannerPath;
	std::shared_ptr<DetectorSetup> mp_detectors;
};
}  // namespace yrt
