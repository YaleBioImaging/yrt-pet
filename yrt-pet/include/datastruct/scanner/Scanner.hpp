/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/scanner/DetCoord.hpp"
#include "geometry/Vector3D.hpp"

#include <filesystem>
#include <string>

#define SCANNER_FILE_VERSION 3.1

namespace fs = std::filesystem;

class Scanner
{
public:
	Scanner(std::string pr_scannerName, float p_axialFOV,
	        float p_crystalSize_z, float p_crystalSize_trans, float p_crystalDepth,
	        float p_scannerRadius, size_t p_detsPerRing, size_t p_numRings,
	        size_t p_numDOI, size_t p_maxRingDiff, size_t p_minAngDiff,
	        size_t p_detsPerBlock);
	explicit Scanner(const std::string& p_definitionFile);
	void readFromFile(const std::string& p_definitionFile);
	void readFromString(const std::string& fileContents);
	std::string getScannerPath() const;
	size_t getNumDets() const;
	size_t getTheoreticalNumDets() const;
	Vector3DFloat getDetectorPos(det_id_t id) const;
	Vector3DFloat getDetectorOrient(det_id_t id) const;
	std::shared_ptr<DetectorSetup> getDetectorSetup() const;
	bool isValid() const;

	// Allocate and fill array with detector positions
	void createLUT(Array2D<float>& lut) const;
	void setDetectorSetup(const std::shared_ptr<DetectorSetup>& pp_detectors);

public:
	std::string scannerName;
	float axialFOV, crystalSize_z, crystalSize_trans, crystalDepth,
	    scannerRadius;
	float collimatorRadius, fwhm, energyLLD;  // Optional, for scatter only

	// dets_per_ring : Number of detectors per ring (not counting DOI)
	// num_rings : Number of rings in total (not countring DOI)
	// num_doi : Number of DOI crystals (ex: 2 for SAVANT)
	// max_ring_diff : Maximum ring difference (number of rings)
	// min_ang_diff : Minimum angular difference, in terms of detector indices
	size_t detsPerRing, numRings, numDOI, maxRingDiff, minAngDiff;
	size_t detsPerBlock;

protected:
	fs::path m_scannerPath;
	std::shared_ptr<DetectorSetup> mp_detectors;
};
