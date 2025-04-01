/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/Line3D.hpp"
#include "utils/Array.hpp"

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

struct SinogramParams
{
	// Configuration parameters
	float rMax, axialFOV, thetaMax, maxTOF;
	size_t numR, numPhi, numZ, numTheta, numTOFBins;
};

class Sinogram
{
public:
	explicit Sinogram(const std::string& configPath);

	const SinogramParams& getParams() const;

	float getValue(size_t rIdx, size_t phiIdx, size_t zIdx, size_t thetaIdx,
	               size_t tofIdx) const;

	Line3D getLORFromCoords(size_t rIdx, size_t phiIdx, size_t zIdx,
	                        size_t thetaIdx) const;
	float getTOFFromCoord(size_t tofIdx) const;

	float interpolate(const Line3D& lor, float tof) const;

private:
	std::tuple<float, float, float, float>
	    getLORParameters(const Line3D& lor) const;

	float interpolate5D(float rIdx, float phiIdx, float zIdx, float thetaIdx,
	                    float tofIdx) const;

	static std::tuple<size_t, size_t, float> getNeighbors(float idx,
	                                                      size_t max);

protected:
	SinogramParams m_sinoParams;

	// Raw data storage
	std::unique_ptr<Array5DBase<float>> mp_data;
};

class SinogramOwned : public Sinogram
{
public:
	SinogramOwned(const std::string& configPath, const std::string& dataPath);
};
