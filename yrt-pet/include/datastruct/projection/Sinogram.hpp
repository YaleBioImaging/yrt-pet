/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "geometry/Line3D.hpp"
#include "utils/Array.hpp"

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

struct SinogramParams
{
	explicit SinogramParams(const std::string& params_fname);

	// Properties:
	//  Maximum distance to center
	float rMax;
	//  Maximum axial angle
	float thetaMax;
	//  Maximum TOF value
	float maxTOF;

	// Dimensions:
	size_t numR, numPhi, numZ, numTheta, numTOFBins;
};

class Sinogram : public Histogram
{
public:
	Sinogram(const Scanner& pr_scanner, const SinogramParams& params);

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
	std::unique_ptr<Array5DBase<float>> mp_array;
};

class SinogramOwned : public Sinogram
{
public:
	SinogramOwned(const Scanner& pr_scanner, const SinogramParams& params);
	SinogramOwned(const Scanner& pr_scanner, const SinogramParams& params,
	              const std::string& array_fname);
	void allocate();
	void readFromFile(const std::string& array_fname);
};

class SinogramAlias : public Sinogram
{
public:
	SinogramAlias(const Scanner& pr_scanner, const SinogramParams& params);
	void bind(Array5DBase<float>& pr_array);
};
