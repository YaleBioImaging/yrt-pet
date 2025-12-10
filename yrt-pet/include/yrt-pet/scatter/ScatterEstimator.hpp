/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/scatter/Crystal.hpp"
#include "yrt-pet/scatter/ScatterSpace.hpp"
#include "yrt-pet/scatter/SingleScatterSimulator.hpp"

namespace yrt
{

class Scanner;
class Image;

namespace scatter
{
class ScatterEstimator
{
public:
	static constexpr float DefaultAttThreshold = 0.9523809f;  // 1/1.05
	static constexpr int DefaultSeed = 13;
	static constexpr auto DefaultCrystal = CrystalMaterial::LYSO;
	static constexpr size_t DefaultScatterTailsMaskWidth = 2ull;

	ScatterEstimator(
	    const Scanner& pr_scanner, const Image& pr_lambda, const Image& pr_mu,
	    const ProjectionData& pr_prompts, size_t numTOFBins, size_t numPlanes,
	    size_t numAngles, const Histogram3D* pp_randomsHis = nullptr,
	    const Histogram3D* pp_sensitivityHis = nullptr,
	    CrystalMaterial p_crystalMaterial = DefaultCrystal,
	    int seedi = DefaultSeed,
	    size_t scatterTailsMaskWidth = DefaultScatterTailsMaskWidth,
	    float attThreshold = DefaultAttThreshold,
	    const std::string& saveIntermediary_dir = "");

	void allocate();

	// This function calls all the steps
	void computeTailFittedScatterEstimate();

	// Steps (YN: Maybe they should be protected/private)
	void computeScatterEstimate();
	void computeInsideMaskInScatterSpace();
	void computeScatterTailsMask();
	void computePromptsAndRandomsInScatterSpace();
	float computeTailFittingFactor() const;

	// Getters
	const ScatterSpace& getScatterEstimate() const;
	const ScatterSpace& getPromptsInScatterSpace() const;
	const ScatterSpace& getRandomsInScatterSpace() const;
	const ScatterSpace& getInsideMaskInScatterSpace() const;
	const ScatterSpace& getTailInScatterSpace() const;
	bool useRandomsEstimates() const;
	bool useSensitivity() const;

protected:
	void fillInsideMask();
	void fillScatterTailsMask();
	void fillPromptsAndRandoms();

private:
	const Scanner& mr_scanner;
	SingleScatterSimulator m_sss;

	// -------------------------------------------------------------------------

	// Inputs for tail-fitting

	// Input projection data (can be list-mode or histogram)
	const ProjectionData& mr_prompts;
	// If randoms estimates histogram is null, the randoms estimates are
	// gathered from the prompts ProjectionData
	const Histogram* mp_randomsHis;
	// For normalisation correction
	const Histogram* mp_sensitivityHis;

	// For the scatter tails mask:
	// Number of neighboring virtual detectors
	const size_t m_scatterTailsMaskWidth;
	// Threshold on the forward projection of the attenuation image to consider
	// an LOR "inside" the object
	const float m_attThreshold;

	// Where to save intermediary scatter-space values
	std::filesystem::path m_saveIntermediary_dir;

	// -------------------------------------------------------------------------
	// Scatter-space values

	// Note: "scs" stands for "scatter-space"

	// Scatter estimate
	std::unique_ptr<ScatterSpace> mp_scatter_scs;

	// For tail-fitting purposes:
	// Populated from "mr_prompts"
	std::unique_ptr<ScatterSpace> mp_prompts_scs;
	// Populated from randoms estimates
	std::unique_ptr<ScatterSpace> mp_randoms_scs;

	// TODO: We might need to also store a scatter-space vector for the
	//  sensitivity (multiplied by livetime if available)

	// LOR inside the object: 1.0; Outside the object: 0.0
	std::unique_ptr<ScatterSpace> mp_insideMask_scs;
	// LOR inside the tail: 1.0; Outside the tail: 0.0
	std::unique_ptr<ScatterSpace> mp_tail_scs;
};
}  // namespace scatter
}  // namespace yrt
