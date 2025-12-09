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
	static constexpr float DefaultACFThreshold = 0.9523809f;  // 1/1.05
	static constexpr int DefaultSeed = 13;
	static constexpr auto DefaultCrystal = CrystalMaterial::LYSO;
	static constexpr size_t DefaultMaskWidth = 2ull;

	ScatterEstimator(const Scanner& pr_scanner, const Image& pr_lambda,
	                 const Image& pr_mu, const ProjectionData& pr_prompts,
	                 size_t numTOFBins, size_t numPlanes, size_t numAngles,
	                 const Histogram3D* pp_randomsHis = nullptr,
	                 const Histogram3D* pp_sensitivityHis = nullptr,
	                 CrystalMaterial p_crystalMaterial = DefaultCrystal,
	                 int seedi = DefaultSeed,
	                 size_t maskWidth = DefaultMaskWidth,
	                 float maskThreshold = DefaultACFThreshold,
	                 const std::string& saveIntermediary_dir = "");

	void allocate();

	// This function calls all the steps
	void computeTailFittedScatterEstimate();

	// Steps (YN: Maybe they should be protected/private)
	void computeScatterEstimate();
	void computeAcfInScatterSpace();
	void computeScatterTailsMask();
	void computePromptsAndRandomsInScatterSpace();
	float computeTailFittingFactor() const;

	// Getters
	const ScatterSpace& getScatterEstimate() const;
	const ScatterSpace& getPromptsInScatterSpace() const;
	const ScatterSpace& getAcfInScatterSpace() const;
	const ScatterSpace& getRandomsInScatterSpace() const;
	const ScatterSpace& getTailInScatterSpace() const;
	bool useRandomsEstimates() const;
	bool useSensitivity() const;

protected:
	void fillAcf();
	void fillScatterTailsMask();
	void fillPromptsAndRandoms();

private:
	// TODO: Eventually, this class should not depend on the fully-sampled
	//  histograms. It should instead use the List-Mode instead of the
	//  prompts and return an under-sampled sinogram instead of a
	//  fully-sampled histogram.
	const Scanner& mr_scanner;
	SingleScatterSimulator m_sss;

	// Inputs for tail-fitting
	// Input projection data (can be list-mode or histogram)
	const ProjectionData& mr_prompts;
	// If randoms estimates histogram is null, the randoms estimates are
	// gathered from the prompts ProjectionData
	const Histogram* mp_randomsHis;
	// For normalisation correction
	const Histogram* mp_sensitivityHis;

	// For the scatter tails mask
	size_t m_scatterTailsMaskWidth;  // Number of neighboring virtual detectors
	float m_maskThreshold;

	// Where to save intermediary scatter-space values
	std::filesystem::path m_saveIntermediary_dir;

	// Note: "scs" stands for "scatter-space"
	// Scatter estimate in scatter-space
	std::unique_ptr<ScatterSpace> mp_scatter_scs;

	// Scatter-space values for tail-fitting purposes
	std::unique_ptr<ScatterSpace> mp_prompts_scs;  // From "mr_prompts"
	std::unique_ptr<ScatterSpace> mp_acf_scs;      // Used to generate tail mask
	std::unique_ptr<ScatterSpace> mp_randoms_scs;  // Randoms estimates
	std::unique_ptr<ScatterSpace> mp_tail_scs;     // Tail mask
};
}  // namespace scatter
}  // namespace yrt
