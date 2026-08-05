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
	static constexpr float DefaultAttThresholdTail = 0.05;  // 1/cm
	static constexpr float DefaultLORDownsamplingFactor = 0.1f;
	static constexpr size_t DefaultScatterTailsMaskWidth = 2ull;

	ScatterEstimator(
	    const Scanner& pr_scanner, const Image& pr_lambda, const Image& pr_mu,
	    const ProjectionData& pr_prompts, size_t numTOFBins, size_t numPlanes,
	    size_t numAngles, const Histogram* pp_randomsHis = nullptr,
	    const Histogram* pp_sensitivityHis = nullptr,
	    timestamp_t p_scanDuration = 0,
	    int seedi = SingleScatterSimulator::DefaultSeed,
	    size_t p_scatterTailsMaskWidth = DefaultScatterTailsMaskWidth,
	    float p_attThresholdTail = DefaultAttThresholdTail,
	    float p_attThresholdSampling =
	        SingleScatterSimulator::DefaultAttThresholdSampling,
	    float p_numSampFrac = SingleScatterSimulator::DefaultNumSampFrac,
	    float p_detectionThreshold =
	        SingleScatterSimulator::DefaultDetectionThreshold,
	    const std::string& p_saveIntermediary_dir = "",
	    bool p_onlyDirectPlanes = true, bool p_useGPU = false,
	    float p_lorDownsamplingFactor = DefaultLORDownsamplingFactor);

	// Allocate scatter-space buffers
	void allocate();
	void initScanDuration();

	// This function calls all the steps
	void computeTailFittedScatterEstimate(
	    const ScatterSpace* unscaledScatterEstimate = nullptr);

	// Steps (YN: Maybe they should be protected/private)
	void computeScatterEstimate();
	void computeAttenuationInsideMaskInScatterSpace();
	static void computeInsideMaskInScatterSpace(const Image& image,
	                                            float threshold,
	                                            ScatterSpace& insideMask);
	void computeScatterTailsMask();
	static void
	    computeTailsMask(const ScatterSpace& insideMask,
	                     ScatterSpace& tailsMask,
	                     size_t maskWidth = DefaultScatterTailsMaskWidth);
	void computePromptsInScatterSpace();
	void computeSensitivityAndRandomsInScatterSpace();
	float computeTailFittingFactor() const;

	// Getters
	bool isPromptsListMode() const;  // Return true if prompts are a list-mode
	const ScatterSpace& getScatterEstimate() const;
	const ScatterSpace& getPromptsInScatterSpace() const;
	const ScatterSpace& getRandomsInScatterSpace() const;
	const ScatterSpace& getInsideMaskInScatterSpace() const;
	const ScatterSpace& getTailInScatterSpace() const;
	bool useRandomsEstimates() const;
	bool useSensitivity() const;

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
	// Threshold on the attenuation image to consider a voxel "inside" the
	//  object
	const float m_attThresholdTail;
	// Duration of the scan (in seconds)
	timestamp_t m_scanDuration;
	// If true, only estimate direct plane and fill non-direct from average
	bool m_onlyEstimateDirectPlanes;
	// Use the GPU-accelerated version of the SSS calculation
	bool m_useGPU;
	// The ratio of LORs to consider for the estimation of sensitivity and
	//  randoms for every possible LOR.
	//  (example: 0.02 -> take 2% of LORs, 1.0 -> take all LORs)
	float m_lorDownsamplingFactor;

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
	// Populated from the sensitivity histogram
	std::unique_ptr<ScatterSpace> mp_sensitivity_scs;

	// LOR inside the object: 1.0; Outside the object: 0.0
	std::unique_ptr<ScatterSpace> mp_insideMask_scs;
	// LOR inside the tail: 1.0; Outside the tail: 0.0
	std::unique_ptr<ScatterSpace> mp_tail_scs;
};
}  // namespace scatter
}  // namespace yrt
