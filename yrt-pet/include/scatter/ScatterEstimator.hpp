/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/Histogram3D.hpp"
#include "scatter/Crystal.hpp"
#include "scatter/SingleScatterSimulator.hpp"

class Scanner;
class Image;

namespace Scatter
{
	class ScatterEstimator
	{
	public:
		static constexpr float DefaultACFThreshold = 0.9523809f;  // 1/1.05
		static constexpr int DefaultSeed = 13;
		static constexpr auto DefaultCrystal = CrystalMaterial::LYSO;

		ScatterEstimator(const Scanner& pr_scanner, const Image& pr_lambda,
		                 const Image& pr_mu, const Histogram3D* pp_promptsHis,
		                 const Histogram3D* pp_randomsHis,
		                 const Histogram3D* pp_acfHis,
		                 CrystalMaterial p_crystalMaterial = DefaultCrystal,
		                 int seedi = DefaultSeed, int maskWidth = -1,
		                 float maskThreshold = DefaultACFThreshold,
		                 const std::string& saveIntermediary_dir = "");

		std::shared_ptr<Histogram3DOwned>
		    computeTailFittedScatterEstimate(size_t numberZ, size_t numberPhi,
		                                     size_t numberR);

		std::shared_ptr<Histogram3DOwned>
		    computeScatterEstimate(size_t numberZ, size_t numberPhi,
		                           size_t numberR);

		std::shared_ptr<Histogram3DOwned> generateScatterTailsMask() const;

		float
		    computeTailFittingFactor(const Histogram3D* scatterHistogram,
		                             const Histogram3D* scatterTailsMask) const;

	protected:
		static void fillScatterTailsMask(const Histogram3D& acfHis,
		                                 Histogram3D& mask, size_t maskWidth,
		                                 float maskThreshold);

	private:
		// TODO: Eventually, this class should not depend on the fully-sampled
		//  histograms. It should instead use the List-Mode instead of the
		//  prompts and return an under-sampled sinogram instead of a
		//  fully-sampled histogram.
		const Scanner& mr_scanner;
		SingleScatterSimulator m_sss;
		const Histogram3D* mp_promptsHis;
		const Histogram3D* mp_randomsHis;
		const Histogram3D* mp_acfHis;

		// For the scatter tails mask
		std::filesystem::path m_saveIntermediary_dir;  // save the scatter tails mask used
		float m_maskThreshold;
		size_t m_scatterTailsMaskWidth;
	};
}  // namespace Scatter
