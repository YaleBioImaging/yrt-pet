/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/ScatterEstimator.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/scatter/Crystal.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_scatterestimator(py::module& m)
{
	auto c = py::class_<scatter::ScatterEstimator>(m, "ScatterEstimator");
	c.def(py::init<const Scanner&, const Image&, const Image&,
	               const Histogram3D*, const Histogram3D*, const Histogram3D*,
	               const Histogram3D*, scatter::CrystalMaterial, int, int,
	               float, const std::string&>(),
	      "scanner"_a, "source_image"_a, "attenuation_image"_a, "prompts_his"_a,
	      "randoms_his"_a, "acf_his"_a, "sensitivity_his"_a,
	      "crystal_material"_a = scatter::ScatterEstimator::DefaultCrystal,
	      "seed"_a = scatter::ScatterEstimator::DefaultSeed,
	      "mask_width"_a = -1,
	      "mask_threshold"_a = scatter::ScatterEstimator::DefaultACFThreshold,
	      "save_intermediary"_a = "");

	c.def("computeTailFittedScatterEstimate",
	      &scatter::ScatterEstimator::computeTailFittedScatterEstimate,
	      "num_z"_a, "num_phi"_a, "num_r"_a, "denormalize"_a = true);
	c.def("computeScatterEstimate",
	      &scatter::ScatterEstimator::computeScatterEstimate, "num_z"_a,
	      "num_phi"_a, "num_r"_a);
	c.def("generateScatterTailsMask",
	      &scatter::ScatterEstimator::generateScatterTailsMask);
	c.def("computeTailFittingFactor",
	      &scatter::ScatterEstimator::computeTailFittingFactor,
	      "scatter_histogram"_a, "scatter_tails_mask"_a);
}
}  // namespace yrt

#endif

namespace yrt
{
namespace scatter
{

ScatterEstimator::ScatterEstimator(
    const Scanner& pr_scanner, const Image& pr_lambda, const Image& pr_mu,
    const Histogram3D* pp_promptsHis, const Histogram3D* pp_randomsHis,
    const Histogram3D* pp_acfHis, const Histogram3D* pp_sensitivityHis,
    CrystalMaterial p_crystalMaterial, int seedi, int maskWidth,
    float maskThreshold, const std::string& saveIntermediary_dir)
    : mr_scanner(pr_scanner),
      m_sss(pr_scanner, pr_mu, pr_lambda, p_crystalMaterial, seedi)
{
	mp_promptsHis = pp_promptsHis;
	mp_acfHis = pp_acfHis;
	ASSERT_MSG(mp_promptsHis != nullptr, "Prompts histogram provided is null");
	ASSERT_MSG(mp_acfHis != nullptr, "ACF histogram provided is null");

	mp_randomsHis = pp_randomsHis;
	mp_sensitivityHis = pp_sensitivityHis;
	if (maskWidth > 0)
	{
		m_scatterTailsMaskWidth = maskWidth;
	}
	else
	{
		// Use one tenth of the histogram width
		m_scatterTailsMaskWidth = mp_promptsHis->numR / 10;
	}
	m_maskThreshold = maskThreshold;
	m_saveIntermediary_dir = saveIntermediary_dir;
}

std::unique_ptr<Histogram3DOwned>
    ScatterEstimator::computeTailFittedScatterEstimate(size_t numberZ,
                                                       size_t numberPhi,
                                                       size_t numberR,
                                                       bool denormalize)
{
	auto scatterEstimate = computeScatterEstimate(numberZ, numberPhi, numberR);

	const auto scatterTailsMask = generateScatterTailsMask();
	if (!m_saveIntermediary_dir.empty())
	{
		scatterTailsMask->writeToFile(m_saveIntermediary_dir /
		                              "intermediary_scatterTailsMask.his");
	}

	const float fac =
	    computeTailFittingFactor(scatterEstimate.get(), scatterTailsMask.get());

	std::cout << "Applying tail-fit factor..." << std::endl;
	scatterEstimate->getData() *= fac;

	if (mp_sensitivityHis != nullptr && denormalize)
	{
		// Since the scatter estimate was tail-fitted using the sensitivity
		//  as a denominator to prompts and randoms, it is necessary to
		//  multiply it with the sensitivity again before using it in the
		//  reconstruction
		std::cout << "Denormalize scatter histogram..." << std::endl;
		scatterEstimate->operationOnEachBinParallel(
		    [this, &scatterEstimate](bin_t bin) -> float
		    {
			    return mp_sensitivityHis->getProjectionValue(bin) *
			           scatterEstimate->getProjectionValue(bin);
		    });
	}

	return scatterEstimate;
}

std::unique_ptr<Histogram3DOwned>
    ScatterEstimator::computeScatterEstimate(size_t numberZ, size_t numberPhi,
                                             size_t numberR)
{
	auto scatterHisto = std::make_unique<Histogram3DOwned>(mr_scanner);
	scatterHisto->allocate();
	scatterHisto->clearProjections();

	m_sss.runSSS(numberZ, numberPhi, numberR, *scatterHisto);

	return scatterHisto;
}

std::unique_ptr<Histogram3DOwned>
    ScatterEstimator::generateScatterTailsMask() const
{
	std::cout << "Generating scatter tails mask..." << std::endl;
	auto scatterTailsMask = std::make_unique<Histogram3DOwned>(mr_scanner);
	scatterTailsMask->allocate();

	fillScatterTailsMask(*mp_acfHis, *scatterTailsMask, m_scatterTailsMaskWidth,
	                     m_maskThreshold);

	return scatterTailsMask;
}

float ScatterEstimator::computeTailFittingFactor(
    const Histogram3D* scatterHistogram,
    const Histogram3D* scatterTailsMask) const
{
	std::cout << "Computing Tail-fit factor..." << std::endl;
	ASSERT_MSG(scatterHistogram->count() == scatterTailsMask->count(),
	           "Size mismatch between input histograms");
	double scatterSum = 0.0f;
	double promptsSum = 0.0f;

	for (bin_t bin = 0; bin < scatterHistogram->count(); bin++)
	{
		// Only fit inside the mask
		if (scatterTailsMask->getProjectionValue(bin) > 0.0f)
		{
			float binValue = mp_promptsHis->getProjectionValue(bin);
			if (mp_randomsHis != nullptr)
			{
				binValue -= mp_randomsHis->getProjectionValue(bin);
			}
			if (mp_sensitivityHis != nullptr)
			{
				const float sensitivityVal =
				    mp_sensitivityHis->getProjectionValue(bin);
				if (sensitivityVal > 1e-8)
				{
					binValue /= sensitivityVal;
				}
				else
				{
					// Ignore zero bins altogether to avoid numerical
					// instability
					continue;
				}
			}

			promptsSum += binValue;
			scatterSum += scatterHistogram->getProjectionValue(bin);
		}
	}
	const float fac = promptsSum / scatterSum;
	std::cout << "Tail-fitting factor: " << fac << std::endl;
	return fac;
}

void ScatterEstimator::fillScatterTailsMask(const Histogram3D& acfHis,
                                            Histogram3D& mask, size_t maskWidth,
                                            float maskThreshold)
{
	const size_t numBins = acfHis.count();
	ASSERT(mask.isMemoryValid());
	ASSERT(acfHis.isMemoryValid());
	mask.clearProjections(0.0f);

	for (bin_t binId = 0; binId < numBins; binId++)
	{
		const float acfValue = acfHis.getProjectionValue(binId);
		const bool initValueOn = acfValue == 0.0 /* For invalid acf bins */ ||
		                         acfValue > maskThreshold;
		mask.setProjectionValue(binId, initValueOn ? 1.0f : 0.0f);
	}

	for (size_t zBin = 0; zBin < acfHis.numZBin; zBin++)
	{
		for (size_t phi = 0; phi < acfHis.numPhi; phi++)
		{
			const size_t initRowBinId = acfHis.getBinIdFromCoords(0, phi, zBin);

			// Process beginning of the mask
			size_t r;
			for (r = 0; r < acfHis.numR; r++)
			{
				const bin_t binId = initRowBinId + r;
				if (mask.getProjectionValue(binId) < 1.0f)
				{
					if (r > maskWidth)
					{
						// Put zeros from the beginning of the row to the
						// current position minus the width of the mask
						for (bin_t newBinId = initRowBinId;
						     newBinId < binId - maskWidth; newBinId++)
						{
							mask.setProjectionValue(newBinId, 0.0f);
						}
					}
					break;
				}
			}

			// For when the line is true everywhere
			if (r == acfHis.numR)
			{
				for (bin_t binId = initRowBinId;
				     binId < initRowBinId + acfHis.numR; binId++)
				{
					mask.setProjectionValue(binId, 0.0f);
				}
				continue;
			}

			// Process end of the mask
			const long lastRValue = static_cast<long>(acfHis.numR - 1);
			for (long reverseR = lastRValue; reverseR >= 0; reverseR--)
			{
				const bin_t binId = initRowBinId + reverseR;
				if (mask.getProjectionValue(binId) < 1.0f)
				{
					if (reverseR < static_cast<long>(acfHis.numR - maskWidth))
					{
						// Put zeros from the end of the row to the
						// current position plus the width of the mask
						for (long newR = lastRValue;
						     newR >= static_cast<long>(reverseR + maskWidth);
						     newR--)
						{
							const bin_t newBinId = newR + initRowBinId;
							mask.setProjectionValue(newBinId, 0.0f);
						}
					}
					break;
				}
			}
		}
	}
}
}  // namespace scatter
}  // namespace yrt
