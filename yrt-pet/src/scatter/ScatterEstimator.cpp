/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/ScatterEstimator.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/operators/OperatorProjectorSiddon.hpp"
#include "yrt-pet/scatter/Crystal.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"
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
	      &scatter::ScatterEstimator::computeScatterTailsMask);
	c.def("computeTailFittingFactor",
	      &scatter::ScatterEstimator::computeTailFittingFactor,
	      "scatter_histogram"_a, "scatter_tails_mask"_a);
}
}  // namespace yrt

#endif

namespace yrt::scatter
{

ScatterEstimator::ScatterEstimator(
    const Scanner& pr_scanner, const Image& pr_lambda, const Image& pr_mu,
    const ProjectionData& pr_prompts, size_t numTOFBins, size_t numPlanes,
    size_t numAngles, const Histogram3D* pp_randomsHis,
    const Histogram3D* pp_sensitivityHis, CrystalMaterial p_crystalMaterial,
    int seedi, size_t maskWidth, float maskThreshold,
    const std::string& saveIntermediary_dir)
    : mr_scanner(pr_scanner),
      m_sss(pr_scanner, pr_mu, pr_lambda, p_crystalMaterial, seedi),
      mr_prompts(pr_prompts),
      mp_randomsHis(pp_randomsHis),
      mp_sensitivityHis(pp_sensitivityHis),
      m_scatterTailsMaskWidth(maskWidth),
      m_maskThreshold(maskThreshold),
      m_saveIntermediary_dir(saveIntermediary_dir)
{
	// Scatter estimate in scatter-space
	mp_scatter_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                                numPlanes, numAngles);

	// Other scatter-space components
	mp_prompts_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                                numPlanes, numAngles);
	mp_acf_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                            numPlanes, numAngles);
	mp_randoms_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                                numPlanes, numAngles);
	mp_tail_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                             numPlanes, numAngles);
}

void ScatterEstimator::allocate()
{
	mp_scatter_scs->allocate();
	mp_prompts_scs->allocate();
	mp_acf_scs->allocate();

	if (useRandomsEstimates())
	{
		mp_randoms_scs->allocate();
	}

	mp_tail_scs->allocate();
}

std::unique_ptr<Histogram3DOwned>
    ScatterEstimator::computeTailFittedScatterEstimate()
{
	computeScatterEstimate();

	if (!m_saveIntermediary_dir.empty())
	{
		mp_scatter_scs->writeToFile(
		    m_saveIntermediary_dir /
		    "intermediary_scatterEstimate_nonfitted.his");
	}

	computeScatterTailsMask();

	if (!m_saveIntermediary_dir.empty())
	{
		mp_tail_scs->writeToFile(m_saveIntermediary_dir /
		                         "intermediary_scatterTailsMask.his");
	}

	computePromptsAcfAndRandomsInScatterSpace();

	const float fac = computeTailFittingFactor();

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

void ScatterEstimator::computeScatterEstimate()
{
	std::cout << "Estimating scatter..." << std::endl;
	ASSERT_MSG(mp_scatter_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for scatter estimates)");

	m_sss.runSSS(*mp_scatter_scs);
}

void ScatterEstimator::computeScatterTailsMask()
{
	std::cout << "Generating scatter tails mask..." << std::endl;
	ASSERT_MSG(mp_tail_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for tail)");

	fillScatterTailsMask();
}

void ScatterEstimator::computePromptsAcfAndRandomsInScatterSpace()
{
	const bool useRandoms = useRandomsEstimates();

	std::cout << "Populating prompts ";
	if (useRandoms)
	{
		std::cout << "and randoms estimates ";
	}
	std::cout << "in scatter space..." << std::endl;
	ASSERT_MSG(mp_prompts_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for prompts)");
	if (useRandoms)
	{
		ASSERT_MSG(mp_randoms_scs->isMemoryValid(),
		           "Scatter-space array is unallocated (for randoms estimates)");
	}

	fillPromptsAndRandoms();

	std::cout << "Populating attenuation correction factors in scatter space..."
	          << std::endl;
	ASSERT_MSG(mp_acf_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for ACFs)");

	fillAcf();
}

float ScatterEstimator::computeTailFittingFactor() const
{
	std::cout << "Computing tail-fit factor..." << std::endl;
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

const ScatterSpace& ScatterEstimator::getScatterEstimate() const
{
	return *mp_scatter_scs;
}

const ScatterSpace& ScatterEstimator::getPromptsInScatterSpace() const
{
	return *mp_prompts_scs;
}

const ScatterSpace& ScatterEstimator::getAcfInScatterSpace() const
{
	return *mp_acf_scs;
}

const ScatterSpace& ScatterEstimator::getRandomsInScatterSpace() const
{
	return *mp_randoms_scs;
}

const ScatterSpace& ScatterEstimator::getTailInScatterSpace() const
{
	return *mp_tail_scs;
}

bool ScatterEstimator::useRandomsEstimates() const
{
	return mp_randomsHis != nullptr || mr_prompts.hasRandomsEstimates();
}

bool ScatterEstimator::useSensitivity() const
{
	return mp_sensitivityHis != nullptr;
}

void ScatterEstimator::fillScatterTailsMask()
{
	// TODO NOW:
	//  - For every bin in the scatter-space:
	//    - Compute the ACF and populate the ACF scatter-space array
	//    - Designate whether it's a tail or not
	//  - Expand the tail by "m_scatterTailsMaskWidth" detectors
	throw std::runtime_error("Function fillScatterTailsMask() unimplemented");
}

void ScatterEstimator::fillPromptsAndRandoms()
{
	// Iterate on all events or all histogram bins
	const size_t count = mr_prompts.count();

	const bool applySensitivity = useSensitivity();

	// Only used for printing purposes
	const int numThreads = globals::getNumThreads();
	const size_t progressMax = count;
	util::ProgressDisplayMultiThread progressBar(numThreads, progressMax, 5);

	util::parallelForChunked(
	    count, globals::getNumThreads(),
	    [&progressBar, applySensitivity, this](size_t binId, size_t threadId)
	    {
		    progressBar.incrementProgress(threadId);

		    // Gather prompts
		    float promptsValue = mr_prompts.getProjectionValue(binId);

		    // Histogram bin
		    const histo_bin_t histoBin = mr_prompts.getHistogramBin(binId);

		    // Normalize prompts value
		    if (applySensitivity)
		    {
			    const float sensitivity =
			        mp_sensitivityHis->getProjectionValueFromHistogramBin(
			            histoBin);
			    if (sensitivity > EPS_FLT)
			    {
				    promptsValue /= sensitivity;
			    }
			    else
			    {
				    promptsValue = 0.0f;
			    }
		    }

		    // Gather randoms estimate if needed
		    float randomsEstimate = 0.0f;
		    if (mp_randomsHis != nullptr)
		    {
			    randomsEstimate =
			        mp_randomsHis->getProjectionValueFromHistogramBin(histoBin);
		    }
		    else if (mr_prompts.hasRandomsEstimates())
		    {
			    randomsEstimate = mr_prompts.hasRandomsEstimates();
		    }

		    // Gather scatter-space index
		    const ScatterSpace::ScatterSpacePosition scsPos =
		        mp_prompts_scs->histogramBinToScatterSpacePosition(histoBin);
		    const ScatterSpace::ScatterSpaceIndex scsIdx =
		        mp_prompts_scs->getNearestNeighborIndex(scsPos);

		    // Increment scatter-space arrays (Atomic)
		    mp_prompts_scs->incrementValueAtomic(scsIdx, promptsValue);
		    if (randomsEstimate > 0.0f)
		    {
			    mp_randoms_scs->incrementValueAtomic(scsIdx, randomsEstimate);
		    }
	    });
}

void ScatterEstimator::fillAcf()
{

	const size_t numSamples = mp_acf_scs->getSizeTotal();

	// Only used for printing purposes
	const int numThreads = globals::getNumThreads();
	const size_t progressMax = numSamples;
	util::ProgressDisplayMultiThread progressBar(numThreads, progressMax, 5);

	util::parallelForChunked(
	    numSamples, globals::getNumThreads(),
	    [&progressBar, this](size_t sampleId, size_t threadId)
	    {
		    progressBar.incrementProgress(threadId, 1);

		    const ScatterSpace::ScatterSpaceIndex scsIdx =
		        mp_acf_scs->unravelIndex(sampleId);

		    // Ignore TOF
		    const Line3D lor = mp_acf_scs->getLORFromIndex(scsIdx);

		    // Forward-project the attenuation image
		    const float att = OperatorProjectorSiddon::singleForwardProjection(
		                          &m_sss.getAttenuationImage(), lor) /
		                      10.0f;
		    const float acf = exp(-att);

		    mp_acf_scs->setValueFlat(sampleId, acf);
	    });

}  // namespace yrt::scatter

}  // namespace yrt::scatter
