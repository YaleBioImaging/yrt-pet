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
	// TODO NOW: Python bindings
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

void ScatterEstimator::computeTailFittedScatterEstimate()
{
	const bool saveIntermediate = !m_saveIntermediary_dir.empty();

	computeScatterEstimate();
	if (saveIntermediate)
	{
		mp_scatter_scs->writeToFile(
		    m_saveIntermediary_dir /
		    "intermediary_scatterEstimateNonFitted.his");
	}

	computeAcfInScatterSpace();
	if (saveIntermediate)
	{
		mp_acf_scs->writeToFile(m_saveIntermediary_dir /
		                        "intermediary_AcfInScatterSpace.his");
	}

	computeScatterTailsMask();
	if (saveIntermediate)
	{
		mp_tail_scs->writeToFile(m_saveIntermediary_dir /
		                         "intermediary_scatterTailsMask.his");
	}

	computePromptsAndRandomsInScatterSpace();

	const float fac = computeTailFittingFactor();

	std::cout << "Applying tail-fit factor..." << std::endl;
	mp_scatter_scs->scaleValues(fac);

	// TODO NOW: Apply sensitivity on the scatter estimate

	/*
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
	*/
}

void ScatterEstimator::computeScatterEstimate()
{
	std::cout << "Estimating scatter..." << std::endl;
	ASSERT_MSG(mp_scatter_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for scatter estimates)");

	m_sss.runSSS(*mp_scatter_scs);
}

void ScatterEstimator::computeAcfInScatterSpace()
{
	std::cout << "Populating attenuation correction factors in scatter space..."
	          << std::endl;
	ASSERT_MSG(mp_acf_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for ACFs)");

	fillAcf();
}

void ScatterEstimator::computeScatterTailsMask()
{
	std::cout << "Generating scatter tails mask..." << std::endl;
	ASSERT_MSG(mp_tail_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for tail)");

	fillScatterTailsMask();
}

void ScatterEstimator::computePromptsAndRandomsInScatterSpace()
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
		ASSERT_MSG(
		    mp_randoms_scs->isMemoryValid(),
		    "Scatter-space array is unallocated (for randoms estimates)");
	}

	fillPromptsAndRandoms();
}

float ScatterEstimator::computeTailFittingFactor() const
{
	std::cout << "Computing tail-fit factor..." << std::endl;

	const size_t numSamples = mp_scatter_scs->getSizeTotal();

	// Sanity checks
	ASSERT(numSamples == mp_prompts_scs->getSizeTotal());
	ASSERT(numSamples == mp_tail_scs->getSizeTotal());

	const bool hasRandoms = mp_randoms_scs->isMemoryValid();
	if (hasRandoms)
	{
		ASSERT(numSamples == mp_randoms_scs->getSizeTotal());
	}

	// Only used for printing purposes
	const int numThreads = globals::getNumThreads();
	const size_t progressMax = numSamples;
	util::ProgressDisplayMultiThread progressBar(numThreads, progressMax, 5);

	// Scatter and prompts sum per thread
	std::vector<double> scatterSumPerThread(numThreads, 0.0);
	std::vector<double> promptsSumPerThread(numThreads, 0.0);

	util::parallelForChunked(
	    numSamples, globals::getNumThreads(),
	    [&progressBar, &scatterSumPerThread, &promptsSumPerThread, hasRandoms,
	     this](size_t sampleId, size_t threadId)
	    {
		    progressBar.incrementProgress(threadId, 1);

		    // Only fit inside the mask
		    if (mp_tail_scs->getValueFlat(sampleId) > 0.0f)
		    {
			    float promptsMinusRandoms =
			        mp_prompts_scs->getValueFlat(sampleId);

			    // Remove randoms estimate
			    if (hasRandoms)
			    {
				    promptsMinusRandoms -=
				        mp_randoms_scs->getProjectionValue(sampleId);
			    }

			    promptsSumPerThread[threadId] += promptsMinusRandoms;

			    scatterSumPerThread[threadId] +=
			        mp_scatter_scs->getValueFlat(sampleId);
		    }
	    });

	// TODO NOW: Reduce the sums per thread
	double scatterSum = 0.0;
	double promptsSum = 0.0;
	for (int threadId = 0; threadId < numThreads; threadId++)
	{
		scatterSum += scatterSumPerThread[threadId];
		promptsSum += promptsSumPerThread[threadId];
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

		    // Normalize prompts and randoms estimate
		    if (applySensitivity)
		    {
			    const float sensitivity =
			        mp_sensitivityHis->getProjectionValueFromHistogramBin(
			            histoBin);
			    if (sensitivity > EPS_FLT)
			    {
				    promptsValue /= sensitivity;
				    randomsEstimate /= sensitivity;
			    }
			    else
			    {
				    promptsValue = 0.0f;
				    randomsEstimate = 0.0f;
			    }
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
