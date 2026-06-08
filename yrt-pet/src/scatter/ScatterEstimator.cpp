/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/ScatterEstimator.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/operators/ProjectorSiddon.hpp"
#include "yrt-pet/scatter/Crystal.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/utils/Types.hpp"

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
	               const ProjectionData&, size_t, size_t, size_t,
	               const Histogram3D*, const Histogram3D*,
	               scatter::CrystalMaterial, int, size_t, float, float,
	               const std::string&, bool>(),
	      "scanner"_a, "lambda"_a, "mu"_a, "prompts"_a, "num_tof_bins"_a,
	      "num_planes"_a, "num_angles"_a, "randoms_his"_a = nullptr,
	      "sensitivity_his"_a = nullptr,
	      "crystal_material"_a = scatter::ScatterEstimator::DefaultCrystal,
	      "seedi"_a = scatter::ScatterEstimator::DefaultSeed,
	      "scatter_tails_mask_width"_a =
	          scatter::ScatterEstimator::DefaultScatterTailsMaskWidth,
	      "att_threshold"_a = scatter::ScatterEstimator::DefaultAttThreshold,
	      "num_samp_frac"_a = scatter::ScatterEstimator::DefaultNumSampFrac,
	      "saveIntermediary_dir"_a = "", "only_direct_planes"_a = true);

	// Allocation
	c.def("allocate", &scatter::ScatterEstimator::allocate);

	// Main function
	c.def("computeTailFittedScatterEstimate",
	      &scatter::ScatterEstimator::computeTailFittedScatterEstimate);

	// Steps
	c.def("computeScatterEstimate",
	      &scatter::ScatterEstimator::computeScatterEstimate);
	c.def("computeInsideMaskInScatterSpace",
	      &scatter::ScatterEstimator::computeInsideMaskInScatterSpace);
	c.def("computeScatterTailsMask",
	      &scatter::ScatterEstimator::computeScatterTailsMask);
	c.def_static("computeTailsMask",
	             &scatter::ScatterEstimator::computeTailsMask, "inside_mask"_a,
	             "tails_mask"_a,
	             "mask_width"_a =
	                 scatter::ScatterEstimator::DefaultScatterTailsMaskWidth);
	c.def("computePromptsAndRandomsInScatterSpace",
	      &scatter::ScatterEstimator::computePromptsAndRandomsInScatterSpace);
	c.def("computeTailFittingFactor",
	      &scatter::ScatterEstimator::computeTailFittingFactor);

	// Getters
	c.def("getScatterEstimate", &scatter::ScatterEstimator::getScatterEstimate);
	c.def("getPromptsInScatterSpace",
	      &scatter::ScatterEstimator::getPromptsInScatterSpace);
	c.def("getRandomsInScatterSpace",
	      &scatter::ScatterEstimator::getRandomsInScatterSpace);
	c.def("getInsideMaskInScatterSpace",
	      &scatter::ScatterEstimator::getInsideMaskInScatterSpace);
	c.def("getTailInScatterSpace",
	      &scatter::ScatterEstimator::getTailInScatterSpace);
	c.def("useRandomsEstimates",
	      &scatter::ScatterEstimator::useRandomsEstimates);
	c.def("useSensitivity", &scatter::ScatterEstimator::useSensitivity);
}
}  // namespace yrt

#endif

namespace yrt::scatter
{

ScatterEstimator::ScatterEstimator(
    const Scanner& pr_scanner, const Image& pr_lambda, const Image& pr_mu,
    const ProjectionData& pr_prompts, size_t numTOFBins, size_t numPlanes,
    size_t numAngles, const Histogram* pp_randomsHis,
    const Histogram* pp_sensitivityHis, CrystalMaterial p_crystalMaterial,
    int seedi, size_t p_scatterTailsMaskWidth, float p_attThreshold,
    float p_numSampFrac, const std::string& p_saveIntermediary_dir,
    bool p_onlyDirectPlanes)
    : mr_scanner(pr_scanner),
      m_sss(pr_scanner, pr_mu, pr_lambda, p_crystalMaterial, seedi,
            p_numSampFrac),
      mr_prompts(pr_prompts),
      mp_randomsHis(pp_randomsHis),
      mp_sensitivityHis(pp_sensitivityHis),
      m_scatterTailsMaskWidth(p_scatterTailsMaskWidth),
      m_attThreshold(p_attThreshold),
      m_saveIntermediary_dir(p_saveIntermediary_dir),
      m_onlyEstimateDirectPlanes(p_onlyDirectPlanes)
{
	// Scatter estimate in scatter-space
	mp_scatter_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                                numPlanes, numAngles);

	// Other scatter-space components
	mp_prompts_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                                numPlanes, numAngles);
	mp_randoms_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                                numPlanes, numAngles);

	// Tail-fitting is done without TOF
	constexpr size_t numTOFBinsForTailFitting = 1ull;
	mp_insideMask_scs = std::make_unique<ScatterSpace>(
	    mr_scanner, numTOFBinsForTailFitting, numPlanes, numAngles);
	mp_tail_scs = std::make_unique<ScatterSpace>(
	    mr_scanner, numTOFBinsForTailFitting, numPlanes, numAngles);
}

void ScatterEstimator::allocate()
{
	mp_scatter_scs->allocate();
	mp_prompts_scs->allocate();
	mp_insideMask_scs->allocate();

	if (useRandomsEstimates())
	{
		mp_randoms_scs->allocate();
	}

	mp_tail_scs->allocate();
}

void ScatterEstimator::computeTailFittedScatterEstimate()
{
	const bool saveIntermediate = !m_saveIntermediary_dir.empty();

	// Note: Technically, the tail selection can be done in parallel to the
	//  scatter estimation. This would save some computation time.

	computeScatterEstimate();
	if (saveIntermediate)
	{
		mp_scatter_scs->writeToFile(
		    m_saveIntermediary_dir /
		    "intermediary_scatterEstimateNonFitted.scs");
	}

	computeInsideMaskInScatterSpace();
	if (saveIntermediate)
	{
		mp_insideMask_scs->writeToFile(m_saveIntermediary_dir /
		                               "intermediary_insideMask.scs");
	}

	computeScatterTailsMask();
	if (saveIntermediate)
	{
		mp_tail_scs->writeToFile(m_saveIntermediary_dir /
		                         "intermediary_scatterTailsMask.scs");
	}

	computePromptsAndRandomsInScatterSpace();
	if (saveIntermediate)
	{
		mp_prompts_scs->writeToFile(m_saveIntermediary_dir /
		                            "intermediary_prompts.scs");
		mp_randoms_scs->writeToFile(m_saveIntermediary_dir /
		                            "intermediary_randoms.scs");
	}

	const float fac = computeTailFittingFactor();

	std::cout << "Applying tail-fit factor..." << std::endl;
	mp_scatter_scs->scaleValues(fac);

	// TODO: Apply sensitivity on the scatter estimate

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
	std::cout << "Estimating scatter";
	if (m_onlyEstimateDirectPlanes)
	{
		std::cout << " (direct planes only, filling non-direct from average)";
	}
	std::cout << "..." << std::endl;

	ASSERT_MSG(mp_scatter_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for scatter estimates)");

	m_sss.runSSS(*mp_scatter_scs, m_onlyEstimateDirectPlanes);

	if (m_onlyEstimateDirectPlanes)
	{
		mp_scatter_scs->fillNonDirectPlanes();
	}
}

void ScatterEstimator::computeInsideMaskInScatterSpace()
{
	// TODO: This should be simplified.
	//  We don't need to do the full forward projection, just a check for
	//  whether the LOR intersects the object or not. We would need to re-write
	//  a new "projector"
	//  It can also be simplified further by only projecting d1-d2 and not d2-d1

	std::cout << "Computing inside-outside mask in preparation for tail-fitting"
	          << std::endl;

	// Note: The attenuation image used should not include the bed
	ASSERT_MSG(mp_insideMask_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for ACFs)");

	const size_t numSamples = mp_insideMask_scs->getSizeTotal();

	// Only used for printing purposes
	const int numThreads = globals::getNumThreads();
	const size_t progressMax = numSamples;
	util::ProgressDisplayMultiThread progressBar(numThreads, progressMax, 5);

	util::parallelForChunked(
	    numSamples, numThreads,
	    [&progressBar, this](size_t sampleId, size_t threadId)
	    {
		    progressBar.incrementProgress(threadId);

		    const ScatterSpace::ScatterSpaceIndex scsIdx =
		        mp_insideMask_scs->unravelIndex(sampleId);

		    // Ignore TOF
		    const Line3D lor = mp_insideMask_scs->getLORFromIndex(scsIdx);

		    // Forward-project the attenuation image
		    const float att = ProjectorSiddon::singleForwardProjection(
		        &m_sss.getAttenuationImage(), lor);

		    const float inside = (att > m_attThreshold) ? 1.0f : 0.0f;

		    mp_insideMask_scs->setValueFlat(sampleId, inside);
	    });
}

void ScatterEstimator::computeScatterTailsMask()
{
	std::cout << "Generating scatter tails mask..." << std::endl;
	computeTailsMask(*mp_insideMask_scs, *mp_tail_scs, m_scatterTailsMaskWidth);
}

void ScatterEstimator::computeTailsMask(const ScatterSpace& insideMask,
                                        ScatterSpace& tailsMask,
                                        size_t maskWidth)
{
	ASSERT(insideMask.isMemoryValid());
	ASSERT(tailsMask.isMemoryValid());
	ASSERT(insideMask.getSizeTotal() == tailsMask.getSizeTotal());

	const size_t numAngles1 = insideMask.getNumAngles();
	const size_t numAngles2 = numAngles1;
	const size_t numSamples = insideMask.getSizeTotal();
	const size_t maskWidthDiv2 = maskWidth / 2;  // Integer division
	const size_t neighborhoodSize = maskWidth + 1;

	// For printing purposes
	const int numThreads = globals::getNumThreads();
	const size_t progressMax = numSamples;
	util::ProgressDisplayMultiThread progressBar(numThreads, progressMax, 5);

	// Need to build a little "kernel" and populate it with all the neighbors
	std::vector<std::vector<float>> neighborhoodPerThread;
	neighborhoodPerThread.resize(numThreads);
	for (auto& neighborhood : neighborhoodPerThread)
	{
		neighborhood.resize(neighborhoodSize);
	}

	// Parallelize over planeIndex1
	util::parallelForChunked(
	    numSamples, numThreads,
	    [&progressBar, numAngles2, &insideMask, maskWidth, maskWidthDiv2,
	     neighborhoodSize, &neighborhoodPerThread,
	     &tailsMask](size_t sampleIdx, unsigned int threadId)
	    {
		    progressBar.incrementProgress(threadId);

		    auto& neighborhood = neighborhoodPerThread[threadId];

		    const auto scsIdx = insideMask.unravelIndex(sampleIdx);
		    const size_t angleIdx1 = scsIdx.angleIndex1;
		    const size_t planeIdx1 = scsIdx.planeIndex1;
		    const size_t angleIdx2 = scsIdx.angleIndex2;
		    const size_t planeIdx2 = scsIdx.planeIndex2;

		    if (angleIdx2 < maskWidth || angleIdx2 + maskWidth >= numAngles2)
		    {
			    // Outside bounds
			    tailsMask.setValue(0, planeIdx1, angleIdx1, planeIdx2,
			                       angleIdx2, 0.0f);
			    return;
		    }

		    // Fill-up the neighborhood
		    for (size_t neighborIdx = 0; neighborIdx < neighborhoodSize;
		         neighborIdx++)
		    {
			    const ssize_t neighborOffset = -maskWidthDiv2 + neighborIdx;
			    const ssize_t neighborAngleIdx = angleIdx2 + neighborOffset;

			    neighborhood[neighborIdx] = insideMask.getValue(
			        0, planeIdx1, angleIdx1, planeIdx2, neighborAngleIdx);
		    }

		    // If the neighborhood is all ones or all zeros, we are
		    //  not in a tail, otherwise we are.
		    const auto firstVal = neighborhood[0];
		    bool isInTail = false;
		    for (size_t neighborIdx = 1; neighborIdx < neighborhoodSize;
		         neighborIdx++)
		    {
			    if (neighborhood[neighborIdx] != firstVal)
			    {
				    // Some values are different in the
				    // neighborhood. We are therefore in a tail.
				    isInTail = true;
			    }
		    }

		    tailsMask.setValue(0, planeIdx1, angleIdx1, planeIdx2, angleIdx2,
		                       isInTail ? 1.0f : 0.0f);
	    });
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

	// Iterate on all events or all histogram bins
	const size_t count = mr_prompts.count();

	const bool applySensitivity = useSensitivity();

	// Only used for printing purposes
	const int numThreads = globals::getNumThreads();

	// Prompts
	{
		util::ProgressDisplayMultiThread progressBar(numThreads, count, 5);
		util::parallelForChunked(
		    count, numThreads,
		    [&progressBar, applySensitivity, this](size_t binId,
		                                           size_t threadId)
		    {
			    progressBar.incrementProgress(threadId);

			    // Gather prompts
			    float promptsValue = mr_prompts.getProjectionValue(binId);

			    // Histogram bin
			    const histo_bin_t histoBin = mr_prompts.getHistogramBin(binId);

			    // Normalize prompts and randoms estimate
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

			    // Gather scatter-space index
			    const ScatterSpace::ScatterSpacePosition scsPos =
			        mp_prompts_scs->histogramBinToScatterSpacePosition(
			            histoBin);
			    const ScatterSpace::ScatterSpaceIndex scsIdx =
			        mp_prompts_scs->getNearestNeighborIndex(scsPos);

			    // Increment scatter-space arrays (Atomic)
			    mp_prompts_scs->incrementValueAtomic(scsIdx, promptsValue);
		    });
		mp_prompts_scs->symmetrizeIfNeeded();
	}

	// Randoms
	if (useRandoms)
	{
		auto histo = Histogram3DAlias(mr_scanner);
		auto binIter = histo.getBinIter(1, 0);
		auto binIter_ptr = binIter.get();
		util::ProgressDisplayMultiThread progressBar(numThreads,
		                                             binIter->size(), 5);
		util::parallelForChunked(
		    binIter->size(), numThreads,
		    [&progressBar, applySensitivity, &histo, binIter_ptr,
		     this](size_t binIdx, size_t threadId)
		    {
			    progressBar.incrementProgress(threadId);
			    bin_t binId = binIter_ptr->get(binIdx);
			    det_pair_t histoBin = histo.getDetPairFromBinId(binId);

			    // Gather prompts
			    float randomsValue =
			        mp_randomsHis->getProjectionValueFromHistogramBin(histoBin);

			    // Normalize prompts and randoms estimate
			    if (applySensitivity)
			    {
				    const float sensitivity =
				        mp_sensitivityHis->getProjectionValueFromHistogramBin(
				            histoBin);
				    if (sensitivity > EPS_FLT)
				    {
					    randomsValue /= sensitivity;
				    }
				    else
				    {
					    randomsValue = 0.0f;
				    }
			    }

			    // Gather scatter-space index
			    const ScatterSpace::ScatterSpacePosition scsPos =
			        mp_randoms_scs->histogramBinToScatterSpacePosition(
			            histoBin);
			    const ScatterSpace::ScatterSpaceIndex scsIdx =
			        mp_randoms_scs->getNearestNeighborIndex(scsPos);

			    // Increment scatter-space arrays (Atomic)
			    mp_randoms_scs->incrementValueAtomic(scsIdx, randomsValue);
		    });
		mp_randoms_scs->symmetrizeIfNeeded();
	}
}

float ScatterEstimator::computeTailFittingFactor() const
{
	std::cout << "Computing tail-fitting factor..." << std::endl;

	ASSERT(mp_tail_scs->isMemoryValid());
	ASSERT(mp_scatter_scs->isMemoryValid());
	ASSERT(mp_prompts_scs->isMemoryValid());

	const size_t numSamples = mp_scatter_scs->getSizeTotal();

	// Sanity checks
	ASSERT(numSamples == mp_prompts_scs->getSizeTotal());

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
	std::vector<double> alphaNumeratorSumPerThread(numThreads, 0.0);
	std::vector<double> alphaDenominatorSumPerThread(numThreads, 0.0);

	util::parallelForChunked(
	    numSamples, numThreads,
	    [&progressBar, &alphaDenominatorSumPerThread,
	     &alphaNumeratorSumPerThread, hasRandoms,
	     this](size_t sampleId, size_t threadId)
	    {
		    progressBar.incrementProgress(threadId);

		    const ScatterSpace::ScatterSpaceIndex scsIdx =
		        mp_scatter_scs->unravelIndex(sampleId);

		    // Gather the tail value using the TOF-disabled scatter-space index
		    const float tailValue =
		        mp_tail_scs->getValue(0, scsIdx.planeIndex1, scsIdx.angleIndex1,
		                              scsIdx.planeIndex2, scsIdx.angleIndex2);

		    // Only fit inside the tail mask (Value should be 1.0)
		    if (tailValue > 0.0f)
		    {
			    // Gather prompts-randoms
			    float alphaNumerator = mp_prompts_scs->getValueFlat(sampleId);

			    // Remove randoms estimate if available
			    if (hasRandoms)
			    {
				    alphaNumerator -=
				        mp_randoms_scs->getProjectionValue(sampleId);
			    }

			    alphaNumeratorSumPerThread[threadId] += alphaNumerator;

			    const float alphaDenominator =
			        mp_scatter_scs->getValueFlat(sampleId);
			    alphaDenominatorSumPerThread[threadId] += alphaDenominator;
		    }
	    });

	double alphaNumerator = 0.0;
	double alphaDenominator = 0.0;
	for (int threadId = 0; threadId < numThreads; threadId++)
	{
		alphaNumerator += alphaNumeratorSumPerThread[threadId];
		alphaDenominator += alphaDenominatorSumPerThread[threadId];
	}

	const float fac = alphaNumerator / alphaDenominator;
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

const ScatterSpace& ScatterEstimator::getInsideMaskInScatterSpace() const
{
	return *mp_insideMask_scs;
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

}  // namespace yrt::scatter
