/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/ScatterEstimator.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/scatter/SingleScatterSimulatorUtils.cuh"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/GPUStream.cuh"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/utils/Timer.hpp"
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
	               const Histogram3D*, const Histogram3D*, timestamp_t, int,
	               size_t, float, float, float, float, const std::string&, bool,
	               bool, float>(),
	      "scanner"_a, "lambda"_a, "mu"_a, "prompts"_a, "num_tof_bins"_a,
	      "num_planes"_a, "num_angles"_a, "randoms_his"_a = nullptr,
	      "sensitivity_his"_a = nullptr, "scan_duration"_a = 0,
	      "seedi"_a = scatter::SingleScatterSimulator::DefaultSeed,
	      "scatter_tails_mask_width"_a =
	          scatter::ScatterEstimator::DefaultScatterTailsMaskWidth,
	      "att_threshold_tail"_a =
	          scatter::ScatterEstimator::DefaultAttThresholdTail,
	      "att_threshold_sampling"_a =
	          scatter::SingleScatterSimulator::DefaultAttThresholdSampling,
	      "num_samp_frac"_a =
	          scatter::SingleScatterSimulator::DefaultNumSampFrac,
	      "detection_threshold"_a =
	          scatter::SingleScatterSimulator::DefaultDetectionThreshold,
	      "saveIntermediary_dir"_a = "", "only_direct_planes"_a = true,
	      "use_gpu"_a = false,
	      "lor_downsampling"_a =
	          scatter::ScatterEstimator::DefaultLORDownsamplingFactor);

	// Allocation
	c.def("allocate", &scatter::ScatterEstimator::allocate);

	// Main function
	c.def("computeTailFittedScatterEstimate",
	      &scatter::ScatterEstimator::computeTailFittedScatterEstimate,
	      "unscaled_scatter_estimate"_a = nullptr,
	      "Compute the scatter estimate and perform the tail fitting. To skip "
	      "the scatter estimation calculation and only perform the "
	      "tail-fitting, provide an unscaled scatter estimate as a parameter.");

	// Steps
	c.def("computeScatterEstimate",
	      &scatter::ScatterEstimator::computeScatterEstimate);
	c.def(
	    "computeAttenuationInsideMaskInScatterSpace",
	    &scatter::ScatterEstimator::computeAttenuationInsideMaskInScatterSpace);
	c.def_static("computeInsideMaskInScatterSpace",
	             &scatter::ScatterEstimator::computeInsideMaskInScatterSpace,
	             "image"_a, "threshold"_a, "inside_mask"_a);
	c.def("computeScatterTailsMask",
	      &scatter::ScatterEstimator::computeScatterTailsMask);
	c.def_static("computeTailsMask",
	             &scatter::ScatterEstimator::computeTailsMask, "inside_mask"_a,
	             "tails_mask"_a,
	             "mask_width"_a =
	                 scatter::ScatterEstimator::DefaultScatterTailsMaskWidth);
	c.def("computePromptsAndRandomsInScatterSpace",
	      &scatter::ScatterEstimator::computePromptsInScatterSpace);
	c.def(
	    "computeSensitivityAndRandomsInScatterSpace",
	    &scatter::ScatterEstimator::computeSensitivityAndRandomsInScatterSpace);
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
    const Histogram* pp_sensitivityHis, timestamp_t p_scanDuration, int seedi,
    size_t p_scatterTailsMaskWidth, float p_attThresholdTail,
    float p_attThresholdSampling, float p_numSampFrac,
    float p_detectionThreshold, const std::string& p_saveIntermediary_dir,
    bool p_onlyDirectPlanes, bool p_useGPU, float p_lorDownsamplingFactor)
    : mr_scanner(pr_scanner),
      m_sss(pr_scanner, pr_mu, pr_lambda, seedi, p_attThresholdSampling,
            p_numSampFrac, p_detectionThreshold),
      mr_prompts(pr_prompts),
      mp_randomsHis(pp_randomsHis),
      mp_sensitivityHis(pp_sensitivityHis),
      m_scatterTailsMaskWidth(p_scatterTailsMaskWidth),
      m_attThresholdTail(p_attThresholdTail),
      m_scanDuration(p_scanDuration),
      m_onlyEstimateDirectPlanes(p_onlyDirectPlanes),
      m_useGPU(p_useGPU),
      m_lorDownsamplingFactor(p_lorDownsamplingFactor),
      m_saveIntermediary_dir(p_saveIntermediary_dir)
{
	// Scatter estimate in scatter-space
	mp_scatter_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                                numPlanes, numAngles);

	// Other scatter-space components
	mp_prompts_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                                numPlanes, numAngles);
	mp_randoms_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                                numPlanes, numAngles);
	mp_sensitivity_scs = std::make_unique<ScatterSpace>(mr_scanner, numTOFBins,
	                                                    numPlanes, numAngles);

	initScanDuration();

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

	mp_sensitivity_scs->allocate();

	mp_tail_scs->allocate();
}

void ScatterEstimator::initScanDuration()
{
	if (m_scanDuration == 0)
	{
		if (isPromptsListMode())
		{
			// This is the scan duration in milliseconds
			const timestamp_t scanDuration_ms = mr_prompts.getScanDuration();
			// Get the scan duration in *seconds*
			m_scanDuration = scanDuration_ms / 1000;
		}
		else
		{
			// When working with a Histogram, do not scale by scan duration
			//  unless the user provides a scaling.
			m_scanDuration = 1;
		}
	}
	std::cout << "Scan duration to be used: " << m_scanDuration << " sec"
	          << std::endl;
	ASSERT_MSG(m_scanDuration != 0.0f, "Scan duration cannot be zero");
}

void ScatterEstimator::computeTailFittedScatterEstimate(
    const ScatterSpace* unscaledScatterEstimate)
{
	const bool saveIntermediate = !m_saveIntermediary_dir.empty();

	// Note: Technically, the tail selection can be done in parallel to the
	//  scatter estimation. This would save some computation time.

	util::Timer timer;

	auto measureTimeTaken = [&timer]<typename FuncType>(FuncType oper)
	{
		std::cout << "------------" << std::endl;
		timer.run();
		oper();
		std::cout << "Time taken: " << timer.getElapsedSeconds() << "s"
		          << std::endl;
		timer.reset();
	};

	if (unscaledScatterEstimate != nullptr)
	{
		// Copy it into the member field
		mp_scatter_scs->copyFrom(*unscaledScatterEstimate);
	}
	else
	{
		// Perform the estimation
		measureTimeTaken([this]() { computeScatterEstimate(); });
	}
	if (saveIntermediate)
	{
		mp_scatter_scs->writeToFile(
		    m_saveIntermediary_dir /
		    "intermediary_scatterEstimateNonFitted.scs");
	}

	measureTimeTaken([this]()
	                 { computeAttenuationInsideMaskInScatterSpace(); });
	if (saveIntermediate)
	{
		mp_insideMask_scs->writeToFile(m_saveIntermediary_dir /
		                               "intermediary_insideMask.scs");
	}

	measureTimeTaken([this]() { computeScatterTailsMask(); });
	if (saveIntermediate)
	{
		mp_tail_scs->writeToFile(m_saveIntermediary_dir /
		                         "intermediary_scatterTailsMask.scs");
	}

	measureTimeTaken([this]() { computePromptsInScatterSpace(); });
	if (saveIntermediate)
	{
		mp_prompts_scs->writeToFile(m_saveIntermediary_dir /
		                            "intermediary_prompts.scs");
	}

	measureTimeTaken([this]()
	                 { computeSensitivityAndRandomsInScatterSpace(); });
	if (saveIntermediate)
	{
		if (mp_randoms_scs->isMemoryValid())
		{
			mp_randoms_scs->writeToFile(m_saveIntermediary_dir /
			                            "intermediary_randoms.scs");
		}
		mp_sensitivity_scs->writeToFile(m_saveIntermediary_dir /
		                                "intermediary_sensitivity.scs");
	}

	float fac = -1;
	measureTimeTaken([this, &fac]() { fac = computeTailFittingFactor(); });
	ASSERT(fac > 0.0f);

	std::cout << "Applying tail-fit factor..." << std::endl;
	measureTimeTaken([this, fac]() { mp_scatter_scs->scaleValues(fac); });
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

	if (m_useGPU)
	{
#if BUILD_CUDA
		m_sss.runSSSDevice(*mp_scatter_scs, m_onlyEstimateDirectPlanes);
#else
		throw std::runtime_error("GPU SSS unavailable if project was not "
		                         "compiled with CUDA enabled");
#endif
	}
	else
	{
		m_sss.runSSS(*mp_scatter_scs, m_onlyEstimateDirectPlanes);
	}

	if (m_onlyEstimateDirectPlanes)
	{
		mp_scatter_scs->fillNonDirectPlanes();
	}
}

void ScatterEstimator::computeAttenuationInsideMaskInScatterSpace()
{
	std::cout << "Computing inside-outside mask for tail-fitting..."
	          << std::endl;

	computeInsideMaskInScatterSpace(m_sss.getAttenuationImage(),
	                                m_attThresholdTail, *mp_insideMask_scs);
}

void ScatterEstimator::computeInsideMaskInScatterSpace(const Image& image,
                                                       float threshold,
                                                       ScatterSpace& insideMask)
{
	// Note: The attenuation image used here should not include the bed
	ASSERT_MSG(insideMask.isMemoryValid(),
	           "Scatter-space array is unallocated (for ACFs)");

	const size_t numSamples = insideMask.getSizeTotal();

	const RawImageConst imageRaw = getRawImage(image);

	// Only used for printing purposes
	const int numThreads = globals::getNumThreads();
	const size_t progressMax = numSamples;
	util::ProgressDisplayMultiThread progressBar(numThreads, progressMax, 10);

	util::parallelForChunked(
	    numSamples, numThreads,
	    [&progressBar, &imageRaw, threshold, &insideMask](size_t sampleId,
	                                                      size_t threadId)
	    {
		    progressBar.incrementProgress(threadId);

		    const ScatterSpace::ScatterSpaceIndex scsIdx =
		        insideMask.unravelIndex(sampleId);

		    // Ignore TOF
		    const Line3D lor = insideMask.getLORFromIndex(scsIdx);

		    // Forward-project the attenuation image
		    const bool inside =
		        doesLineIntersectImageThreshold(lor, imageRaw, threshold);

		    insideMask.setValueFlat(sampleId, inside ? 1.0f : 0.0f);
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
	util::ProgressDisplayMultiThread progressBar(numThreads, progressMax, 10);

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

void ScatterEstimator::computePromptsInScatterSpace()
{
	std::cout << "Populating prompts in scatter space..." << std::endl;
	ASSERT_MSG(mp_prompts_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for prompts)");

	// Iterate on all events or all histogram bins
	const size_t count = mr_prompts.count();

	// Detector mask
	const bool hasMask = mr_scanner.hasMask();

	// Only used for printing purposes
	const int numThreads = globals::getNumThreads();

	// Prompts
	util::ProgressDisplayMultiThread progressBar(numThreads, count, 5);
	util::parallelForChunked(
	    count, numThreads,
	    [&progressBar, hasMask, this](size_t binId, size_t threadId)
	    {
		    progressBar.incrementProgress(threadId);

		    // Gather prompts
		    const float promptsValue = mr_prompts.getProjectionValue(binId);

		    // Histogram bin
		    const histo_bin_t histoBin = mr_prompts.getHistogramBin(binId);

		    if (hasMask)
		    {
			    det_id_t d1, d2;
			    bool skip = false;
			    if (std::holds_alternative<det_pair_t>(histoBin))
			    {
				    const auto detPair = std::get<det_pair_t>(histoBin);
				    d1 = detPair.d1;
				    d2 = detPair.d2;
				    skip |= !mr_scanner.isDetectorAllowed(d1);
				    skip |= !mr_scanner.isDetectorAllowed(d2);
			    }
			    else if (std::holds_alternative<det_pair_tof_t>(histoBin))
			    {
				    const auto detPairTOF = std::get<det_pair_tof_t>(histoBin);
				    d1 = detPairTOF.d1;
				    d2 = detPairTOF.d2;
				    skip |= !mr_scanner.isDetectorAllowed(d1);
				    skip |= !mr_scanner.isDetectorAllowed(d2);
			    }

			    if (skip)
			    {
				    // Skip detector pair
				    return;
			    }
		    }

		    // Gather scatter-space index
		    const ScatterSpace::ScatterSpacePosition scsPos =
		        mp_prompts_scs->histogramBinToScatterSpacePosition(histoBin);
		    const ScatterSpace::ScatterSpaceIndex scsIdx =
		        mp_prompts_scs->getNearestNeighborIndex(scsPos);

		    // Increment scatter-space arrays (Atomic)
		    mp_prompts_scs->incrementValueAtomic(scsIdx, promptsValue);
	    });
	mp_prompts_scs->symmetrizeIfNeeded();
}

void ScatterEstimator::computeSensitivityAndRandomsInScatterSpace()
{
	const bool useRandoms = this->useRandomsEstimates();
	const bool useSensitivity = this->useSensitivity();

	std::cout << "Populating ";
	if (useRandoms)
	{
		std::cout << "randoms estimates ";
	}
	if (useRandoms && useSensitivity)
	{
		std::cout << "and ";
	}
	if (useSensitivity)
	{
		std::cout << "sensitivity ";
	}
	std::cout << "in scatter space..." << std::endl;

	ASSERT_MSG(mp_prompts_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for prompts)");
	ASSERT_MSG(useRandoms == mp_randoms_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for randoms estimates)");
	ASSERT_MSG(mp_sensitivity_scs->isMemoryValid(),
	           "Scatter-space array is unallocated (for sensitivity)");

	// Detector mask
	const bool hasMask = mr_scanner.hasMask();

	// Only used for printing purposes
	const int numThreads = globals::getNumThreads();

	// Randoms and sensitivity
	if (useRandoms || useSensitivity)
	{
		auto histo = Histogram3DAlias(mr_scanner);
		auto binIter = histo.getBinIter(1, 0);
		auto binIter_ptr = binIter.get();
		util::ProgressDisplayMultiThread progressBar(numThreads,
		                                             binIter->size(), 5);
		util::parallelForChunkedRandomized(
		    binIter->size(), numThreads, m_lorDownsamplingFactor,
		    [&progressBar, useRandoms, useSensitivity, &histo, binIter_ptr,
		     hasMask,
		     this](size_t binIdx, size_t /*counter*/, unsigned int threadId)
		    {
			    progressBar.incrementProgress(threadId);

			    const bin_t binId = binIter_ptr->get(binIdx);
			    const det_pair_t histoBin = histo.getDetPairFromBinId(binId);

			    if (hasMask)
			    {
				    if (!mr_scanner.isDetectorAllowed(histoBin.d1) ||
				        !mr_scanner.isDetectorAllowed(histoBin.d2))
				    {
					    // Skip detector pair
					    return;
				    }
			    }

			    float randomsValue = 0.0f;
			    if (useRandoms)
			    {
				    // Gather randoms estimate of current detector pair
				    randomsValue =
				        mp_randomsHis->getProjectionValueFromHistogramBin(
				            histoBin);
			    }

			    float sensitivityValue = 1.0f;
			    if (useSensitivity)
			    {
				    // Gather sensitivity of current detector pair
				    sensitivityValue =
				        mp_sensitivityHis->getProjectionValueFromHistogramBin(
				            histoBin);
			    }

			    // Gather scatter-space index
			    const ScatterSpace::ScatterSpacePosition scsPos =
			        mp_randoms_scs->histogramBinToScatterSpacePosition(
			            histoBin);
			    const ScatterSpace::ScatterSpaceIndex scsIdx =
			        mp_randoms_scs->getNearestNeighborIndex(scsPos);

			    // Increment scatter-space arrays (Atomic)
			    if (useRandoms)
			    {
				    mp_randoms_scs->incrementValueAtomic(scsIdx, randomsValue);
			    }
			    mp_sensitivity_scs->incrementValueAtomic(scsIdx,
			                                             sensitivityValue);
		    });

		if (useRandoms)
		{
			// Since we only took a given ratio of the possible LORs,
			//  scale this amount by the inverse to get the proper scaling
			// Since the randoms are in units of "counts/seconds", we need to
			//  scale them by the scan duration to have them in units of
			//  "counts".
			mp_randoms_scs->scaleValues(static_cast<float>(m_scanDuration) /
			                            m_lorDownsamplingFactor);

			mp_randoms_scs->symmetrizeIfNeeded();
		}
		// We scale sensitivity for the same reason as for the randoms:
		// Since we only took a given ratio of the possible LORs,
		//  scale this amount by the inverse to get the proper scaling
		mp_sensitivity_scs->scaleValues(1.0f / m_lorDownsamplingFactor);

		mp_sensitivity_scs->symmetrizeIfNeeded();
	}
	else
	{
		std::cout << "Skipped..." << std::endl;
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
	ASSERT(mp_sensitivity_scs->isMemoryValid());
	ASSERT(numSamples == mp_sensitivity_scs->getSizeTotal());

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

			    float alphaDenominator = mp_scatter_scs->getValueFlat(sampleId);

			    alphaDenominator *= mp_sensitivity_scs->getValueFlat(sampleId);

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

	// The denominator should be scaled by the scan duration
	alphaDenominator *= m_scanDuration;

	const float fac = alphaNumerator / alphaDenominator;

	std::cout << "Tail-fitting factor: " << fac << std::endl;

	ASSERT_MSG_WARNING(fac != 0, "Tail-fitting failure: The factor is zero");
	ASSERT_MSG_WARNING(fac >= 0,
	                   "Tail-fitting failure: The factor is negative");

	return fac;
}

bool ScatterEstimator::isPromptsListMode() const
{
	const ListMode* promptsAsListMode =
	    dynamic_cast<const ListMode*>(&mr_prompts);
	return promptsAsListMode != nullptr;
}

const ScatterSpace& ScatterEstimator::getScatterEstimate() const
{
	return *mp_scatter_scs;
}

const ScatterSpace& ScatterEstimator::getPromptsInScatterSpace() const
{
	return *mp_prompts_scs;
}

const ScatterSpace& ScatterEstimator::getRandomsInScatterSpace() const
{
	return *mp_randoms_scs;
}

const ScatterSpace& ScatterEstimator::getInsideMaskInScatterSpace() const
{
	return *mp_insideMask_scs;
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
