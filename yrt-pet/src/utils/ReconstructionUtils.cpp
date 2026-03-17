/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/ReconstructionUtils.hpp"

#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/datastruct/scanner/DetectorMask.hpp"
#include "yrt-pet/geometry/Matrix.hpp"
#include "yrt-pet/recon/LREM_CPU.hpp"
#include "yrt-pet/recon/OSEM_CPU.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplay.hpp"
#include "yrt-pet/utils/Tools.hpp"

#if BUILD_CUDA
#include "yrt-pet/utils/GPUStream.cuh"
#include "yrt-pet/utils/ReconstructionUtilsDevice.cuh"
#endif


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_reconstructionutils(pybind11::module& m)
{
	m.def("histogram3DToListModeLUT", &util::histogram3DToListModeLUT);
	m.def("compareListModes", &util::compareListModes);

	m.def(
	    "convertToHistogram3D",
	    [](const Histogram& dat, Histogram3D& histoOut,
	       const DetectorMask* detectorMask)
	    {
		    util::convertToHistogram3D<false, true>(dat, histoOut,
		                                            detectorMask);
	    },
	    "histo_input"_a, "histo_out"_a, "detector_mask"_a = nullptr);
	m.def(
	    "convertToHistogram3D",
	    [](const ListMode& dat, Histogram3D& histoOut,
	       const DetectorMask* detectorMask)
	    {
		    util::convertToHistogram3D<true, true>(dat, histoOut, detectorMask);
	    },
	    "listmode_input"_a, "histo_out"_a, "detector_mask"_a = nullptr);
	m.def(
	    "convertToHistogram3D",
	    [](const Histogram& dat, const DetectorMask* detectorMask)
	    { return util::convertToHistogram3D<false, true>(dat, detectorMask); },
	    "histo_input"_a, "detector_mask"_a = nullptr);
	m.def(
	    "convertToHistogram3D",
	    [](const ListMode& dat, const DetectorMask* detectorMask)
	    { return util::convertToHistogram3D<true, true>(dat, detectorMask); },
	    "listmode_input"_a, "detector_mask"_a = nullptr);

	m.def("convertToListModeLUT", &util::convertToListModeLUT<true>,
	      "listmode"_a, "detector_mask"_a = nullptr);

	m.def(
	    "createOSEM",
	    [](const Scanner& scanner, bool useGPU, bool isLowRank)
	    {
		    auto osem = util::createOSEM(scanner, useGPU, isLowRank);
		    osem->enableNeedToMakeCopyOfSensImage();
		    return osem;
	    },
	    "scanner"_a, "use_gpu"_a = false, "is_low_rank"_a = false);

	m.def(
	    "createOperatorProjector",
	    [](const ProjectorParams& projParams, const BinIterator* binIter,
	       const std::vector<Constraint*>& constraints)
	    {
		    return std::make_unique<OperatorProjector>(projParams, binIter,
		                                               constraints);
	    },
	    "proj_params"_a, "bin_iter"_a,
	    "constraints"_a = std::vector<Constraint*>(),
	    "Helper function that create a projection operator. This function will "
	    "simply call the constructor of OperatorProjector");

	m.def("getFullTimeRange", util::getFullTimeRange, "lor_motion"_a,
	      "Get the maximum time range occupied by an LORMotion");

	m.def("timeAverageMoveImage",
	      static_cast<std::unique_ptr<ImageOwned> (*)(
	          const LORMotion&, const Image*)>(&util::timeAverageMoveImage),
	      "lor_motion"_a, "unmoved_image"_a,
	      "Blur a given image based on given motion information. Return the "
	      "resulting image.");

	m.def(
	    "timeAverageMoveImage",
	    static_cast<void (*)(const LORMotion&, const Image*, Image*, frame_t)>(
	        &util::timeAverageMoveImage),
	    "lor_motion"_a, "unmoved_image"_a, "out_image"_a,
	    "out_dynamic_frame"_a = 0,
	    "Blur a given image based on given motion information. Write directly "
	    "in \"out_image\" in the dynamic frame \"out_dynamic_frame\".");

	m.def("timeAverageMoveImage",
	      static_cast<std::unique_ptr<ImageOwned> (*)(
	          const LORMotion&, const Image*, timestamp_t, timestamp_t)>(
	          &util::timeAverageMoveImage),
	      "lor_motion"_a, "unmoved_image"_a, "time_start"_a, "time_stop"_a,
	      "Blur a given image based on given motion information. Return the "
	      "resulting image. Use \"time_start\" and \"time_stop\" to define how "
	      "motion frames are selected and weighted.");

	m.def("timeAverageMoveImage",
	      static_cast<void (*)(const LORMotion&, const Image*, Image*,
	                           timestamp_t, timestamp_t, frame_t)>(
	          &util::timeAverageMoveImage),
	      "lor_motion"_a, "unmoved_image"_a, "out_image"_a, "time_start"_a,
	      "time_stop"_a, "out_dynamic_frame"_a = 0,
	      "Blur a given image based on given motion information. Write "
	      "directly in \"out_image\" in the dynamic frame "
	      "\"out_dynamic_frame\". Use \"time_start\" and \"time_stop\" to "
	      "define how motion frames are selected and weighted.");

	m.def("timeAverageMoveImageDynamic",
	      static_cast<std::unique_ptr<ImageOwned> (*)(
	          const LORMotion& lorMotion, const Image* unmovedImage,
	          const DynamicFraming& dynamicFraming)>(
	          &util::timeAverageMoveImageDynamic),
	      "lor_motion"_a, "unmoved_image"_a, "dynamic_framing"_a,
	      "Blur a given image based on the given motion information, but "
	      "follow the dynamic framing provided. The dynamic framing provided "
	      "will be used to select the motion frames to use for each blurring. "
	      "This is used for generating a 4-dimensional sensitivity image "
	      "with both a dynamic framing and rigid motion correction. Return the "
	      "resulting image.");
	m.def("timeAverageMoveImageDynamic",
	      static_cast<void (*)(const LORMotion&, const Image*, Image*,
	                           const DynamicFraming&)>(
	          &util::timeAverageMoveImageDynamic),
	      "lor_motion"_a, "unmoved_image"_a, "out_image"_a, "dynamic_framing"_a,
	      "Blur a given image based on the given motion information, but "
	      "follow the dynamic framing provided. The dynamic framing provided "
	      "will be used to select the motion frames to use for each blurring. "
	      "This is used for generating a 4-dimensional sensitivity image "
	      "with both a dynamic framing and rigid motion correction. The "
	      "resulting image will be written in \"out_image\".");

	m.def("generateTORRandomDOI", &util::generateTORRandomDOI, "scanner"_a,
	      "d1"_a, "d2"_a, "vmax"_a);

	m.def("forwProject",
	      static_cast<void (*)(const Scanner& scanner, const Image& img,
	                           ProjectionData& projData,
	                           ProjectorType projectorType, bool useGPU)>(
	          &util::forwProject),
	      "scanner"_a, "img"_a, "proj_data"_a,
	      "projectorType"_a = ProjectorType::SIDDON, "use_gpu"_a = false);
	m.def("forwProject",
	      static_cast<void (*)(
	          const Scanner& scanner, const Image& img,
	          ProjectionData& projData, const BinIterator& binIterator,
	          ProjectorType projectorType, bool useGPU)>(&util::forwProject),
	      "scanner"_a, "img"_a, "proj_data"_a, "bin_iter"_a,
	      "projector_type"_a = ProjectorType::SIDDON, "use_gpu"_a = false);
	m.def("forwProject",
	      static_cast<void (*)(const Image& img, ProjectionData& projData,
	                           const ProjectorParams& projParams,
	                           const BinIterator& binIter, bool useGPU)>(
	          &util::forwProject),
	      "img"_a, "proj_data"_a, "proj_params"_a, "bin_iter"_a,
	      "use_gpu"_a = false);

	m.def("backProject",
	      static_cast<void (*)(const Scanner& scanner, Image& img,
	                           const ProjectionData& projData,
	                           ProjectorType projectorType, bool useGPU)>(
	          &util::backProject),
	      "scanner"_a, "img"_a, "proj_data"_a,
	      "projector_type"_a = ProjectorType::SIDDON, "use_gpu"_a = false);
	m.def("backProject",
	      static_cast<void (*)(
	          const Scanner& scanner, Image& img,
	          const ProjectionData& projData, const BinIterator& binIterator,
	          ProjectorType projectorType, bool useGPU)>(&util::backProject),
	      "scanner"_a, "img"_a, "proj_data"_a, "bin_iter"_a,
	      "projector_type"_a = ProjectorType::SIDDON, "use_gpu"_a = false);
	m.def("backProject",
	      static_cast<void (*)(Image& img, const ProjectionData& projData,
	                           const ProjectorParams& projParams,
	                           const BinIterator& binIter, bool useGPU)>(
	          &util::backProject),
	      "img"_a, "proj_data"_a, "proj_params"_a, "bin_iter"_a,
	      "use_gpu"_a = false);
}
}  // namespace yrt

#endif

namespace yrt::util
{

void histogram3DToListModeLUT(const Histogram3D* histo, ListModeLUTOwned* lmOut,
                              size_t numEvents)
{
	ASSERT(lmOut != nullptr);
	const float* dataPtr = histo->getData().getRawPointer();

	// Phase 1: calculate sum of histogram values
	double sum = 0.0;
	std::atomic_ref<double> sumRef(sum);
	util::parallelForChunked(histo->count(), globals::getNumThreads(),
	                         [dataPtr, &sumRef](bin_t binId, size_t /*tid*/)
	                         { sumRef.fetch_add(dataPtr[binId]); });

	// Default target number of events (histogram sum)
	if (numEvents == 0)
	{
		numEvents = std::lround(sum);
	}
	// Phase 2: calculate actual number of events
	size_t sumInt = 0.0;
	std::atomic_ref<size_t> sumIntRef(sumInt);
	util::parallelForChunked(
	    histo->count(), globals::getNumThreads(),
	    [dataPtr, sum, numEvents, &sumIntRef](bin_t binId, size_t /*tid*/)
	    {
		    sumIntRef.fetch_add(
		        std::lround(dataPtr[binId] / sum * (double)numEvents));
	    });

	// Allocate list-mode data
	lmOut->allocate(sumInt);

	int numThreads = globals::getNumThreads();
	if (numThreads > 1)
	{
		size_t numBinsPerThread =
		    std::ceil(double(histo->count()) / (double)numThreads);
		Array1DOwned<size_t> partialSums;
		partialSums.allocate(numThreads);

		// Phase 3: prepare partial sums for parallelization
		util::parallelDoIndexed(
		    numThreads,
		    [numBinsPerThread, histo, &partialSums, dataPtr, sum,
		     numEvents](int ti)
		    {
			    bin_t binStart = ti * numBinsPerThread;
			    bin_t binEnd = std::min(histo->count() - 1,
			                            binStart + numBinsPerThread - 1);
			    for (bin_t binId = binStart; binId <= binEnd; binId++)
			    {
				    partialSums[ti] +=
				        std::lround(dataPtr[binId] / sum * (double)numEvents);
			    }
		    });

		// Calculate indices
		Array1DOwned<size_t> lmStartIdx;
		lmStartIdx.allocate(numThreads);
		lmStartIdx[0] = 0;
		for (int ti = 1; ti < numThreads; ti++)
		{
			lmStartIdx[ti] = lmStartIdx[ti - 1] + partialSums[ti - 1];
		}

		// Phase 4: create events by block
		util::parallelDoIndexed(
		    numThreads,
		    [numBinsPerThread, histo, &lmStartIdx, dataPtr, sum, numEvents,
		     lmOut](int ti)
		    {
			    bin_t binStart = ti * numBinsPerThread;
			    bin_t binEnd = std::min(histo->count() - 1,
			                            binStart + numBinsPerThread - 1);
			    bin_t eventId = lmStartIdx[ti];
			    for (bin_t binId = binStart; binId <= binEnd; binId++)
			    {
				    if (dataPtr[binId] != 0.f)
				    {
					    auto [d1, d2] = histo->getDetectorPair(binId);
					    int numEventsBin = std::lround(dataPtr[binId] / sum *
					                                   (double)numEvents);
					    for (int ei = 0; ei < numEventsBin; ei++)
					    {
						    lmOut->setDetectorIdsOfEvent(eventId++, d1, d2);
					    }
				    }
			    }
		    });
	}
	else
	{
		bin_t eventId = 0;
		for (bin_t binId = 0; binId < histo->count(); binId++)
		{
			if (dataPtr[binId] != 0.f)
			{
				auto [d1, d2] = histo->getDetectorPair(binId);
				int numEventsBin =
				    std::lround(dataPtr[binId] / sum * (double)numEvents);
				for (int ei = 0; ei < numEventsBin; ei++)
				{
					lmOut->setDetectorIdsOfEvent(eventId++, d1, d2);
				}
			}
		}
	}
}

size_t compareListModes(const ListMode& lm1, const ListMode& lm2)
{
	const size_t numEvents = lm1.count();

	const bool hasTOF = lm1.hasTOF();
	const bool hasRandoms = lm1.hasRandomsEstimates();
	const bool hasMotion = lm1.hasMotion();
	const bool hasArbitraryLORs = lm1.hasArbitraryLORs();
	const bool isUniform = lm1.isUniform();

	ASSERT_MSG(numEvents == lm2.count(),
	           "The two listmodes given do not have the same number of events");
	ASSERT_MSG(hasTOF == lm2.hasTOF(),
	           "One given list-mode has TOF while the other one doesn't");
	ASSERT_MSG(hasRandoms == lm2.hasRandomsEstimates(),
	           "One given list-mode has randoms estimates while the other one "
	           "doesn't");
	ASSERT_MSG(hasMotion == lm2.hasMotion(),
	           "One given list-mode is bound to motion information while the "
	           "other one isn't");
	ASSERT_MSG(
	    hasArbitraryLORs == lm2.hasArbitraryLORs(),
	    "One given list-mode has arbitrary LORs while the other one doesn't");
	ASSERT_MSG(isUniform == lm2.isUniform(),
	           "One given list-mode is uniform while the other one isn't");

	ASSERT_MSG(!hasArbitraryLORs,
	           "This function does not support list-modes with arbitrary LORs");

	const int numThreads = globals::getNumThreads();
	std::vector<size_t> numMismatchesPerThread(numThreads, 0ull);

	std::set<ProjectionPropertyType> variables;
	variables.insert(ProjectionPropertyType::TIMESTAMP);
	variables.insert(ProjectionPropertyType::DET_ID);
	if (hasTOF)
	{
		variables.insert(ProjectionPropertyType::TOF);
	}
	if (hasRandoms)
	{
		variables.insert(ProjectionPropertyType::RANDOMS_ESTIMATE);
	}

	const ProjectionPropertyManager propManager(variables);
	const auto props = propManager.createDataArray(2 * numThreads);
	PropertyUnit* props_ptr = props.get();

	parallelForChunked(
	    numEvents, numThreads,
	    [&](size_t evId, unsigned int threadId)
	    {
		    const size_t lm1Index = 2 * threadId + 0;
		    const size_t lm2Index = 2 * threadId + 1;

		    // Gather properties
		    lm1.collectProjectionProperties(propManager, props_ptr, lm1Index,
		                                    evId);
		    lm2.collectProjectionProperties(propManager, props_ptr, lm2Index,
		                                    evId);

		    // Check timestamp
		    const timestamp_t timestamp1 =
		        propManager.getDataValue<timestamp_t>(
		            props_ptr, lm1Index, ProjectionPropertyType::TIMESTAMP);
		    const timestamp_t timestamp2 =
		        propManager.getDataValue<timestamp_t>(
		            props_ptr, lm2Index, ProjectionPropertyType::TIMESTAMP);
		    if (timestamp1 != timestamp2)
		    {
			    numMismatchesPerThread[threadId]++;
			    return;
		    }

		    // Check detector pair
		    const det_pair_t detPair1 = propManager.getDataValue<det_pair_t>(
		        props_ptr, lm1Index, ProjectionPropertyType::DET_ID);
		    const det_pair_t detPair2 = propManager.getDataValue<det_pair_t>(
		        props_ptr, lm2Index, ProjectionPropertyType::DET_ID);
		    if (detPair1.d1 != detPair2.d1 || detPair1.d2 != detPair2.d2)
		    {
			    numMismatchesPerThread[threadId]++;
			    return;
		    }

		    // Check TOF if needed
		    if (hasTOF)
		    {
			    const float tof1 = propManager.getDataValue<float>(
			        props_ptr, lm1Index, ProjectionPropertyType::TOF);
			    const float tof2 = propManager.getDataValue<float>(
			        props_ptr, lm2Index, ProjectionPropertyType::TOF);
			    if (tof1 != tof2)
			    {
				    numMismatchesPerThread[threadId]++;
				    return;
			    }
		    }

		    // Check randoms estimates if needed
		    if (hasRandoms)
		    {
			    const float randoms1 = propManager.getDataValue<float>(
			        props_ptr, lm1Index,
			        ProjectionPropertyType::RANDOMS_ESTIMATE);
			    const float randoms2 = propManager.getDataValue<float>(
			        props_ptr, lm2Index,
			        ProjectionPropertyType::RANDOMS_ESTIMATE);
			    if (randoms1 != randoms2)
			    {
				    numMismatchesPerThread[threadId]++;
				    return;
			    }
		    }
	    });

	// Sum each thread's sum
	size_t numMismatches = 0ull;
	for (const size_t numMismatchInThread : numMismatchesPerThread)
	{
		numMismatches += numMismatchInThread;
	}
	return numMismatches;
}

std::tuple<timestamp_t, timestamp_t>
    getFullTimeRange(const LORMotion& lorMotion)
{
	const size_t numFrames = lorMotion.getNumFrames();
	ASSERT(numFrames > 0);
	const timestamp_t startingTimestampFirstFrame =
	    lorMotion.getStartingTimestamp(0);

	if (numFrames == 1)
	{
		// If the user provides only one frame, then only one transformation
		// will be applied. The weight of that frame will be 1 out of 1.
		return {startingTimestampFirstFrame, startingTimestampFirstFrame + 1};
	}

	const frame_t lastFrame = numFrames - 1;
	const timestamp_t startingTimestampLastFrame =
	    lorMotion.getStartingTimestamp(lastFrame);

	// We add the duration of the second-to-last frame to take into account
	// the duration of the last frame (which is unknown)
	const timestamp_t duration2ndToLastFrame =
	    startingTimestampLastFrame -
	    lorMotion.getStartingTimestamp(lastFrame - 1);

	return {startingTimestampFirstFrame,
	        startingTimestampLastFrame + duration2ndToLastFrame};
}

template <bool PrintProgress>
std::unique_ptr<ImageOwned> timeAverageMoveImage(const LORMotion& lorMotion,
                                                 const Image* unmovedImage)
{
	auto [timeStart, timeStop] = getFullTimeRange(lorMotion);
	return timeAverageMoveImage<PrintProgress>(lorMotion, unmovedImage,
	                                           timeStart, timeStop);
}
template std::unique_ptr<ImageOwned>
    timeAverageMoveImage<true>(const LORMotion&, const Image*);
template std::unique_ptr<ImageOwned>
    timeAverageMoveImage<false>(const LORMotion&, const Image*);

template <bool PrintProgress>
void timeAverageMoveImage(const LORMotion& lorMotion, const Image* unmovedImage,
                          Image* outImage, frame_t outDynamicFrame)
{
	auto [timeStart, timeStop] = getFullTimeRange(lorMotion);
	timeAverageMoveImage<PrintProgress>(lorMotion, unmovedImage, outImage,
	                                    timeStart, timeStop, outDynamicFrame);
}
template void timeAverageMoveImage<true>(const LORMotion&, const Image*, Image*,
                                         frame_t);
template void timeAverageMoveImage<false>(const LORMotion&, const Image*,
                                          Image*, frame_t);

template <bool PrintProgress>
std::unique_ptr<ImageOwned>
    timeAverageMoveImage(const LORMotion& lorMotion, const Image* unmovedImage,
                         timestamp_t timeStart, timestamp_t timeStop)
{
	ASSERT_MSG(unmovedImage != nullptr, "Null input image given");
	const ImageParams& params = unmovedImage->getParams();
	ASSERT_MSG(params.isValid(), "Image parameters incomplete");
	ASSERT_MSG(unmovedImage->isMemoryValid(),
	           "Sensitivity image given is not allocated");

	auto outImage = std::make_unique<ImageOwned>(params);
	outImage->allocate();

	timeAverageMoveImage<PrintProgress>(lorMotion, unmovedImage, outImage.get(),
	                                    timeStart, timeStop, 0);

	return outImage;
}
template std::unique_ptr<ImageOwned>
    timeAverageMoveImage<true>(const LORMotion&, const Image*, timestamp_t,
                               timestamp_t);
template std::unique_ptr<ImageOwned>
    timeAverageMoveImage<false>(const LORMotion&, const Image*, timestamp_t,
                                timestamp_t);

template <bool PrintProgress>
void timeAverageMoveImage(const LORMotion& lorMotion, const Image* unmovedImage,
                          Image* outImage, timestamp_t timeStart,
                          timestamp_t timeStop, frame_t outDynamicFrame)
{
	ASSERT_MSG(unmovedImage != nullptr, "Null input image given");
	ASSERT_MSG(outImage->isMemoryValid(), "Output image not allocated");

	const int64_t numFrames = lorMotion.getNumFrames();
	const auto scanDuration = static_cast<float>(timeStop - timeStart);

	ProgressDisplay progress{numFrames};

	// TODO: Consider edge case:
	//  timeStart precedes the first frame's start time, therefore, we must
	//  add an *unmoved* image that has a weight scaled by:
	//  <time between timeStart and lorMotion.getStartingTimestamp(0)>/
	//  scanDuration
	//  This would be done in order to take into account the cases
	//  when the camera has been started after the scan start.

	for (frame_t frame = 0; frame < numFrames; frame++)
	{
		if constexpr (PrintProgress)
		{
			progress.progress(frame);
		}

		const timestamp_t startingTimestamp =
		    lorMotion.getStartingTimestamp(frame);
		if (startingTimestamp >= timeStart)
		{
			if (startingTimestamp > timeStop)
			{
				break;
			}
			transform_t transform = lorMotion.getTransform(frame);
			const float weight = lorMotion.getDuration(frame) / scanDuration;
			unmovedImage->transformImage(transform, *outImage, weight,
			                             outDynamicFrame);
		}
	}
}
template void timeAverageMoveImage<true>(const LORMotion&, const Image*, Image*,
                                         timestamp_t, timestamp_t, frame_t);
template void timeAverageMoveImage<false>(const LORMotion&, const Image*,
                                          Image*, timestamp_t, timestamp_t,
                                          frame_t);


template <bool PrintProgress>
std::unique_ptr<ImageOwned>
    timeAverageMoveImageDynamic(const LORMotion& lorMotion,
                                const Image* unmovedImage,
                                const DynamicFraming& dynamicFraming)
{
	ASSERT_MSG(unmovedImage != nullptr, "Null input image given");

	ImageParams params = unmovedImage->getParams();
	params.nt = dynamicFraming.getNumFrames();

	auto outImage = std::make_unique<ImageOwned>(params);
	outImage->allocate();

	timeAverageMoveImageDynamic<PrintProgress>(lorMotion, unmovedImage,
	                                           outImage.get(), dynamicFraming);

	return outImage;
}
template std::unique_ptr<ImageOwned>
    timeAverageMoveImageDynamic<true>(const LORMotion&, const Image*,
                                      const DynamicFraming&);
template std::unique_ptr<ImageOwned>
    timeAverageMoveImageDynamic<false>(const LORMotion&, const Image*,
                                       const DynamicFraming&);

template <bool PrintProgress>
void timeAverageMoveImageDynamic(const LORMotion& lorMotion,
                                 const Image* unmovedImage, Image* outImage,
                                 const DynamicFraming& dynamicFraming)
{
	ASSERT_MSG(unmovedImage != nullptr, "Null input image given");
	ASSERT_MSG(outImage != nullptr, "Output image given is null");

	const frame_t numDynamicFrames =
	    static_cast<frame_t>(dynamicFraming.getNumFrames());

	ASSERT_MSG(outImage->getNumFrames() == numDynamicFrames,
	           "Output image does not have the same number of frames as the "
	           "given dynamic framing.");

	for (frame_t dynamicFrame = 0; dynamicFrame < numDynamicFrames;
	     dynamicFrame++)
	{
		const timestamp_t dynamicFrameStart =
		    dynamicFraming.getStartingTimestamp(dynamicFrame);
		const timestamp_t dynamicFrameStop =
		    dynamicFraming.getStoppingTimestamp(dynamicFrame);
		timeAverageMoveImage<PrintProgress>(lorMotion, unmovedImage, outImage,
		                                    dynamicFrameStart, dynamicFrameStop,
		                                    dynamicFrame);
	}
}
template void timeAverageMoveImageDynamic<true>(const LORMotion&, const Image*,
                                                Image*, const DynamicFraming&);
template void timeAverageMoveImageDynamic<false>(const LORMotion&, const Image*,
                                                 Image*, const DynamicFraming&);

// Helper function
template <bool RequiresAtomicAccumulation, bool UseDetectorMask,
          bool PrintProgress>
void convertToHistogram3DInternal(const ProjectionData& dat,
                                  Histogram3D& histoOut,
                                  const DetectorMask* detectorMask)
{
	float* histoDataPointer = histoOut.getData().getRawPointer();
	const size_t numDatBins = dat.count();

	if constexpr (UseDetectorMask)
	{
		ASSERT(detectorMask != nullptr);
	}

	ProgressDisplay progressBar(numDatBins, 5);

	const Histogram3D* histoOut_constptr = &histoOut;
	const ProjectionData* dat_constptr = &dat;
	util::parallelForChunked(
	    numDatBins, globals::getNumThreads(),
	    [&progressBar, dat_constptr, histoOut_constptr, histoDataPointer,
	     detectorMask](bin_t datBin, size_t tid)
	    {
		    if constexpr (PrintProgress)
		    {
			    if (tid == 0)
			    {
				    progressBar.progress(datBin);
			    }
		    }

		    const float projValue = dat_constptr->getProjectionValue(datBin);
		    if (projValue > 0)
		    {
			    const auto [d1, d2] = dat_constptr->getDetectorPair(datBin);
			    if (d1 == d2)
			    {
				    // Do not crash
				    return;
			    }

			    if constexpr (UseDetectorMask)
			    {
				    bool skipEvent = false;
				    skipEvent |= detectorMask->checkDetector(d1);
				    skipEvent |= detectorMask->checkDetector(d2);

				    if (skipEvent)
				    {
					    // Continue to next event
					    return;
				    }
			    }

			    const bin_t histoBin =
			        histoOut_constptr->getBinIdFromDetPair(d1, d2);
			    if constexpr (RequiresAtomicAccumulation)
			    {
				    std::atomic_ref<float> outRef(histoDataPointer[histoBin]);
				    outRef += projValue;
			    }
			    else
			    {
				    histoDataPointer[histoBin] += projValue;
			    }
		    }
	    });
}

template <bool RequiresAtomic, bool PrintProgress>
void convertToHistogram3D(const ProjectionData& pr_dat,
                          Histogram3D& pr_histoOut,
                          const DetectorMask* pp_detectorMask)
{
	ASSERT_MSG(pr_dat.getScanner().getNumDets() ==
	               pr_histoOut.getScanner().getNumDets(),
	           "The projection-space dataset and the histogram provided point "
	           "to scanners with a different number of detectors");

	auto detMask =
	    std::make_unique<DetectorMask>(pr_dat.getScanner().getNumDets());
	if (pp_detectorMask != nullptr)
	{
		detMask->logicalAndWithOther(*pp_detectorMask);
	}

	const auto detectorSetup = pr_dat.getScanner().getDetectorSetup();
	if (detectorSetup->hasMask())
	{
		detMask->logicalAndWithOther(detectorSetup->getMask());
	}

	if (detMask->areAllDetectorsEnabled())
	{
		// Do not use a detector mask
		convertToHistogram3DInternal<RequiresAtomic, false, PrintProgress>(
		    pr_dat, pr_histoOut, nullptr);
	}
	else
	{
		convertToHistogram3DInternal<RequiresAtomic, true, PrintProgress>(
		    pr_dat, pr_histoOut, detMask.get());
	}
}
template void convertToHistogram3D<true, true>(const ProjectionData&,
                                               Histogram3D&,
                                               const DetectorMask*);
template void convertToHistogram3D<false, true>(const ProjectionData&,
                                                Histogram3D&,
                                                const DetectorMask*);
template void convertToHistogram3D<true, false>(const ProjectionData&,
                                                Histogram3D&,
                                                const DetectorMask*);
template void convertToHistogram3D<false, false>(const ProjectionData&,
                                                 Histogram3D&,
                                                 const DetectorMask*);

template <bool RequiresAtomic, bool PrintProgress>
std::unique_ptr<Histogram3DOwned>
    convertToHistogram3D(const ProjectionData& dat,
                         const DetectorMask* detectorMask)
{
	const Scanner& scanner = dat.getScanner();
	auto histogram3D = std::make_unique<Histogram3DOwned>(scanner);
	histogram3D->allocate();
	histogram3D->clearProjections(0);
	convertToHistogram3D<RequiresAtomic, PrintProgress>(dat, *histogram3D,
	                                                    detectorMask);
	return histogram3D;
}
template std::unique_ptr<Histogram3DOwned>
    convertToHistogram3D<true, true>(const ProjectionData& dat,
                                     const DetectorMask* detectorMask);
template std::unique_ptr<Histogram3DOwned>
    convertToHistogram3D<false, true>(const ProjectionData& dat,
                                      const DetectorMask* detectorMask);
template std::unique_ptr<Histogram3DOwned>
    convertToHistogram3D<true, false>(const ProjectionData& dat,
                                      const DetectorMask* detectorMask);
template std::unique_ptr<Histogram3DOwned>
    convertToHistogram3D<false, false>(const ProjectionData& dat,
                                       const DetectorMask* detectorMask);

template <bool PrintProgress>
std::unique_ptr<ListModeLUTOwned>
    convertToListModeLUT(const ListMode& lm, const DetectorMask* detectorMask)
{
	const Scanner& scanner = lm.getScanner();
	const bool hasTOF = lm.hasTOF();
	const bool hasRandoms = lm.hasRandomsEstimates();

	auto lmOut =
	    std::make_unique<ListModeLUTOwned>(scanner, hasTOF, hasRandoms);
	const size_t numEvents = lm.count();
	ASSERT(numEvents > 0);
	lmOut->allocate(numEvents);

	ProgressDisplay progressBar(numEvents, 5);

	util::parallelForChunked(
	    numEvents, globals::getNumThreads(),
	    [&progressBar, &lmOut, &lm, detectorMask, hasTOF,
	     hasRandoms](size_t evId, size_t tid)
	    {
		    if constexpr (PrintProgress)
		    {
			    if (tid == 0)
			    {
				    progressBar.progress(evId);
			    }
		    }

		    lmOut->setTimestampOfEvent(evId, lm.getTimestamp(evId));
		    auto [d1, d2] = lm.getDetectorPair(evId);

		    if (detectorMask != nullptr)
		    {
			    bool skipEvent = false;
			    skipEvent |= detectorMask->checkDetector(d1);
			    skipEvent |= detectorMask->checkDetector(d2);

			    if (skipEvent)
			    {
				    // Put 0,0 as detector pair to disable event
				    lmOut->setDetectorIdsOfEvent(evId, 0, 0);

				    // Go to next event
				    return;
			    }
		    }

		    lmOut->setDetectorIdsOfEvent(evId, d1, d2);
		    if (hasTOF)
		    {
			    lmOut->setTOFValueOfEvent(evId, lm.getTOFValue(evId));
		    }
		    if (hasRandoms)
		    {
			    lmOut->setRandomsEstimateOfEvent(evId,
			                                     lm.getRandomsEstimate(evId));
		    }
	    });
	return lmOut;
}
template std::unique_ptr<ListModeLUTOwned>
    convertToListModeLUT<true>(const ListMode& lm,
                               const DetectorMask* detectorMask);
template std::unique_ptr<ListModeLUTOwned>
    convertToListModeLUT<false>(const ListMode& lm,
                                const DetectorMask* detectorMask);

Line3D getNativeLOR(const Scanner& scanner, const ProjectionData& dat,
                    bin_t binId)
{
	const auto [d1, d2] = dat.getDetectorPair(binId);
	const Vector3D p1 = scanner.getDetectorPos(d1);
	const Vector3D p2 = scanner.getDetectorPos(d2);
	return Line3D{p1, p2};
}

void convertProjectionValuesToACF(ProjectionData& dat, float unitFactor)
{
	dat.operationOnEachBinParallel(
	    [&dat, unitFactor](bin_t bin) -> float
	    {
		    return util::getAttenuationCoefficientFactor(
		        dat.getProjectionValue(bin), unitFactor);
	    });
}

std::tuple<Line3D, Vector3D, Vector3D>
    generateTORRandomDOI(const Scanner& scanner, det_id_t d1, det_id_t d2,
                         int vmax)
{
	const Vector3D p1 = scanner.getDetectorPos(d1);
	const Vector3D p2 = scanner.getDetectorPos(d2);
	const Vector3D n1 = scanner.getDetectorOrient(d1);
	const Vector3D n2 = scanner.getDetectorOrient(d2);
	const float doi_1_t = (rand() % vmax) / (static_cast<float>(1 << 8) - 1) *
	                      scanner.crystalDepth;
	const float doi_2_t = (rand() % vmax) / (static_cast<float>(1 << 8) - 1) *
	                      scanner.crystalDepth;
	const Vector3D p1_doi{p1.x + doi_1_t * n1.x, p1.y + doi_1_t * n1.y,
	                      p1.z + doi_1_t * n1.z};
	const Vector3D p2_doi{p2.x + doi_2_t * n2.x, p2.y + doi_2_t * n2.y,
	                      p2.z + doi_2_t * n2.z};
	const Line3D lorDOI{p1_doi, p2_doi};
	return {lorDOI, n1, n2};
}

std::unique_ptr<OSEM> createOSEM(const Scanner& scanner, bool useGPU,
                                 bool isLowRank)
{
	std::unique_ptr<OSEM> osem;
	if (useGPU)
	{
#if BUILD_CUDA
		osem = createOSEM_GPU(scanner, isLowRank);
#else
		throw std::runtime_error(
		    "Project not compiled with CUDA, GPU projectors unavailable.");
#endif
	}
	else
	{
		osem = createOSEM_CPU(scanner, isLowRank);
	}
	return osem;
}

std::unique_ptr<OSEM> createOSEM_CPU(const Scanner& scanner, bool isLowRank)
{
	if (!isLowRank)
	{
		return std::make_unique<OSEM_CPU>(scanner);
	}
	else
	{
		return std::make_unique<LREM_CPU>(scanner);
	}
}

// Forward and backward projections
template <bool IS_FWD>
static void project(Image* img, ProjectionData* projData,
                    const ProjectorParams& projParams,
                    const BinIterator& binIter, bool useGPU)
{
#ifdef BUILD_CUDA
	std::unique_ptr<GPUStream> mainStream;
	std::unique_ptr<GPUStream> auxStream;
#endif
	if (useGPU)
	{
#ifdef BUILD_CUDA
		mainStream = std::make_unique<GPUStream>();
		auxStream = std::make_unique<GPUStream>();
#else
		throw std::runtime_error("GPU is not supported because project was "
		                         "not compiled with CUDA");
#endif
	}
	std::unique_ptr<OperatorProjectorBase> oper;
	std::vector<std::unique_ptr<Constraint>> constraints;
	projData->getScanner().collectConstraints(constraints);
	std::vector<Constraint*> constraintsPtr;
	for (auto& constraint : constraints)
	{
		constraintsPtr.emplace_back(constraint.get());
	}

	if (useGPU)
	{
#if BUILD_CUDA
		oper = createOperatorProjectorDevice(
		    projParams, binIter, constraintsPtr, &mainStream->getStream(),
		    &auxStream->getStream());
#else
		throw std::runtime_error("GPU is not supported because project was "
		                         "not compiled with CUDA");
#endif
	}
	else
	{
		oper = std::make_unique<OperatorProjector>(projParams, &binIter,
		                                           constraintsPtr);
	}

	if constexpr (IS_FWD)
	{
		std::cout << "Forward projecting all LORs ..." << std::endl;
		oper->applyA(img, projData);
	}
	else
	{
		std::cout << "Backprojecting all LORs ..." << std::endl;
		oper->applyAH(projData, img);
	}
}

void forwProject(const Scanner& scanner, const Image& img,
                 ProjectionData& projData, ProjectorType projectorType,
                 bool useGPU)
{
	const auto binIter = projData.getBinIter(1, 0);
	ProjectorParams projParams(scanner);
	projParams.projectorType = projectorType;

	forwProject(img, projData, projParams, *binIter, useGPU);
}

void forwProject(const Scanner& scanner, const Image& img,
                 ProjectionData& projData, const BinIterator& binIter,
                 ProjectorType projectorType, bool useGPU)
{
	ProjectorParams projParams(scanner);
	projParams.projectorType = projectorType;

	forwProject(img, projData, projParams, binIter, useGPU);
}

void forwProject(const Image& img, ProjectionData& projData,
                 const ProjectorParams& projParams, const BinIterator& binIter,
                 bool useGPU)
{
	project<true>(const_cast<Image*>(&img), &projData, projParams, binIter,
	              useGPU);
}

void backProject(const Scanner& scanner, Image& img,
                 const ProjectionData& projData, ProjectorType projectorType,
                 bool useGPU)
{
	const auto binIter = projData.getBinIter(1, 0);
	ProjectorParams projParams(scanner);
	projParams.projectorType = projectorType;

	backProject(img, projData, projParams, *binIter, useGPU);
}

void backProject(const Scanner& scanner, Image& img,
                 const ProjectionData& projData, const BinIterator& binIterator,
                 ProjectorType projectorType, bool useGPU)
{
	ProjectorParams projParams(scanner);
	projParams.projectorType = projectorType;

	backProject(img, projData, projParams, binIterator, useGPU);
}

void backProject(Image& img, const ProjectionData& projData,
                 const ProjectorParams& projParams,
                 const BinIterator& binIterator, bool useGPU)
{
	project<false>(&img, const_cast<ProjectionData*>(&projData), projParams,
	               binIterator, useGPU);
}

}  // namespace yrt::util
