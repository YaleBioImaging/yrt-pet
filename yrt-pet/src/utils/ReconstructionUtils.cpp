/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/ReconstructionUtils.hpp"

#include "datastruct/IO.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "geometry/Matrix.hpp"
#include "operators/OperatorProjectorDD.hpp"
#include "operators/OperatorProjectorSiddon.hpp"
#include "recon/OSEM_CPU.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ProgressDisplay.hpp"
#include "utils/ProgressDisplayMultiThread.hpp"
#include "utils/Tools.hpp"

#if BUILD_CUDA
#include "operators/OperatorProjectorDD_GPU.cuh"
#include "operators/OperatorProjectorSiddon_GPU.cuh"
#include "recon/OSEM_GPU.cuh"
#endif


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

void py_setup_reconstructionutils(pybind11::module& m)
{
	m.def("histogram3DToListModeLUT", &Util::histogram3DToListModeLUT);
	m.def(
	    "convertToHistogram3D", [](const Histogram& dat, Histogram3D& histoOut)
	    { Util::convertToHistogram3D<false>(dat, histoOut); },
	    py::arg("histogramDataInput"), py::arg("histoOut"));
	m.def(
	    "convertToHistogram3D", [](const ListMode& dat, Histogram3D& histoOut)
	    { Util::convertToHistogram3D<true>(dat, histoOut); },
	    py::arg("listmodeDataInput"), py::arg("histoOut"));
	m.def(
	    "createOSEM",
	    [](const Scanner& scanner, bool useGPU)
	    {
		    auto osem = Util::createOSEM(scanner, useGPU);
		    osem->enableNeedToMakeCopyOfSensImage();
		    return osem;
	    },
	    "scanner"_a, "useGPU"_a = false);

	m.def("timeAverageMoveImage",
	      static_cast<std::unique_ptr<ImageOwned> (*)(
	          const LORMotion&, const Image&)>(&Util::timeAverageMoveImage),
	      "lorMotion"_a, "unmovedSensImage"_a,
	      "Blur a given image based on given motion information");

	m.def("timeAverageMoveImage",
	      static_cast<std::unique_ptr<ImageOwned> (*)(
	          const LORMotion&, const Image&, timestamp_t, timestamp_t)>(
	          &Util::timeAverageMoveImage),
	      "lorMotion"_a, "unmovedSensImage"_a, "timeStart"_a, "timeStop"_a,
	      "Blur a given image based on given motion information");


	m.def("generateTORRandomDOI", &Util::generateTORRandomDOI, "scanner"_a,
	      "d1"_a, "d2"_a, "vmax"_a);

	m.def(
	    "forwProject",
	    static_cast<void (*)(
	        const Scanner& scanner, const Image& img, ProjectionData& projData,
	        OperatorProjector::ProjectorType projectorType, bool useGPU)>(
	        &Util::forwProject),
	    "scanner"_a, "img"_a, "projData"_a,
	    "projectorType"_a = OperatorProjector::ProjectorType::SIDDON,
	    "useGPU"_a = false);
	m.def("forwProject",
	      static_cast<void (*)(
	          const Scanner& scanner, const Image& img,
	          ProjectionData& projData, const BinIterator& binIterator,
	          OperatorProjector::ProjectorType projectorType, bool useGPU)>(
	          &Util::forwProject),
	      py::arg("scanner"), py::arg("img"), py::arg("projData"),
	      py::arg("binIterator"),
	      py::arg("projectorType") = OperatorProjector::SIDDON,
	      "useGPU"_a = false);
	m.def("forwProject",
	      static_cast<void (*)(const Image& img, ProjectionData& projData,
	                           const OperatorProjectorParams& projParams,
	                           OperatorProjector::ProjectorType projectorType,
	                           bool useGPU)>(&Util::forwProject),
	      "img"_a, "projData"_a, "projParams"_a,
	      "projectorType"_a = OperatorProjector::SIDDON, "useGPU"_a = false);

	m.def(
	    "backProject",
	    static_cast<void (*)(
	        const Scanner& scanner, Image& img, const ProjectionData& projData,
	        OperatorProjector::ProjectorType projectorType, bool useGPU)>(
	        &Util::backProject),
	    "scanner"_a, "img"_a, "projData"_a,
	    "projectorType"_a = OperatorProjector::ProjectorType::SIDDON,
	    "useGPU"_a = false);
	m.def("backProject",
	      static_cast<void (*)(
	          const Scanner& scanner, Image& img,
	          const ProjectionData& projData, const BinIterator& binIterator,
	          OperatorProjector::ProjectorType projectorType, bool useGPU)>(
	          &Util::backProject),
	      "scanner"_a, "img"_a, "projData"_a, "binIterator"_a,
	      "projectorType"_a = OperatorProjector::SIDDON, "useGPU"_a = false);
	m.def("backProject",
	      static_cast<void (*)(Image& img, const ProjectionData& projData,
	                           const OperatorProjectorParams& projParams,
	                           OperatorProjector::ProjectorType projectorType,
	                           bool useGPU)>(&Util::backProject),
	      "img"_a, "projData"_a, "projParams"_a,
	      "projectorType"_a = OperatorProjector::SIDDON, "useGPU"_a = false);
}

#endif

namespace Util
{
	void histogram3DToListModeLUT(const Histogram3D* histo,
	                              ListModeLUTOwned* lmOut, size_t numEvents)
	{
		ASSERT(lmOut != nullptr);
		const float* dataPtr = histo->getData().getRawPointer();

		// Phase 1: calculate sum of histogram values
		double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
		for (bin_t binId = 0; binId < histo->count(); binId++)
		{
			sum += dataPtr[binId];
		}

		// Default target number of events (histogram sum)
		if (numEvents == 0)
		{
			numEvents = std::lround(sum);
		}
		// Phase 2: calculate actual number of events
		size_t sumInt = 0.0;
#pragma omp parallel for reduction(+ : sumInt)
		for (bin_t binId = 0; binId < histo->count(); binId++)
		{
			sumInt += std::lround(dataPtr[binId] / sum * (double)numEvents);
		}

		// Allocate list-mode data
		lmOut->allocate(sumInt);

		int numThreads = Globals::get_num_threads();
		if (numThreads > 1)
		{
			size_t numBinsPerThread =
			    std::ceil(double(histo->count()) / (double)numThreads);
			Array1D<size_t> partialSums;
			partialSums.allocate(numThreads);

			// Phase 3: prepare partial sums for parallelization
#pragma omp parallel for num_threads(numThreads)
			for (int ti = 0; ti < numThreads; ti++)
			{
				bin_t binStart = ti * numBinsPerThread;
				bin_t binEnd = std::min(histo->count() - 1,
				                        binStart + numBinsPerThread - 1);
				for (bin_t binId = binStart; binId <= binEnd; binId++)
				{
					partialSums[ti] +=
					    std::lround(dataPtr[binId] / sum * (double)numEvents);
				}
			}

			// Calculate indices
			Array1D<size_t> lmStartIdx;
			lmStartIdx.allocate(numThreads);
			lmStartIdx[0] = 0;
			for (int ti = 1; ti < numThreads; ti++)
			{
				lmStartIdx[ti] = lmStartIdx[ti - 1] + partialSums[ti - 1];
			}

			// Phase 4: create events by block
#pragma omp parallel for num_threads(numThreads)
			for (int ti = 0; ti < numThreads; ti++)
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
			}
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

	std::unique_ptr<ImageOwned> timeAverageMoveImage(const LORMotion& lorMotion,
	                                                 const Image& unmovedImage)
	{
		const size_t numFrames = lorMotion.getNumFrames();
		ASSERT(numFrames > 0);
		const timestamp_t startingTimestampFirstFrame =
		    lorMotion.getStartingTimestamp(0);

		if (numFrames == 1)
		{
			// If the user provides only one frame, then only one transformation
			// will be applied. The weight of that frame will be 1 out of 1.
			return timeAverageMoveImage(lorMotion, unmovedImage,
			                            startingTimestampFirstFrame,
			                            startingTimestampFirstFrame + 1);
		}

		const frame_t lastFrame = numFrames - 1;
		const timestamp_t startingTimestampLastFrame =
		    lorMotion.getStartingTimestamp(lastFrame);

		// We add the duration of the second-to-last frame to take into account
		// the duration of the last frame (which is unknown)
		const timestamp_t duration2ndToLastFrame =
		    startingTimestampLastFrame -
		    lorMotion.getStartingTimestamp(lastFrame - 1);

		return timeAverageMoveImage(
		    lorMotion, unmovedImage, startingTimestampFirstFrame,
		    startingTimestampLastFrame + duration2ndToLastFrame);
	}

	std::unique_ptr<ImageOwned> timeAverageMoveImage(const LORMotion& lorMotion,
	                                                 const Image& unmovedImage,
	                                                 timestamp_t timeStart,
	                                                 timestamp_t timeStop)
	{
		const ImageParams& params = unmovedImage.getParams();
		ASSERT_MSG(params.isValid(), "Image parameters incomplete");
		ASSERT_MSG(unmovedImage.isMemoryValid(),
		           "Sensitivity image given is not allocated");

		const int64_t numFrames = lorMotion.getNumFrames();
		const auto scanDuration = static_cast<float>(timeStop - timeStart);

		ProgressDisplay progress{numFrames};

		auto movedSensImage = std::make_unique<ImageOwned>(params);
		movedSensImage->allocate();

		for (frame_t frame = 0; frame < numFrames; frame++)
		{
			progress.progress(frame);
			const timestamp_t startingTimestamp =
			    lorMotion.getStartingTimestamp(frame);
			if (startingTimestamp >= timeStart)
			{
				if (startingTimestamp > timeStop)
				{
					break;
				}
				transform_t transform = lorMotion.getTransform(frame);
				const float weight =
				    lorMotion.getDuration(frame) / scanDuration;
				unmovedImage.transformImage(transform, *movedSensImage, weight);
			}
		}

		return movedSensImage;
	}

	template <bool RequiresAtomicAccumulation>
	void convertToHistogram3D(const ProjectionData& dat, Histogram3D& histoOut)
	{
		float* histoDataPointer = histoOut.getData().getRawPointer();
		const size_t numDatBins = dat.count();

		ProgressDisplayMultiThread progressBar(Globals::get_num_threads(),
		                                       numDatBins, 5);

		const Histogram3D* histoOut_constptr = &histoOut;
		const ProjectionData* dat_constptr = &dat;
#pragma omp parallel for default(none)                            \
    firstprivate(histoDataPointer, numDatBins, histoOut_constptr, \
                     dat_constptr) shared(progressBar)
		for (bin_t datBin = 0; datBin < numDatBins; ++datBin)
		{
			progressBar.progress(omp_get_thread_num(), 1);

			const float projValue = dat_constptr->getProjectionValue(datBin);
			if (projValue > 0)
			{
				const auto [d1, d2] = dat_constptr->getDetectorPair(datBin);
				if (d1 == d2)
				{
					// Do not crash
					continue;
				}
				const bin_t histoBin =
				    histoOut_constptr->getBinIdFromDetPair(d1, d2);
				if constexpr (RequiresAtomicAccumulation)
				{
#pragma omp atomic
					histoDataPointer[histoBin] += projValue;
				}
				else
				{
					histoDataPointer[histoBin] += projValue;
				}
			}
		}
	}

	template void convertToHistogram3D<true>(const ProjectionData&,
	                                         Histogram3D&);
	template void convertToHistogram3D<false>(const ProjectionData&,
	                                          Histogram3D&);

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
			    return Util::getAttenuationCoefficientFactor(
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
		const float doi_1_t = (rand() % vmax) /
		                      (static_cast<float>(1 << 8) - 1) *
		                      scanner.crystalDepth;
		const float doi_2_t = (rand() % vmax) /
		                      (static_cast<float>(1 << 8) - 1) *
		                      scanner.crystalDepth;
		const Vector3D p1_doi{p1.x + doi_1_t * n1.x, p1.y + doi_1_t * n1.y,
		                      p1.z + doi_1_t * n1.z};
		const Vector3D p2_doi{p2.x + doi_2_t * n2.x, p2.y + doi_2_t * n2.y,
		                      p2.z + doi_2_t * n2.z};
		const Line3D lorDOI{p1_doi, p2_doi};
		return {lorDOI, n1, n2};
	}

	std::unique_ptr<OSEM> createOSEM(const Scanner& scanner, bool useGPU)
	{
		std::unique_ptr<OSEM> osem;
		if (useGPU)
		{
#if BUILD_CUDA
			osem = std::make_unique<OSEM_GPU>(scanner);
#else
			throw std::runtime_error(
			    "Project not compiled with CUDA, GPU projectors unavailable.");
#endif
		}
		else
		{
			osem = std::make_unique<OSEM_CPU>(scanner);
		}
		return osem;
	}


	// Forward and backward projections
	template <bool IS_FWD>
	static void project(Image* img, ProjectionData* projData,
	                    const OperatorProjectorParams& projParams,
	                    OperatorProjector::ProjectorType projectorType,
	                    bool useGPU)
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
		if (projectorType == OperatorProjector::SIDDON)
		{
			if (useGPU)
			{
#ifdef BUILD_CUDA
				oper = std::make_unique<OperatorProjectorSiddon_GPU>(
				    projParams, &mainStream->getStream(),
				    &auxStream->getStream());
#else
				throw std::runtime_error(
				    "Siddon GPU projector not supported because "
				    "project was not compiled with CUDA");
#endif
			}
			else
			{
				oper = std::make_unique<OperatorProjectorSiddon>(projParams);
			}
		}
		else if (projectorType == OperatorProjector::DD)
		{
			if (useGPU)
			{
#ifdef BUILD_CUDA
				oper = std::make_unique<OperatorProjectorDD_GPU>(
				    projParams, &mainStream->getStream(),
				    &auxStream->getStream());
#else
				throw std::runtime_error(
				    "Distance-driven GPU projector not supported because "
				    "project was not compiled with CUDA");
#endif
			}
			else
			{
				oper = std::make_unique<OperatorProjectorDD>(projParams);
			}
		}
		else
		{
			throw std::runtime_error("Unknown error");
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
	                 ProjectionData& projData,
	                 OperatorProjector::ProjectorType projectorType,
	                 bool useGPU)
	{
		const auto binIter = projData.getBinIter(1, 0);
		const OperatorProjectorParams projParams(binIter.get(), scanner);
		forwProject(img, projData, projParams, projectorType, useGPU);
	}

	void forwProject(const Scanner& scanner, const Image& img,
	                 ProjectionData& projData, const BinIterator& binIterator,
	                 OperatorProjector::ProjectorType projectorType,
	                 bool useGPU)
	{
		const OperatorProjectorParams projParams(&binIterator, scanner);
		forwProject(img, projData, projParams, projectorType, useGPU);
	}

	void forwProject(const Image& img, ProjectionData& projData,
	                 const OperatorProjectorParams& projParams,
	                 OperatorProjector::ProjectorType projectorType,
	                 bool useGPU)
	{
		project<true>(const_cast<Image*>(&img), &projData, projParams,
		              projectorType, useGPU);
	}

	void backProject(const Scanner& scanner, Image& img,
	                 const ProjectionData& projData,
	                 OperatorProjector::ProjectorType projectorType,
	                 bool useGPU)
	{
		const auto binIter = projData.getBinIter(1, 0);
		const OperatorProjectorParams projParams(binIter.get(), scanner);
		backProject(img, projData, projParams, projectorType, useGPU);
	}

	void backProject(const Scanner& scanner, Image& img,
	                 const ProjectionData& projData,
	                 const BinIterator& binIterator,
	                 OperatorProjector::ProjectorType projectorType,
	                 bool useGPU)
	{
		const OperatorProjectorParams projParams(&binIterator, scanner);
		backProject(img, projData, projParams, projectorType, useGPU);
	}

	void backProject(Image& img, const ProjectionData& projData,
	                 const OperatorProjectorParams& projParams,
	                 OperatorProjector::ProjectorType projectorType,
	                 bool useGPU)
	{
		project<false>(&img, const_cast<ProjectionData*>(&projData), projParams,
		               projectorType, useGPU);
	}

}  // namespace Util
