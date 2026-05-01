/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 *
 * PopulateScatterSpace
 *
 * Purpose:
 *   Bins list-mode events into per-frame downsampled "sinograms" stored as
 *   ScatterSpace (.scs) files. The ScatterSpace uses cylindrical coordinates
 *   instead of detector IDs, so its memory footprint is independent of the
 *   scanner's LOR count. The resulting files can be loaded in Python and fed
 *   to an NMF solver to obtain initial temporal bases H for list-mode
 *   low-rank dynamic PET reconstruction.
 *
 * Time windowing:
 *   --time_start, --time_stop, and any other format-specific time options
 *   (e.g. the NX plugin's startTime / stopTime) are forwarded directly to
 *   openProjectionData(), which filters events during loading. The events
 *   returned are therefore already confined to the requested window.
 *   This executable then frames those events with uniform bins of duration
 *   --time_step (in milliseconds), anchored to the timestamp of the first
 *   loaded event.
 *
 * Usage example (10 min of data, 1 s frames):
 *   yrt_populate_scatter_space \
 *       --scanner  scanner.json       \
 *       --input    data.lm            \
 *       --format   LM                 \
 *       --out      frame_scs/frame    \
 *       --time_step 1000              \
 *       --num_planes 20               \
 *       --num_angles 100
 *
 *   (To restrict the window for NX data, pass the plugin's own --startTime /
 *   --stopTime flags; these are forwarded automatically via
 *   addOptionsFromPlugins.)
 *
 * Output:
 *   <out>_frame000.scs, <out>_frame001.scs, …
 *   <out>_manifest.txt  — one output filename per line, for easy Python import
 */

#include "../PluginOptionsHelper.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/scatter/ScatterSpace.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/utils/Version.hpp"

#include <cxxopts.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace yrt;

// ---------------------------------------------------------------------------
// Helper: zero-padded frame index string
// ---------------------------------------------------------------------------
static std::string frameTag(size_t frame, size_t numFrames)
{
	int width = 1;
	size_t tmp = (numFrames > 1) ? numFrames - 1 : 1;
	while (tmp >= 10)
	{
		tmp /= 10;
		width++;
	}
	std::ostringstream ss;
	ss << std::setw(width) << std::setfill('0') << frame;
	return ss.str();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
	try
	{
		io::ArgumentRegistry registry{};

		const std::string coreGroup    = "0. Core";
		const std::string inputGroup   = "1. Input";
		const std::string outputGroup  = "2. Output";
		const std::string framingGroup = "3. Framing";
		const std::string scsGroup     = "4. ScatterSpace";

		// ---- Core ----
		registry.registerArgument(
		    "scanner", "Scanner parameters file", true,
		    io::TypeOfArgument::STRING, "", coreGroup, "s");
		registry.registerArgument(
		    "num_threads", "Number of threads to use (-1 = all available)",
		    false, io::TypeOfArgument::INT, -1, coreGroup);

		// ---- Input ----
		registry.registerArgument(
		    "input", "Input list-mode file", true,
		    io::TypeOfArgument::STRING, "", inputGroup, "i");
		registry.registerArgument(
		    "format",
		    "Input format. Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYLISTMODES),
		    true, io::TypeOfArgument::STRING, "", inputGroup, "f");

		// ---- Output ----
		registry.registerArgument(
		    "out",
		    "Output filename prefix. Files will be written as "
		    "<out>_frame<NNN>.scs and a manifest as <out>_manifest.txt.",
		    true, io::TypeOfArgument::STRING, "", outputGroup, "o");

		// ---- Framing ----
		registry.registerArgument(
		    "time_step",
		    "Frame duration in milliseconds (default: 1000 = 1 second). "
		    "Framing is anchored to the first loaded event's timestamp.",
		    false, io::TypeOfArgument::INT, 1000, framingGroup);

		// ---- ScatterSpace geometry ----
		registry.registerArgument(
		    "num_planes",
		    "Number of axial plane samples in the ScatterSpace "
		    "(default: " +
		        std::to_string(ScatterSpace::RecommendedNumPlanes) + ")",
		    false, io::TypeOfArgument::INT,
		    static_cast<int>(ScatterSpace::RecommendedNumPlanes), scsGroup);
		registry.registerArgument(
		    "num_angles",
		    "Number of angular samples in the ScatterSpace "
		    "(default: " +
		        std::to_string(ScatterSpace::RecommendedNumAngles) + ")",
		    false, io::TypeOfArgument::INT,
		    static_cast<int>(ScatterSpace::RecommendedNumAngles), scsGroup);
		registry.registerArgument(
		    "no_symmetrize",
		    "Don't Symmetrize each frame's ScatterSpace after filling (recommended "
		    "for non-TOF data). Default: false.",
		    false, io::TypeOfArgument::BOOL, false, scsGroup);

		// Forward all plugin-specific options (includes NX startTime/stopTime,
		// useTOF, timeWindow, etc.) so the time window is applied at load time.
		plugin::addOptionsFromPlugins(registry,
		                              plugin::InputFormatsChoice::ONLYLISTMODES);

		// ---- Parse ----
		io::ArgumentReader config{
		    registry,
		    "Convert a list-mode input into per-frame downsampled sinograms "
		    "stored as ScatterSpace files (.scs), for NMF-based initialisation "
		    "of the temporal basis H in list-mode low-rank dynamic PET "
		    "reconstruction."};

		if (!config.loadFromCommandLine(argc, argv))
		{
			return 0;  // --help requested
		}
		if (!config.validate())
		{
			std::cerr << "Invalid configuration. Please check required "
			             "parameters.\n";
			return -1;
		}

		// ---- Read arguments ----
		const auto scanner_fname = config.getValue<std::string>("scanner");
		const auto input_fname   = config.getValue<std::string>("input");
		const auto input_format  = config.getValue<std::string>("format");
		const auto out_prefix    = config.getValue<std::string>("out");
		const int  numThreadsArg = config.getValue<int>("num_threads");
		const int  timeStepArg   = config.getValue<int>("time_step");
		const int  numPlanesArg  = config.getValue<int>("num_planes");
		const int  numAnglesArg  = config.getValue<int>("num_angles");
		const bool doSymmetrize  = !config.getValue<bool>("no_symmetrize");

		ASSERT_MSG(timeStepArg > 0, "time_step must be strictly positive");
		ASSERT_MSG(numPlanesArg > 0, "num_planes must be strictly positive");
		ASSERT_MSG(numAnglesArg >= static_cast<int>(ScatterSpace::MinNumAngles),
		           ("num_angles must be >= " +
		            std::to_string(ScatterSpace::MinNumAngles))
		               .c_str());

		globals::setNumThreads(numThreadsArg);

		// ---- Scanner ----
		std::cout << "Initializing scanner..." << std::endl;
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		// ---- Load list-mode data ----
		// openProjectionData() forwards all plugin options (including any
		// startTime / stopTime for NX) so the returned object already contains
		// only the events inside the requested time window.
		std::cout << "Reading list-mode data..." << std::endl;
		std::unique_ptr<ProjectionData> dataInput = io::openProjectionData(
		    input_fname, input_format, *scanner, config.getAllArguments());

		auto* lm = dynamic_cast<ListMode*>(dataInput.get());
		ASSERT_MSG(lm != nullptr,
		           "The input file does not appear to be list-mode data.");

		const size_t numEvents = lm->count();
		ASSERT_MSG(numEvents > 0,
		           "No events in the loaded data. "
		           "Check the input file and any time-window arguments.");

		std::cout << "Events loaded: " << numEvents << std::endl;

		// ---- Determine frame boundaries ----
		// Anchor framing to the first loaded event. For NXListMode,
		// getTimestamp() returns absolute timetags (ms since midnight); for
		// ListModeLUT they are whatever the file contains. In both cases,
		// subtracting t0 gives a relative offset in the same unit (ms), and
		// the window [t0, tLast] is exactly the data that was loaded.
		const timestamp_t t0       = lm->getTimestamp(0);
		const timestamp_t tLast    = lm->getTimestamp(numEvents - 1);
		const timestamp_t timeStep = static_cast<timestamp_t>(timeStepArg);
		const timestamp_t duration = tLast - t0;

		// Number of frames needed to cover [t0, tLast] inclusive
		const size_t numFrames =
		    static_cast<size_t>(duration / timeStep) + 1;

		const size_t numPlanes = static_cast<size_t>(numPlanesArg);
		const size_t numAngles = static_cast<size_t>(numAnglesArg);

		std::cout << "First event timestamp : " << t0 << " ms\n"
		          << "Last  event timestamp : " << tLast << " ms\n"
		          << "Window duration       : " << duration << " ms\n"
		          << "Frame duration        : " << timeStepArg << " ms\n"
		          << "Number of frames      : " << numFrames << "\n"
		          << "ScatterSpace geometry : "
		          << numPlanes << " planes x " << numAngles
		          << " angles (numTOFBins=1)\n";

		// ---- Allocate per-frame ScatterSpaces ----
		// numTOFBins = 1: TOF is not needed for NMF initialisation
		constexpr size_t numTOFBins = 1;

		std::cout << "Allocating " << numFrames
		          << " ScatterSpace arrays..." << std::endl;

		std::vector<std::unique_ptr<ScatterSpace>> frames;
		frames.reserve(numFrames);
		for (size_t f = 0; f < numFrames; f++)
		{
			auto scs = std::make_unique<ScatterSpace>(*scanner, numTOFBins,
			                                          numPlanes, numAngles);
			scs->allocate();
			frames.push_back(std::move(scs));
		}

		// Raw pointers for the parallel lambda
		std::vector<ScatterSpace*> framePtrs(numFrames);
		for (size_t f = 0; f < numFrames; f++)
		{
			framePtrs[f] = frames[f].get();
		}

		// ---- Bin events into frames ----
		std::cout << "Binning events into frames..." << std::endl;

		const int numThreads = globals::getNumThreads();
		util::ProgressDisplayMultiThread progressBar(numThreads, numEvents, 5);

		util::parallelForChunked(
		    numEvents, numThreads,
		    [&](size_t evId, size_t threadId)
		    {
			    progressBar.incrementProgress(threadId);

			    // Frame index from scan-relative timestamp
			    const timestamp_t relTs = lm->getTimestamp(evId) - t0;
			    const size_t frameIdx =
			        static_cast<size_t>(relTs / timeStep);

			    // Safety guard (should not trigger given allocation above)
			    if (frameIdx >= numFrames)
			    {
				    return;
			    }

			    // getLOR() applies any attached motion correction automatically
			    const Line3D lor = lm->getLOR(evId);

			    // Convert LOR to cylindrical coordinates
			    float planePos1, angle1, planePos2, angle2;
			    ScatterSpace::computeCylindricalCoordinates(
			        lor, planePos1, angle1, planePos2, angle2);

			    // tof_ps = 0 (no TOF)
			    const ScatterSpace::ScatterSpacePosition pos{
			        0.0f, planePos1, angle1, planePos2, angle2};

			    const ScatterSpace::ScatterSpaceIndex idx =
			        framePtrs[frameIdx]->getNearestNeighborIndex(pos);

			    // Atomic increment for thread safety
			    framePtrs[frameIdx]->incrementValueAtomic(idx, 1.0f);
		    });

		// ---- Symmetrize and save ----
		std::cout << "\nSymmetrizing and saving frames..." << std::endl;

		const std::string manifestPath = out_prefix + "_manifest.txt";
		std::ofstream manifest(manifestPath);
		ASSERT_MSG(manifest.is_open(),
		           ("Cannot open manifest file: " + manifestPath).c_str());

		for (size_t f = 0; f < numFrames; f++)
		{
			if (doSymmetrize)
			{
				frames[f]->symmetrizeIfNeeded();
			}

			const std::string fname =
			    out_prefix + "_frame" + frameTag(f, numFrames) + ".scs";

			std::cout << "  Writing frame " << f + 1 << "/" << numFrames
			          << " -> " << fname << std::endl;
			frames[f]->writeToFile(fname);
			manifest << fname << "\n";
		}
		manifest.close();

		std::cout << "Manifest written to: " << manifestPath << "\n";
		std::cout << "Done." << std::endl;
		return 0;
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << "\n";
		return -1;
	}
	catch (const std::exception& e)
	{
		util::printExceptionMessage(e);
		return -1;
	}
}