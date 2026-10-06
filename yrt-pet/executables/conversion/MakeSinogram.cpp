/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 *
 * yrtpet_make_sinogram
 *
 * Bins list-mode events into per-frame Sinogram3D files in the
 * YRT-PET SINO format.  Replaces the legacy MakeDownsampledSinogram
 * (raw float32 + geometry text) with proper YRT data structures.
 *
 * Sinogram geometry is controlled by --num_phi, --num_r, --span, and
 * --max_ring_diff; all other parameters (scanner radius, axial FOV,
 * ring count) come from the scanner JSON.
 *
 * Framing - two mutually exclusive modes:
 *
 *   1. Uniform (--time_start / --time_stop / --time_step, ms):
 *      Equal-duration frames relative to the first loaded event.
 *      Format-specific options (e.g. the NX plugin's --startTime /
 *      --stopTime) are forwarded to openProjectionData(), which
 *      filters events at load time so the first event is already at
 *      time_start.
 *
 *   2. Explicit (--dyn <file.dyn>):
 *      Frames from a YRT-PET dynamic framing file.  Timestamps are
 *      absolute (same reference as event timestamps).
 *
 * Output:
 *   <out>_frame<NNN>.sino  - one Sinogram3DOwned per frame (SINO format)
 *   <out>_manifest.txt     - one line per frame: "<filename> <startMs>
 *                            <stopMs>"
 */

#include "../PluginOptionsHelper.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/DynamicFraming.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/projection/Sinogram3D.hpp"
#include "yrt-pet/datastruct/projection/SinogramConverter.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"

#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace yrt;

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

int main(int argc, char** argv)
{
	try
	{
		io::ArgumentRegistry registry{};

		const std::string coreGroup = '0' + std::string(". Core");
		const std::string inputGroup = '1' + std::string(". Input");
		const std::string outputGroup = '2' + std::string(". Output");
		const std::string framingGroup = '3' + std::string(". Framing");
		const std::string sinoGroup = '4' + std::string(". Sinogram geometry");

		// ---- Core ----
		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument(
		    "num_threads",
		    "Number of threads to use (-1 = all available, default: -1)", false,
		    io::TypeOfArgument::INT, -1, coreGroup);

		// ---- Input ----
		registry.registerArgument(
		    "input", "Input list-mode file (or file list)", true,
		    io::TypeOfArgument::STRING, "", inputGroup, "i");
		registry.registerArgument(
		    "format",
		    "Input format.  Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYLISTMODES),
		    true, io::TypeOfArgument::STRING, "", inputGroup, "f");

		// ---- Output ----
		registry.registerArgument(
		    "out",
		    "Output prefix.  Frames written as <out>_frame<NNN>.sino; "
		    "manifest as <out>_manifest.txt.",
		    true, io::TypeOfArgument::STRING, "", outputGroup, "o");

		// ---- Framing ----
		registry.registerArgument(
		    "dyn",
		    "Dynamic framing file (.dyn).  When given, frames follow "
		    "that file and time_start / time_stop / time_step are ignored.",
		    false, io::TypeOfArgument::STRING, "", framingGroup);
		registry.registerArgument(
		    "time_start",
		    "Start of framing window in ms relative to first loaded event "
		    "(default: 0).  Uniform framing only.",
		    false, io::TypeOfArgument::INT, 0, framingGroup);
		registry.registerArgument(
		    "time_stop",
		    "End of framing window in ms relative to first loaded event.  "
		    "Required unless --dyn is given.",
		    false, io::TypeOfArgument::INT, 0, framingGroup);
		registry.registerArgument(
		    "time_step", "Frame duration in milliseconds (default: 1000).",
		    false, io::TypeOfArgument::INT, 1000, framingGroup);

		// ---- Sinogram geometry ----
		registry.registerArgument(
		    "num_phi", "Number of view-angle bins over [0, pi) (default: 200)",
		    false, io::TypeOfArgument::INT, 200, sinoGroup);
		registry.registerArgument(
		    "num_r", "Number of radial-offset bins over (-R, R) (default: 200)",
		    false, io::TypeOfArgument::INT, 200, sinoGroup);
		registry.registerArgument(
		    "span",
		    "Axial compression span (default: 11).  span=1 = no compression.",
		    false, io::TypeOfArgument::INT, 11, sinoGroup);
		registry.registerArgument(
		    "max_ring_diff",
		    "Maximum ring difference included (default: 0 = scanner max).",
		    false, io::TypeOfArgument::INT, 0, sinoGroup);

		plugin::addOptionsFromPlugins(registry,
		                              plugin::InputFormatsChoice::ALL);

		io::ArgumentReader config{
		    registry,
		    "Bin list-mode data into per-frame Sinogram3D files (SINO format) "
		    "compatible with the YRT-PET projector and corrector pipeline."};

		if (!config.loadFromCommandLine(argc, argv))
			return 0;
		if (!config.validate())
		{
			std::cerr << "Invalid configuration.\n";
			return -1;
		}

		// ---- Read arguments ----
		const auto scanner_fname = config.getValue<std::string>("scanner");
		const auto input_fname = config.getValue<std::string>("input");
		const auto input_format = config.getValue<std::string>("format");
		const auto out_prefix = config.getValue<std::string>("out");
		const auto dyn_fname = config.getValue<std::string>("dyn");
		const int numThreadsArg = config.getValue<int>("num_threads");
		const int timeStartArg = config.getValue<int>("time_start");
		const int timeStopArg = config.getValue<int>("time_stop");
		const int timeStepArg = config.getValue<int>("time_step");
		const int numPhiArg = config.getValue<int>("num_phi");
		const int numRArg = config.getValue<int>("num_r");
		const int spanArg = config.getValue<int>("span");
		const int maxRingDiffArg = config.getValue<int>("max_ring_diff");

		ASSERT_MSG(numPhiArg > 0, "num_phi must be strictly positive");
		ASSERT_MSG(numRArg > 0, "num_r must be strictly positive");
		ASSERT_MSG(spanArg > 0, "span must be strictly positive");
		ASSERT_MSG(maxRingDiffArg >= 0,
		           "max_ring_diff must be non-negative (0 = scanner max)");

		globals::setNumThreads(numThreadsArg);

		// ---- Scanner ----
		std::cout << "Initializing scanner..." << std::endl;
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		// ---- Sinogram parameters ----
		const SinogramParams params = SinogramParams::fromScanner(
		    *scanner, static_cast<size_t>(numPhiArg),
		    static_cast<size_t>(numRArg), static_cast<size_t>(spanArg),
		    static_cast<size_t>(maxRingDiffArg));

		const size_t sinoSize =
		    params.numTOF * params.numZ * params.numPhi * params.numR;

		std::cout << "Sinogram geometry:\n"
		          << "  numPhi=" << params.numPhi << "  numR=" << params.numR
		          << "  numZ=" << params.numZ << "  span=" << params.span
		          << "  maxRingDiff=" << params.maxRingDiff << "\n"
		          << "  Bytes per frame: " << sinoSize * sizeof(float) / 1024
		          << " KB\n";

		// ---- Framing ----
		const bool useDynFraming = !dyn_fname.empty();
		std::shared_ptr<DynamicFraming> framing = nullptr;
		const timestamp_t timeStep = static_cast<timestamp_t>(timeStepArg);
		timestamp_t timeStart = static_cast<timestamp_t>(timeStartArg);
		timestamp_t timeStop = static_cast<timestamp_t>(timeStopArg);
		size_t numFrames;

		if (useDynFraming)
		{
			framing = std::make_shared<DynamicFraming>(dyn_fname);
			ASSERT_MSG(framing->isValid(),
			           "Dynamic framing timestamps are not in chronological "
			           "order.");
			numFrames = framing->getNumFrames();
			ASSERT_MSG(numFrames > 0,
			           "Dynamic framing file defines no frames.");
			timeStart = framing->getStartingTimestamp(0);
			timeStop = framing->getLastTimestamp();
			std::cout << "Framing       : " << dyn_fname << "\n"
			          << "Framing window: [" << timeStart << ", " << timeStop
			          << ") ms (absolute)\n";
		}
		else
		{
			ASSERT_MSG(timeStepArg > 0, "time_step must be strictly positive");
			ASSERT_MSG(timeStopArg > timeStartArg,
			           "time_stop must be greater than time_start. "
			           "Set --time_stop, or use --dyn.");
			numFrames = static_cast<size_t>((timeStop - timeStart) / timeStep);
			std::cout << "Framing       : uniform, " << timeStepArg
			          << " ms per frame\n"
			          << "Framing window: [" << timeStartArg << ", "
			          << timeStopArg
			          << ") ms (relative to first loaded event)\n";
		}

		std::cout << "Number of frames: " << numFrames << "\n"
		          << "Total on disk   : "
		          << sinoSize * sizeof(float) * numFrames / (1024 * 1024)
		          << " MB\n";

		// ---- Load list-mode data ----
		std::cout << "Reading list-mode data..." << std::endl;
		std::unique_ptr<ProjectionData> dataInput = io::openProjectionData(
		    input_fname, input_format, *scanner, config.getAllArguments());

		auto* lm = dynamic_cast<ListMode*>(dataInput.get());
		ASSERT_MSG(lm != nullptr,
		           "Input does not appear to be list-mode data.");

		const size_t numEvents = lm->count();
		ASSERT_MSG(numEvents > 0, "No events loaded. "
		                          "Check the input file and time-window "
		                          "arguments.");
		std::cout << "Events loaded   : " << numEvents << std::endl;

		// Assign frame indices.  For uniform framing we compute them
		// inline; for explicit framing addDynamicFraming precomputes
		// them (returns -1 for events outside the framing).
		const timestamp_t t0 = lm->getTimestamp(0);
		if (useDynFraming)
		{
			std::cout << "Assigning frames from the dynamic framing..."
			          << std::endl;
			lm->addDynamicFraming(framing);
		}

		// ---- Allocate flat multi-frame buffer ----
		// Layout: [frame][tof][z][phi][r] - with numTOF=1 this is simply
		// [frame][z][phi][r].  Bin index within frame f is
		// getBinIdFromDetPair(d1, d2), which already lives in [0, sinoSize).
		std::cout << "Allocating output buffer ("
		          << numFrames * sinoSize * sizeof(float) / (1024 * 1024)
		          << " MB)..." << std::endl;
		std::vector<float> data(numFrames * sinoSize, 0.0f);

		// Reference sinogram: geometry only, no data allocation needed.
		// getBinIdFromDetPair() uses only params and scanner LUT.
		Sinogram3DOwned refSino(*scanner, params);
		const bin_t sentinel = refSino.sinoSize;
		const Sinogram3DOwned* refSinoPtr = &refSino;

		const size_t nThreads = static_cast<size_t>(globals::getNumThreads());
		float* dataPtr = data.data();

		// ---- Bin events ----
		std::cout << "Binning " << numEvents << " events into " << numFrames
		          << " frame(s)..." << std::endl;

		util::parallelForChunked(
		    numEvents, nThreads,
		    [refSinoPtr, lm, dataPtr, sentinel, sinoSize, numFrames,
		     useDynFraming, t0, timeStep, timeStart](bin_t ev, size_t /*tid*/)
		    {
			    size_t frameIdx;
			    if (useDynFraming)
			    {
				    const frame_t fr = lm->getDynamicFrame(ev);
				    if (fr < 0)
					    return;
				    frameIdx = static_cast<size_t>(fr);
			    }
			    else
			    {
				    const timestamp_t relTs = lm->getTimestamp(ev) - t0;
				    frameIdx = static_cast<size_t>(relTs / timeStep);
			    }
			    if (frameIdx >= numFrames)
				    return;

			    const det_id_t d1 = lm->getDetector1(ev);
			    const det_id_t d2 = lm->getDetector2(ev);
			    const bin_t b = refSinoPtr->getBinIdFromDetPair(d1, d2);
			    if (b == sentinel)
				    return;

			    std::atomic_ref<float> slot(dataPtr[frameIdx * sinoSize + b]);
			    slot += 1.0f;
		    });

		// ---- Write frames ----
		std::cout << "Writing frames..." << std::endl;
		const std::string manifestPath = out_prefix + "_manifest.txt";
		std::ofstream manifest(manifestPath);
		ASSERT_MSG(manifest.is_open(),
		           ("Cannot open manifest: " + manifestPath).c_str());

		for (size_t f = 0; f < numFrames; f++)
		{
			const std::string fname =
			    out_prefix + "_frame" + frameTag(f, numFrames) + ".sino";

			// Build an alias sinogram pointing to this frame's slice.
			Array4DAlias<float> frameArray;
			frameArray.bind(dataPtr + f * sinoSize, params.numTOF, params.numZ,
			                params.numPhi, params.numR);
			Sinogram3DAlias frameSino(*scanner, params);
			frameSino.bind(frameArray);
			frameSino.writeToFile(fname);

			timestamp_t frameStart;
			timestamp_t frameStop;
			if (useDynFraming)
			{
				frameStart =
				    framing->getStartingTimestamp(static_cast<frame_t>(f));
				frameStop =
				    framing->getStoppingTimestamp(static_cast<frame_t>(f));
			}
			else
			{
				frameStart = timeStart + static_cast<timestamp_t>(f) * timeStep;
				frameStop = frameStart + timeStep;
			}
			manifest << fname << " " << frameStart << " " << frameStop << "\n";

			std::cout << "  Frame " << f + 1 << "/" << numFrames << " -> "
			          << fname << "\n";
		}
		manifest.close();

		std::cout << "Manifest: " << manifestPath << "\nDone.\n";
		return 0;
	}
	catch (const std::exception& e)
	{
		util::printExceptionMessage(e);
		return -1;
	}
}
