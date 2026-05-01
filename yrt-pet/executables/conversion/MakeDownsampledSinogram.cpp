/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 *
 * MakeDownsampledSinogram
 *
 * Purpose:
 *   Bins list-mode events into per-frame compact 3D "mashed" sinograms with
 *   dimensions [numAxial x numAngles x numRadial], stored as raw float32
 *   binary files. Designed for NMF-based initialisation of the temporal basis
 *   H in list-mode low-rank dynamic PET reconstruction.
 *
 * Sinogram coordinates (derived from cylindrical LOR endpoints):
 *   Given cylindrical coordinates (planePos1, angle1, planePos2, angle2)
 *   where angle1,angle2 in [0, 2pi) and planePos in [-axialFOV/2, axialFOV/2]:
 *
 *   Computed directly from the 3D LOR endpoints (lor.point1, lor.point2):
 *
 *   - View angle  phi  = atan2(P1.y, P1.x) mod pi,  in [0, pi)
 *                        (angle of endpoint 1 in the transaxial plane, folded
 *                        to [0,pi) -- automatically symmetric: phi(P2) = phi(P1)
 *                        since P2 is diametrically opposite P1)
 *   - Radial offset  s   = (P1.x*P2.y - P1.y*P2.x) / |P2xy - P1xy|,  in (-R, R)  SIGNED
 *                        (signed perpendicular distance from scanner center to LOR;
 *                        s=0 for a central LOR; sign flips correctly with DOI;
 *                        no canonical detector ordering required)
 *   - Axial midpoint zMid = (P1.z + P2.z) / 2,  in [-axialFOV/2, axialFOV/2]
 *
 *   Using lor endpoints directly (rather than cylindrical coordinates) handles
 *   DOI correctly: detectors interact at depth inside the crystal, so their 3D
 *   positions are not exactly on the scanner surface.
 *
 * Memory:
 *   numAxial * numAngles * numRadial * 4 bytes per frame.
 *   Example: 20 * 100 * 100 * 4 = 800 KB per frame.
 *   For 6000 frames (10 min at 100 ms): ~4.8 GB total on disk,
 *   but easily reduced by choosing smaller grid sizes (e.g. 10*50*50 = 100KB
 *   per frame -> 600 MB for 6000 frames).
 *
 * Time windowing:
 *   Format-specific options (e.g. NX plugin's --startTime / --stopTime) are
 *   forwarded to openProjectionData() which filters events at load time.
 *   The number of frames is then computed from --time_start, --time_stop and
 *   --time_step (all in milliseconds, relative to the first loaded event),
 *   which is more robust than inferring it from the last timestamp.
 *
 * Usage example (10 min of NX data, 1 s frames):
 *   yrtpet_make_downsampled_sinogram \
 *       --scanner   scanner.json    \
 *       --input     filelist.txt    \
 *       --format    NXLISTMODE      \
 *       --out       sino/frame      \
 *       --time_start  0             \
 *       --time_stop   600000        \
 *       --time_step   1000          \
 *       --num_axial   20            \
 *       --num_angles  100           \
 *       --num_radial  100           \
 *
 * Output:
 *   <out>_frame0000.bin  — raw float32, C-order [numAxial, numAngles, numRadial]
 *   <out>_frame0001.bin  …
 *   <out>_manifest.txt   — one line per frame: "<filename> <startMs> <stopMs>"
 *   <out>_geometry.txt   — grid parameters for Python import
 */

#include "../PluginOptionsHelper.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

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
	while (tmp >= 10) { tmp /= 10; width++; }
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
		const std::string sinoGroup    = "4. Sinogram geometry";

		// ---- Core ----
		registry.registerArgument(
		    "scanner", "Scanner parameters file", true,
		    io::TypeOfArgument::STRING, "", coreGroup, "s");
		registry.registerArgument(
		    "num_threads", "Number of threads to use (-1 = all available)",
		    false, io::TypeOfArgument::INT, -1, coreGroup);

		// ---- Input ----
		registry.registerArgument(
		    "input", "Input list-mode file (or NX file list)", true,
		    io::TypeOfArgument::STRING, "", inputGroup, "i");
		registry.registerArgument(
		    "format",
		    "Input format. Possible values: " +
		        io::possibleFormats(plugin::InputFormatsChoice::ONLYLISTMODES),
		    true, io::TypeOfArgument::STRING, "", inputGroup, "f");

		// ---- Output ----
		registry.registerArgument(
		    "out",
		    "Output filename prefix. Frames written as <out>_frame<NNN>.bin; "
		    "manifest as <out>_manifest.txt; geometry as <out>_geometry.txt.",
		    true, io::TypeOfArgument::STRING, "", outputGroup, "o");

		// ---- Framing ----
		// All in milliseconds, relative to the first loaded event's timestamp.
		// These determine the number of frames independently of the actual
		// last timestamp in the file.
		registry.registerArgument(
		    "time_start",
		    "Start of framing window in ms relative to first loaded event "
		    "(default: 0)",
		    false, io::TypeOfArgument::INT, 0, framingGroup);
		registry.registerArgument(
		    "time_stop",
		    "End of framing window in ms relative to first loaded event. "
		    "Must be set explicitly.",
		    true, io::TypeOfArgument::INT, 0, framingGroup);
		registry.registerArgument(
		    "time_step",
		    "Frame duration in milliseconds (default: 1000 = 1 second)",
		    false, io::TypeOfArgument::INT, 1000, framingGroup);

		// ---- Sinogram geometry ----
		registry.registerArgument(
		    "num_axial",
		    "Number of axial midpoint bins (default: 20)",
		    false, io::TypeOfArgument::INT, 20, sinoGroup);
		registry.registerArgument(
		    "num_angles",
		    "Number of view-angle bins over [0, pi) (default: 100)",
		    false, io::TypeOfArgument::INT, 100, sinoGroup);
		registry.registerArgument(
		    "num_radial",
		    "Number of radial offset bins over [0, R] (default: 100)",
		    false, io::TypeOfArgument::INT, 100, sinoGroup);


		// Forward all plugin-specific options so the time window is applied
		// at load time (e.g. NX --startTime / --stopTime).
		plugin::addOptionsFromPlugins(registry,
		                              plugin::InputFormatsChoice::ONLYLISTMODES);

		// ---- Parse ----
		io::ArgumentReader config{
		    registry,
		    "Bin list-mode data into per-frame compact 3D mashed sinograms "
		    "[numAxial x numAngles x numRadial] for NMF-based initialisation "
		    "of the temporal basis H in list-mode low-rank dynamic PET."};

		if (!config.loadFromCommandLine(argc, argv))
			return 0;
		if (!config.validate())
		{
			std::cerr << "Invalid configuration.\n";
			return -1;
		}

		// ---- Read arguments ----
		const auto scanner_fname  = config.getValue<std::string>("scanner");
		const auto input_fname    = config.getValue<std::string>("input");
		const auto input_format   = config.getValue<std::string>("format");
		const auto out_prefix     = config.getValue<std::string>("out");
		const int  numThreadsArg  = config.getValue<int>("num_threads");
		const int  timeStartArg   = config.getValue<int>("time_start");
		const int  timeStopArg    = config.getValue<int>("time_stop");
		const int  timeStepArg    = config.getValue<int>("time_step");
		const int  numAxialArg    = config.getValue<int>("num_axial");
		const int  numAnglesArg   = config.getValue<int>("num_angles");
		const int  numRadialArg   = config.getValue<int>("num_radial");

		ASSERT_MSG(timeStepArg  > 0, "time_step must be strictly positive");
		ASSERT_MSG(timeStopArg  > timeStartArg,
		           "time_stop must be greater than time_start");
		ASSERT_MSG(numAxialArg  > 0, "num_axial must be strictly positive");
		ASSERT_MSG(numAnglesArg > 0, "num_angles must be strictly positive");
		ASSERT_MSG(numRadialArg > 0, "num_radial must be strictly positive");

		globals::setNumThreads(numThreadsArg);

		const size_t numAxial  = static_cast<size_t>(numAxialArg);
		const size_t numAngles = static_cast<size_t>(numAnglesArg);
		const size_t numRadial = static_cast<size_t>(numRadialArg);
		const size_t sinoSize  = numAxial * numAngles * numRadial;

		// Number of frames derived purely from the requested time window.
		const timestamp_t timeStart = static_cast<timestamp_t>(timeStartArg);
		const timestamp_t timeStop  = static_cast<timestamp_t>(timeStopArg);
		const timestamp_t timeStep  = static_cast<timestamp_t>(timeStepArg);
		const size_t numFrames =
		    static_cast<size_t>((timeStop - timeStart) / timeStep);

		std::cout << "Framing window : [" << timeStartArg << ", "
		          << timeStopArg << ") ms (relative to first loaded event)\n"
		          << "Frame duration : " << timeStepArg << " ms\n"
		          << "Number of frames: " << numFrames << "\n"
		          << "Sinogram shape : [" << numAxial << " axial x "
		          << numAngles << " angles x " << numRadial << " radial]\n"
		          << "Bytes per frame: "
		          << sinoSize * sizeof(float) / 1024 << " KB\n"
		          << "Total on disk  : "
		          << sinoSize * sizeof(float) * numFrames / (1024 * 1024)
		          << " MB\n";

		// ---- Scanner ----
		std::cout << "Initializing scanner..." << std::endl;
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		const float axialFOV = scanner->axialFOV;    // mm
		const float R        = scanner->scannerRadius;  // mm

		// Binning step sizes
		const float axialStep  = axialFOV / static_cast<float>(numAxial);
		const float angleStep  = PI_FLT / static_cast<float>(numAngles);
		const float radialStep = 2.0f * R / static_cast<float>(numRadial);  // s in (-R, R)

		// ---- Load list-mode data ----
		std::cout << "Reading list-mode data..." << std::endl;
		std::unique_ptr<ProjectionData> dataInput = io::openProjectionData(
		    input_fname, input_format, *scanner, config.getAllArguments());

		auto* lm = dynamic_cast<ListMode*>(dataInput.get());
		ASSERT_MSG(lm != nullptr,
		           "The input file does not appear to be list-mode data.");

		const size_t numEvents = lm->count();
		ASSERT_MSG(numEvents > 0,
		           "No events in the loaded data. "
		           "Check the input file and time-window arguments.");

		std::cout << "Events loaded: " << numEvents << std::endl;

		// The framing anchor: all timestamps are expressed relative to the
		// first loaded event, matching the time_start/stop/step convention.
		const timestamp_t t0 = lm->getTimestamp(0);

		// ---- Allocate per-thread sinogram buffers ----
		// Each thread accumulates into its own private buffer with no
		// synchronisation. After the parallel loop a serial reduction sums
		// them. This is the pattern used throughout YRT-PET (cf.
		// convertToHistogram3D / simpleReduceArray in Concurrency.hpp).
		//
		// Cap the thread count to avoid allocating hundreds of large buffers:
		// the bottleneck here is memory bandwidth, not CPU, so 32 threads is
		// plenty and keeps peak extra memory at 32 * totalSize * 4 bytes.
		const int numThreadsRequested = globals::getNumThreads();
		const int numThreads = std::min(numThreadsRequested, 32);
		const size_t totalSize = numFrames * sinoSize;

		std::cout << "Allocating " << numThreads << " per-thread buffers ("
		          << static_cast<double>(totalSize * sizeof(float) * numThreads) /
		                 (1024.0 * 1024.0)
		          << " MB total)..." << std::endl;

		std::vector<std::vector<float>> threadData(
		    numThreads, std::vector<float>(totalSize, 0.0f));

		// ---- Bin events ----
		std::cout << "Binning events..." << std::endl;

		util::ProgressDisplayMultiThread progressBar(numThreads, numEvents, 5);

		util::parallelForChunked(
		    numEvents, numThreads,
		    [&](size_t evId, size_t threadId)
		    {
			    progressBar.incrementProgress(threadId);

			    // t0 is the timestamp of the first loaded event, which is
			    // already at time_start (the NX plugin seeked to it before
			    // loading). So relTs is in [0, time_stop - time_start) and
			    // needs no range check beyond the safety guard below.
			    const timestamp_t relTs = lm->getTimestamp(evId) - t0;
			    const size_t frameIdx = static_cast<size_t>(relTs / timeStep);
			    if (frameIdx >= numFrames)
				    return;

			    // ---- Mashed sinogram coordinates from LOR endpoints ----
			    const Line3D lor = lm->getLOR(evId);

			    // View angle phi in [0, pi):
			    // atan2(P1.y, P1.x) gives the angle of endpoint 1 in [-pi, pi].
			    // Adding TWO_PI_FLT then fmod-ing by PI_FLT folds it to [0, pi)
			    // in one step, and is automatically symmetric: the opposing
			    // endpoint P2 is at angle a1 +/- pi, so (a2 + 2pi) mod pi = phi.
			    const float phi = std::fmod(
			        std::atan2(lor.point1.y, lor.point1.x) + TWO_PI_FLT, PI_FLT);

			    // Axial midpoint: zMid in [-axialFOV/2, axialFOV/2]
			    const float zMid = 0.5f * (lor.point1.z + lor.point2.z);

			    // Signed radial offset s = (P1 x P2) / |P2 - P1| (2D cross product).
			    // This is the perpendicular signed distance from the scanner center
			    // to the LOR. Using the actual 3D coordinates of the LOR endpoints
			    // (which already account for DOI depth) is more accurate than the
			    // trigonometric approximation R*cos(halfDelta) which assumed detectors
			    // sit exactly on the scanner surface.
			    // Swapping P1<->P2 flips the sign of the cross product, but phi is
			    // unchanged (both endpoints give the same phi mod pi), so (phi, s)
			    // is the same bin regardless of which detector fired first.
			    const float dx = lor.point2.x - lor.point1.x;
			    const float dy = lor.point2.y - lor.point1.y;
			    const float s  = (lor.point1.x * lor.point2.y -
			                      lor.point1.y * lor.point2.x)
			                     / std::sqrt(dx * dx + dy * dy);

			    // ---- Bin indices ----
			    // zMid: shift from [-axialFOV/2, axialFOV/2] to [0, axialFOV]
			    int zBin   = static_cast<int>((zMid + 0.5f * axialFOV) / axialStep);
			    int phiBin = static_cast<int>(phi / angleStep);
			    // s in (-R,R): shift by R to get [0,2R), then bin
			    int sBin   = static_cast<int>((s + R) / radialStep);

			    // Clamp to valid range
			    zBin   = std::clamp(zBin,   0, static_cast<int>(numAxial)  - 1);
			    phiBin = std::clamp(phiBin, 0, static_cast<int>(numAngles) - 1);
			    sBin   = std::clamp(sBin,   0, static_cast<int>(numRadial) - 1);

			    const size_t flatIdx =
			        frameIdx * sinoSize +
			        static_cast<size_t>(zBin)   * (numAngles * numRadial) +
			        static_cast<size_t>(phiBin)  * numRadial +
			        static_cast<size_t>(sBin);

			    // No synchronisation needed: each thread writes only to its own buffer
			    threadData[threadId][flatIdx] += 1.0f;
		    });

		// ---- Reduce per-thread buffers into a single array ----
		std::cout << "\nReducing per-thread buffers..." << std::endl;
		std::vector<float> data(totalSize, 0.0f);
		for (int tid = 0; tid < numThreads; tid++)
		{
			for (size_t i = 0; i < totalSize; i++)
				data[i] += threadData[tid][i];
			// Free as we go to avoid doubling peak memory
			threadData[tid].clear();
			threadData[tid].shrink_to_fit();
		}

		// ---- Write output ----
		std::cout << "Writing frames..." << std::endl;

		const std::string manifestPath = out_prefix + "_manifest.txt";
		const std::string geometryPath = out_prefix + "_geometry.txt";

		std::ofstream manifest(manifestPath);
		ASSERT_MSG(manifest.is_open(),
		           ("Cannot open manifest: " + manifestPath).c_str());

		// Write geometry file for easy Python import
		std::ofstream geom(geometryPath);
		ASSERT_MSG(geom.is_open(),
		           ("Cannot open geometry file: " + geometryPath).c_str());
		geom << "num_frames  " << numFrames    << "\n"
		     << "num_axial   " << numAxial     << "\n"
		     << "num_angles  " << numAngles    << "\n"
		     << "num_radial  " << numRadial    << "\n"
		     << "time_start  " << timeStartArg << "\n"
		     << "time_stop   " << timeStopArg  << "\n"
		     << "time_step   " << timeStepArg  << "\n"
		     << "axial_fov   " << axialFOV     << "\n"
		     << "radius      " << R            << "\n"
		     << "radial_min  " << -R           << "\n"
		     << "radial_max  " << R            << "\n"
		     << "dtype       float32\n"
		     << "order       C\n"
		     << "# shape per frame: [num_axial, num_angles, num_radial]\n";
		geom.close();

		for (size_t f = 0; f < numFrames; f++)
		{
			const std::string fname =
			    out_prefix + "_frame" + frameTag(f, numFrames) + ".bin";

			std::ofstream ofs(fname, std::ios::binary);
			ASSERT_MSG(ofs.is_open(),
			           ("Cannot open output file: " + fname).c_str());
			ofs.write(reinterpret_cast<const char*>(data.data() + f * sinoSize),
			          static_cast<std::streamsize>(sinoSize * sizeof(float)));

			const timestamp_t frameStart =
			    timeStart + static_cast<timestamp_t>(f) * timeStep;
			const timestamp_t frameStop = frameStart + timeStep;
			manifest << fname << " " << frameStart << " " << frameStop << "\n";

			std::cout << "  Frame " << f + 1 << "/" << numFrames
			          << " -> " << fname << "\n";
		}
		manifest.close();

		std::cout << "Manifest : " << manifestPath << "\n"
		          << "Geometry : " << geometryPath << "\n"
		          << "Done." << std::endl;
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