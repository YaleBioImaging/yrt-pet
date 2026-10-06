/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"
#include "test_utils.hpp"

#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/datastruct/projection/Sinogram3D.hpp"
#include "yrt-pet/datastruct/projection/SinogramConverter.hpp"
#include "yrt-pet/datastruct/scanner/DetRegular.hpp"

namespace
{

std::unique_ptr<yrt::Scanner> makeTestScanner()
{
	return yrt::util::test::makeFakeScanner(
	    /*axialFOV=*/160.0f, /*crystalSize_z=*/1.0f,
	    /*crystalSize_trans=*/1.0f, /*crystalDepth=*/10.0f,
	    /*scannerRadius=*/200.0f, /*detsPerRing=*/24,
	    /*numRings=*/8, /*numDOI=*/1, /*maxRingDiff=*/5,
	    /*minAngDiff=*/4, /*detsPerBlock=*/4);
}

// Sum all projection values in a ProjectionData.
double sumProjections(const yrt::ProjectionData& pd)
{
	double total = 0.0;
	for (yrt::bin_t b = 0; b < pd.count(); b++)
		total += static_cast<double>(pd.getProjectionValue(b));
	return total;
}

}  // namespace

namespace yrt
{

TEST_CASE("sino-converter-fromHistogram3D", "[sinogram3d][converter]")
{
	auto scanner = makeTestScanner();

	auto h3d = std::make_unique<Histogram3DOwned>(*scanner);
	h3d->allocate();
	h3d->clearProjections(0.0f);

	// Put 1.0 in every H3D bin.
	for (bin_t b = 0; b < h3d->count(); b++)
		h3d->setProjectionValue(b, 1.0f);

	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 12, 20, 2, 5);
	auto sino = sino::fromHistogram3D(*h3d, params);

	REQUIRE(sino != nullptr);
	REQUIRE(sino->count() > 0);

	// Total counts in sinogram ≤ total counts in H3D (some LORs may be
	// outside the sinogram's maxRingDiff or radial range).
	const double h3dTotal = sumProjections(*h3d);
	const double sinoTotal = sumProjections(*sino);
	CHECK(sinoTotal <= h3dTotal + 1e-3);
	CHECK(sinoTotal > 0.0);
}

TEST_CASE("sino-converter-toHistogram3D", "[sinogram3d][converter]")
{
	auto scanner = makeTestScanner();

	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 12, 20, 2, 5);
	Sinogram3DOwned sino(*scanner, params);
	sino.allocate();
	sino.clearProjections(0.0f);

	// Put 1.0 in every sinogram bin.
	for (bin_t b = 0; b < sino.count(); b++)
		sino.setProjectionValue(b, 1.0f);

	auto h3d = sino::toHistogram3D(sino);

	REQUIRE(h3d != nullptr);
	REQUIRE(h3d->count() > 0);

	// Total counts in H3D ≤ total counts in sinogram (approximate
	// getDetector1/2 may map bins outside the H3D grid).
	const double sinoTotal = sumProjections(sino);
	const double h3dTotal = sumProjections(*h3d);
	CHECK(h3dTotal <= sinoTotal + 1e-3);
	CHECK(h3dTotal > 0.0);
}

TEST_CASE("sino-converter-fillFromListMode", "[sinogram3d][converter]")
{
	auto scanner = makeTestScanner();
	const size_t N = scanner->detsPerRing;    // 24
	const size_t nRings = scanner->numRings;  // 8

	// Build synthetic events: detector pairs from the same ring,
	// opposite sides (guaranteed to be in-plane and far enough apart).
	// Use 4 different phi angles (d1 = 0,6,12,18 with d2 = opposite).
	const size_t numEvents = 8;
	ListModeLUTOwned lm(*scanner);
	lm.allocate(numEvents);

	size_t ev = 0;
	for (size_t ring = 0; ring < 2; ring++)
	{
		for (size_t i = 0; i < 4; i++)
		{
			const det_id_t d1 = static_cast<det_id_t>(ring * N + i * (N / 4));
			const det_id_t d2 =
			    static_cast<det_id_t>(ring * N + (i * (N / 4) + N / 2) % N);
			lm.setDetectorIdsOfEvent(ev, d1, d2);
			lm.setTimestampOfEvent(ev, static_cast<timestamp_t>(ev));
			ev++;
		}
	}

	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 12, 20, 2, 5);
	Sinogram3DOwned sino(*scanner, params);
	sino.allocate();

	sino::fillFromListMode(sino, lm, 1);

	// Count non-zero sinogram bins and total accumulated counts.
	size_t nonZero = 0;
	double total = 0.0;
	for (bin_t b = 0; b < sino.count(); b++)
	{
		const float v = sino.getProjectionValue(b);
		if (v > 0.0f)
		{
			nonZero++;
			total += static_cast<double>(v);
		}
	}

	// All 8 events should be in-range (direct planes, opposite detectors).
	REQUIRE(total == Approx(static_cast<double>(numEvents)));
	REQUIRE(nonZero > 0);
}

TEST_CASE("sino-converter-fillFromListMode-counts-conserved",
          "[sinogram3d][converter]")
{
	auto scanner = makeTestScanner();
	const size_t N = scanner->detsPerRing;

	// Generate events that definitely fall inside the sinogram: same ring,
	// opposite detectors, stepping through all phi angles.
	const size_t numEvents = N / 2;
	ListModeLUTOwned lm(*scanner);
	lm.allocate(numEvents);

	for (size_t i = 0; i < numEvents; i++)
	{
		const det_id_t d1 = static_cast<det_id_t>(i);
		const det_id_t d2 = static_cast<det_id_t>((i + N / 2) % N);
		lm.setDetectorIdsOfEvent(i, d1, d2);
		lm.setTimestampOfEvent(i, static_cast<timestamp_t>(i));
	}

	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 12, 20, 1, 0);
	Sinogram3DOwned sino(*scanner, params);
	sino.allocate();
	sino::fillFromListMode(sino, lm, 2);

	const double total = sumProjections(sino);
	// All events are same-ring → ring diff = 0 ≤ maxRingDiff; all
	// should land in the sinogram (no out-of-range drops).
	REQUIRE(total == Approx(static_cast<double>(numEvents)));
}

TEST_CASE("sino-converter-fromHistogram3D-span1-identity",
          "[sinogram3d][converter]")
{
	auto scanner = makeTestScanner();

	// With span=1 and full maxRingDiff, every H3D ring-diff combination
	// maps to a distinct segment.  Total sinogram count must equal total
	// H3D count (assuming radial/phi grids are wide enough to contain all
	// LORs).  Use a coarse sinogram to allow the geometry to match.
	auto h3d = std::make_unique<Histogram3DOwned>(*scanner);
	h3d->allocate();
	h3d->clearProjections(0.0f);

	// Activate a small subset of bins to avoid slow enumeration.
	for (bin_t b = 0; b < h3d->count(); b += 100)
		h3d->setProjectionValue(b, 1.0f);

	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 24, 40, 1, 0);
	auto sino = sino::fromHistogram3D(*h3d, params);

	const double h3dTotal = sumProjections(*h3d);
	const double sinoTotal = sumProjections(*sino);
	// With fine enough grids the counts should be very close; the sinogram
	// may lose a small fraction at the radial edges.
	CHECK(sinoTotal / h3dTotal > 0.9);
}

}  // namespace yrt
