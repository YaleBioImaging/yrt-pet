/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"
#include "test_utils.hpp"

#include "yrt-pet/datastruct/projection/Sinogram3D.hpp"
#include "yrt-pet/datastruct/scanner/DetRegular.hpp"
#include "yrt-pet/geometry/Constants.hpp"

#include <filesystem>
#include <unordered_set>

namespace
{

// Small scanner sufficient for structural checks.
// numRings=8, detsPerRing=24, maxRingDiff=5, span varies per test.
std::unique_ptr<yrt::Scanner> makeTestScanner(size_t maxRingDiff = 5)
{
	return yrt::util::test::makeFakeScanner(
	    /*axialFOV=*/160.0f, /*crystalSize_z=*/1.0f,
	    /*crystalSize_trans=*/1.0f, /*crystalDepth=*/10.0f,
	    /*scannerRadius=*/200.0f, /*detsPerRing=*/24,
	    /*numRings=*/8, /*numDOI=*/1, maxRingDiff,
	    /*minAngDiff=*/4, /*detsPerBlock=*/4);
}

}  // namespace

namespace yrt
{

TEST_CASE("sinogram3d-params", "[sinogram3d]")
{
	auto scanner = makeTestScanner(/*maxRingDiff=*/5);
	// span=2: segments 0,±1,±2 (|s|·span ≤ maxRingDiff=5 → segSide=3)
	const SinogramParams p =
	    SinogramParams::fromScanner(*scanner, 12, 20, 2, 5);

	SECTION("derived-phiStep")
	{
		REQUIRE(p.phiStep == Approx(PI_FLT / static_cast<float>(p.numPhi)));
	}

	SECTION("derived-rStep")
	{
		const float expectedRStep =
		    2.0f * scanner->scannerRadius / static_cast<float>(p.numR);
		REQUIRE(p.rStep == Approx(expectedRStep));
	}

	SECTION("derived-rMin")
	{
		REQUIRE(p.rMin == Approx(-scanner->scannerRadius));
	}

	SECTION("numSegSide")
	{
		// span=2, maxRingDiff=5 → segments with |s|·span ≤ 5: s=0,1,2
		REQUIRE(p.numSegSide == 3);
	}

	SECTION("numZ-formula")
	{
		// Segment 0: numRings=8 planes
		// Segment ±1: numRings - 1*span = 8-2=6 planes each
		// Segment ±2: numRings - 2*span = 8-4=4 planes each
		const size_t expected = 8 + 2 * 6 + 2 * 4;
		REQUIRE(p.numZ == expected);
	}
}

TEST_CASE("sinogram3d-segment-structure", "[sinogram3d]")
{
	auto scanner = makeTestScanner(/*maxRingDiff=*/5);
	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 12, 20, 2, 5);
	Sinogram3DOwned sino(*scanner, params);

	SECTION("getZFromSegmentPlane-roundtrip")
	{
		for (size_t iZ = 0; iZ < sino.numZ; iZ++)
		{
			const int seg = sino.getSegment(iZ);
			const size_t plane = sino.getPlaneInSegment(iZ);
			const size_t recovered = sino.getZFromSegmentPlane(seg, plane);
			REQUIRE(recovered == iZ);
		}
	}

	SECTION("ring-pair-delta-matches-segment")
	{
		for (size_t iZ = 0; iZ < sino.numZ; iZ++)
		{
			coord_t z1, z2;
			sino.getRingPair(iZ, z1, z2);
			const int seg = sino.getSegment(iZ);
			const int delta = static_cast<int>(z2) - static_cast<int>(z1);
			// For segment s>0: z2-z1 = s·span; s<0: z1-z2 = |s|·span
			const int expected = seg * static_cast<int>(params.span);
			REQUIRE(delta == expected);
		}
	}

	SECTION("ring-pair-in-bounds")
	{
		for (size_t iZ = 0; iZ < sino.numZ; iZ++)
		{
			coord_t z1, z2;
			sino.getRingPair(iZ, z1, z2);
			REQUIRE(z1 < scanner->numRings);
			REQUIRE(z2 < scanner->numRings);
		}
	}
}

TEST_CASE("sinogram3d-binId-roundtrip", "[sinogram3d]")
{
	auto scanner = makeTestScanner(/*maxRingDiff=*/3);
	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 8, 16, 1, 3);
	Sinogram3DOwned sino(*scanner, params);

	// Every flat bin decomposes into (tof, z, phi, r) and back.
	for (bin_t b = 0; b < sino.count(); b++)
	{
		size_t tof, z, phi, r;
		sino.getCoords(b, tof, z, phi, r);
		const bin_t recovered = sino.getBinId(tof, z, phi, r);
		REQUIRE(recovered == b);
	}
}

TEST_CASE("sinogram3d-physical-coords", "[sinogram3d]")
{
	auto scanner = makeTestScanner(/*maxRingDiff=*/5);
	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 12, 20, 2, 5);
	Sinogram3DOwned sino(*scanner, params);
	const float R = scanner->scannerRadius;

	SECTION("phi-range")
	{
		for (size_t iPhi = 0; iPhi < sino.numPhi; iPhi++)
		{
			const float phi = sino.getPhi(iPhi);
			REQUIRE(phi >= 0.0f);
			REQUIRE(phi < PI_FLT);
		}
	}

	SECTION("r-range")
	{
		for (size_t iR = 0; iR < sino.numR; iR++)
		{
			const float r = sino.getR(iR);
			REQUIRE(r > -R);
			REQUIRE(r < R);
		}
	}

	SECTION("phi-step-uniform")
	{
		if (sino.numPhi > 1)
		{
			const float step = sino.getPhi(1) - sino.getPhi(0);
			for (size_t iPhi = 1; iPhi < sino.numPhi; iPhi++)
				REQUIRE(sino.getPhi(iPhi) - sino.getPhi(iPhi - 1) ==
				        Approx(step).epsilon(1e-5f));
		}
	}
}

TEST_CASE("sinogram3d-det-pair-binning", "[sinogram3d]")
{
	auto scanner = makeTestScanner(/*maxRingDiff=*/3);
	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 8, 16, 1, 3);
	Sinogram3DOwned sino(*scanner, params);
	const size_t N = scanner->detsPerRing;

	// Try detector pairs from the same ring (ring diff = 0) that are
	// separated by at least minAngDiff.
	size_t validPairs = 0;
	for (size_t d1 = 0; d1 < N; d1 += 3)
	{
		const size_t d2 = (d1 + N / 2) % N;  // opposite side
		const bin_t b = sino.getBinIdFromDetPair(static_cast<det_id_t>(d1),
		                                         static_cast<det_id_t>(d2));
		if (b == sino.sinoSize)
			continue;

		REQUIRE(b < sino.sinoSize);

		// Swap d1/d2 → same bin (LOR is symmetric)
		const bin_t b2 = sino.getBinIdFromDetPair(static_cast<det_id_t>(d2),
		                                          static_cast<det_id_t>(d1));
		CHECK(b2 == b);

		validPairs++;
	}
	// Sanity: at least some pairs must fall inside the sinogram
	REQUIRE(validPairs > 0);
}

TEST_CASE("sinogram3d-projection-values", "[sinogram3d]")
{
	auto scanner = makeTestScanner(/*maxRingDiff=*/3);
	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 8, 16, 1, 3);
	Sinogram3DOwned sino(*scanner, params);
	sino.allocate();

	SECTION("clear-and-set")
	{
		sino.clearProjections(0.0f);
		for (bin_t b = 0; b < sino.count(); b++)
			REQUIRE(sino.getProjectionValue(b) == Approx(0.0f));

		sino.setProjectionValue(0, 42.0f);
		REQUIRE(sino.getProjectionValue(0) == Approx(42.0f));
		REQUIRE(sino.getProjectionValue(1) == Approx(0.0f));
	}

	SECTION("increment")
	{
		sino.clearProjections(1.0f);
		sino.incrementProjection(0, 3.0f);
		REQUIRE(sino.getProjectionValue(0) == Approx(4.0f));
	}
}

TEST_CASE("sinogram3d-file-io", "[sinogram3d]")
{
	auto scanner = makeTestScanner(/*maxRingDiff=*/3);
	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 8, 16, 1, 3);
	Sinogram3DOwned sinoOut(*scanner, params);
	sinoOut.allocate();
	sinoOut.clearProjections(0.0f);

	// Write a known pattern.
	for (bin_t b = 0; b < sinoOut.count(); b++)
		sinoOut.setProjectionValue(b, static_cast<float>(b % 17));

	const std::string tmpPath = "tmp_test_sinogram3d.sino";
	sinoOut.writeToFile(tmpPath);

	Sinogram3DOwned sinoIn(*scanner, params, tmpPath);

	REQUIRE(sinoIn.count() == sinoOut.count());
	for (bin_t b = 0; b < sinoOut.count(); b++)
		CHECK(sinoIn.getProjectionValue(b) ==
		      Approx(sinoOut.getProjectionValue(b)));

	std::filesystem::remove(tmpPath);
}

TEST_CASE("sinogram3d-bin-iterator", "[sinogram3d]")
{
	auto scanner = makeTestScanner(/*maxRingDiff=*/3);
	const SinogramParams params =
	    SinogramParams::fromScanner(*scanner, 8, 16, 1, 3);
	Sinogram3DOwned sino(*scanner, params);

	constexpr int numSubsets = 4;
	std::unordered_set<bin_t> seen;
	size_t totalIterated = 0;

	for (int sub = 0; sub < numSubsets; sub++)
	{
		auto iter = sino.getBinIter(numSubsets, sub);
		for (bin_t i = 0; i < iter->size(); i++)
		{
			const bin_t b = iter->get(i);
			REQUIRE(b < sino.count());
			REQUIRE(seen.find(b) == seen.end());
			seen.insert(b);
			totalIterated++;
		}
	}

	REQUIRE(totalIterated == sino.count());
}

}  // namespace yrt
