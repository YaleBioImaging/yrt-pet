/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/datastruct/scanner/DetRegular.hpp"
#include "test_utils.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

namespace yrt
{

bool check_coords(std::array<coord_t, 3> c1, std::array<coord_t, 3> c2)
{
	return (c1[0] == c2[0] && c1[1] == c2[1] && c1[2] == c2[2]);
}

bool check_det_pairs(det_pair_t p1, det_pair_t p2)
{
	return (p1.d1 == p2.d1 && p1.d2 == p2.d2) ||
	       (p1.d1 == p2.d2 && p1.d2 == p2.d1);
}

bool compare_det_pairs(det_pair_t detPair1, det_pair_t detPair2)
{
	det_pair_t p1 = detPair1;
	if (p1.d1 > p1.d2)
	{
		p1 = {p1.d2, p1.d1};
	}
	det_pair_t p2 = detPair2;
	if (p2.d1 > p2.d2)
	{
		p2 = {p2.d2, p2.d1};
	}

	if (p1.d1 < p2.d1)
	{
		return true;
	}
	else if (p1.d1 > p2.d1)
	{
		return false;
	}
	else
	{
		if (p1.d2 < p2.d2)
		{
			return true;
		}
		else if (p1.d2 > p2.d2)
		{
			return false;
		}
		else
		{
			return false;
		}
	}
}

bool compare_coords(std::array<coord_t, 3> c1, std::array<coord_t, 3> c2)
{
	if (c1[2] < c2[2])
	{
		return true;
	}
	else if (c1[2] > c2[2])
	{
		return false;
	}
	else
	{
		if (c1[1] < c2[1])
		{
			return true;
		}
		else if (c1[1] > c2[1])
		{
			return false;
		}
		else
		{
			if (c1[0] < c2[0])
			{
				return true;
			}
			else if (c1[0] > c2[0])
			{
				return false;
			}
			else
			{
				// Fully equal
				return false;
			}
		}
	}
}

auto getDetectorPositionInScanner(const Scanner& scanner, det_id_t d)
{
	int64_t doi = d / (scanner.numRings * scanner.detsPerRing);
	int64_t ring = (d - doi * scanner.detsPerRing * scanner.numRings) /
	               scanner.detsPerRing;
	int64_t posInRing =
	    (d - (ring + doi * scanner.numRings) * scanner.detsPerRing) %
	    scanner.detsPerRing;
	return std::make_tuple(doi, ring, posInRing);
}

}  // namespace yrt

TEST_CASE("histo3d", "[histo]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<size_t> numRingsDistribution(1, 15);
	std::uniform_int_distribution<size_t> numDOIDistribution(1, 4);
	// detsPerRing is forced to be an even number here (from 4 to 100 crystals)
	std::uniform_int_distribution<size_t> halfDetsPerRingDistribution(2, 50);

	constexpr int NumTrials = 25;

	for (int trial = 0; trial < NumTrials; trial++)
	{
		size_t numDOI = numDOIDistribution(engine);
		size_t numRings = numRingsDistribution(engine);
		size_t detsPerRing = halfDetsPerRingDistribution(engine) * 2ull;

		std::uniform_int_distribution<size_t> minAngDiffDistribution(
		    1, detsPerRing / 2);
		size_t minAngDiff = minAngDiffDistribution(engine);
		size_t maxRingDiff;
		if (numRings > 1)
		{
			std::uniform_int_distribution<size_t> maxRingDiffDistribution(
			    0, numRings - 1);
			maxRingDiff = maxRingDiffDistribution(engine);
		}
		else
		{
			maxRingDiff = 0;
		}

		// Create scanner
		auto scanner = std::make_unique<yrt::Scanner>(
		    "FakeScanner", 200, 1, 1, 10, 200, detsPerRing, numRings, numDOI,
		    maxRingDiff, minAngDiff, 2);
		const auto detRegular =
		    std::make_shared<yrt::DetRegular>(scanner.get());
		detRegular->generateLUT();
		scanner->setDetectorSetup(detRegular);

		size_t numDets =
		    scanner->numDOI * scanner->numRings * scanner->detsPerRing;
		auto histo3d = std::make_unique<yrt::Histogram3DOwned>(*scanner);

		SECTION("histo3d-binIds-trial-" + std::to_string(trial + 1))
		{
			for (yrt::bin_t binId = 0; binId < histo3d->count(); binId++)
			{
				yrt::coord_t r, phi, z_bin;
				histo3d->getCoordsFromBinId(binId, r, phi, z_bin);
				yrt::bin_t supposedBin =
				    histo3d->getBinIdFromCoords(r, phi, z_bin);
				REQUIRE(supposedBin == binId);
			}
		}

		SECTION("histo3d-coords-binning-trial-" + std::to_string(trial + 1))
		{
			yrt::coord_t r, phi, z_bin;
			for (r = 0; r < histo3d->numR; r++)
			{
				for (phi = 0; phi < histo3d->numPhi; phi++)
				{
					for (z_bin = 0; z_bin < histo3d->numZBin; z_bin++)
					{
						yrt::det_id_t d1, d2;
						yrt::coord_t r_supp, phi_supp, z_bin_supp;
						histo3d->getDetPairFromCoords(r, phi, z_bin, d1, d2);
						if (d1 == 0 && d2 == 0)
						{
							// Disabled bin
							continue;
						}
						histo3d->getCoordsFromDetPair(d1, d2, r_supp, phi_supp,
						                              z_bin_supp);
						yrt::det_id_t d1_supp, d2_supp;
						histo3d->getDetPairFromCoords(
						    r_supp, phi_supp, z_bin_supp, d1_supp, d2_supp);

						bool check = (d1_supp == d1 && d2_supp == d2) ||
						             (d1_supp == d2 && d2_supp == d1);
						REQUIRE(check);
						REQUIRE(d1 < numDets);
						REQUIRE(d2 < numDets);
					}
				}
			}
		}

		SECTION("histo3d-detector-swap-trial-" + std::to_string(trial + 1))
		{
			yrt::coord_t r, phi, z_bin;
			for (r = 0; r < histo3d->numR; r++)
			{
				for (phi = 0; phi < histo3d->numPhi; phi++)
				{
					for (z_bin = 0; z_bin < histo3d->numZBin; z_bin++)
					{
						yrt::det_id_t d1, d2;
						yrt::coord_t r_supp, phi_supp, z_bin_supp;
						histo3d->getDetPairFromCoords(r, phi, z_bin, d1, d2);
						if (d1 == 0 && d2 == 0)
						{
							// Disabled bin
							continue;
						}
						histo3d->getCoordsFromDetPair(d2, d1, r_supp, phi_supp,
						                              z_bin_supp);
						CHECK(r_supp == r);
						CHECK(phi_supp == phi);
						CHECK(z_bin_supp == z_bin);
					}
				}
			}
		}

		SECTION("histo3d-coords-uniqueness-trial-" + std::to_string(trial + 1))
		{
			std::vector<std::array<yrt::coord_t, 3>> all_coords;

			for (yrt::det_id_t d1 = 0; d1 < numDets; d1++)
			{
				for (yrt::det_id_t d2 = d1 + 1; d2 < numDets; d2++)
				{
					int d1_ring = d1 % (scanner->detsPerRing);
					int d2_ring = d2 % (scanner->detsPerRing);
					int diff = std::abs(d1_ring - d2_ring);
					diff = (diff < static_cast<int>(scanner->detsPerRing / 2)) ?
					           diff :
					           scanner->detsPerRing - diff;
					if (diff < static_cast<int>(scanner->minAngDiff))
					{
						continue;
					}
					int z1 =
					    (d1 / (scanner->detsPerRing)) % (scanner->numRings);
					int z2 =
					    (d2 / (scanner->detsPerRing)) % (scanner->numRings);
					if (std::abs(z1 - z2) > scanner->maxRingDiff)
					{
						continue;
					}

					yrt::coord_t r, phi, z_bin;
					histo3d->getCoordsFromDetPair(d1, d2, r, phi, z_bin);
					all_coords.push_back({r, phi, z_bin});
				}
			}
			std::sort(std::begin(all_coords), std::end(all_coords),
			          yrt::compare_coords);
			auto u = std::unique(std::begin(all_coords), std::end(all_coords),
			                     yrt::check_coords);
			bool containsDuplicate = u != std::end(all_coords);
			REQUIRE(!containsDuplicate);
		}

		SECTION("histo3d-detector-pairs-uniqueness-trial-" +
		        std::to_string(trial + 1))
		{
			std::vector<yrt::det_pair_t> allDetPairs;

			const size_t numBins = histo3d->count();
			allDetPairs.reserve(numBins);

			for (yrt::bin_t binId = 0; binId < numBins; binId++)
			{
				yrt::det_pair_t currPair = histo3d->getDetectorPair(binId);

				if (currPair.d1 == 0 && currPair.d2 == 0)
				{
					// Disabled bin;
					continue;
				}
				allDetPairs.emplace_back(currPair);
			}

			std::sort(std::begin(allDetPairs), std::end(allDetPairs),
			          yrt::compare_det_pairs);
			auto u = std::unique(std::begin(allDetPairs), std::end(allDetPairs),
			                     yrt::check_det_pairs);
			bool containsDuplicate = u != std::end(allDetPairs);
			REQUIRE(!containsDuplicate);
		}

		SECTION("histo3d-bin-iterator-trial-" + std::to_string(trial + 1))
		{
			size_t num_subsets =
			    histo3d->numPhi;  // Have a subset for every angle
			for (size_t subset = 0; subset < num_subsets; subset++)
			{
				auto binIter = histo3d->getBinIter(num_subsets, subset);
				CHECK(binIter->size() == histo3d->count() / num_subsets);
				CHECK(binIter->size() == histo3d->numR * histo3d->numZBin);
				yrt::coord_t r0, phi0, z_bin0;
				histo3d->getCoordsFromBinId(binIter->get(0), r0, phi0, z_bin0);
				for (yrt::bin_t bin = 1; bin < binIter->size(); bin++)
				{
					yrt::coord_t r, phi, z_bin;
					histo3d->getCoordsFromBinId(binIter->get(bin), r, phi,
					                            z_bin);
					REQUIRE(phi == phi0);
				}
			}
		}

		SECTION("histo3d-get-lor-id-trial-" + std::to_string(trial + 1))
		{
			yrt::bin_t binId = 12;
			auto [d1_ref, d2_ref] = histo3d->getDetectorPair(binId);
			yrt::histo_bin_t lorId = histo3d->getHistogramBin(binId);
			auto newBin = std::get<yrt::bin_t>(lorId);
			yrt::det_pair_t detPair = histo3d->getDetectorPair(newBin);
			CHECK(d1_ref == detPair.d1);
			CHECK(d2_ref == detPair.d2);
		}

		SECTION("histo3d-allowed-lors-trial-" + std::to_string(trial + 1))
		{
			for (yrt::det_id_t d1 = 0; d1 < numDets; d1++)
			{
				for (yrt::det_id_t d2 = d1 + 1; d2 < numDets; d2++)
				{
					auto [doi_d1, ring_d1, posInRing_d1] =
					    getDetectorPositionInScanner(*scanner, d1);
					auto [doi_d2, ring_d2, posInRing_d2] =
					    getDetectorPositionInScanner(*scanner, d2);

					// Check ring difference
					if (std::abs(ring_d1 - ring_d2) >
					    static_cast<int64_t>(scanner->maxRingDiff))
					{
						continue;
					}

					// Check difference in ring (angle difference)
					if (std::min(yrt::util::positiveModulo(
					                 posInRing_d1 - posInRing_d2,
					                 scanner->detsPerRing),
					             yrt::util::positiveModulo(
					                 posInRing_d2 - posInRing_d1,
					                 scanner->detsPerRing)) <
					    static_cast<int64_t>(scanner->minAngDiff))
					{
						continue;
					}

					yrt::coord_t r, phi, z_bin;
					histo3d->getCoordsFromDetPair(d1, d2, r, phi, z_bin);

					yrt::det_pair_t detPair2;
					histo3d->getDetPairFromCoords(r, phi, z_bin, detPair2.d1,
					                              detPair2.d2);

					bool checkDetPair = d1 == detPair2.d1 && d2 == detPair2.d2;
					bool checkDetPairSwapped =
					    d1 == detPair2.d2 && d2 == detPair2.d1;
					CHECK((checkDetPairSwapped || checkDetPair));
				}
			}
		}
	}
}
