/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "datastruct/scanner/DetRegular.hpp"
#include "test_utils.hpp"
#include "utils/ReconstructionUtils.hpp"


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

TEST_CASE("histo3d", "[histo]")
{
	// TODO: Randomize scanner generation
	auto scanner = std::make_unique<Scanner>("FakeScanner", 200, 1, 1, 10, 200,
	                                         48, 12, 2, 8, 1, 4);
	const auto detRegular = std::make_shared<DetRegular>(scanner.get());
	detRegular->generateLUT();
	scanner->setDetectorSetup(detRegular);

	size_t n_total_detectors =
	    scanner->numDOI * scanner->numRings * scanner->detsPerRing;
	auto histo3d = std::make_unique<Histogram3DOwned>(*scanner);

	SECTION("histo3d-binIds")
	{
		for (bin_t binId = 0; binId < histo3d->count(); binId++)
		{
			coord_t r, phi, z_bin;
			histo3d->getCoordsFromBinId(binId, r, phi, z_bin);
			bin_t supposedBin = histo3d->getBinIdFromCoords(r, phi, z_bin);
			REQUIRE(supposedBin == binId);
		}
	}

	SECTION("histo3d-coords-binning")
	{
		coord_t r, phi, z_bin;
		for (r = 0; r < histo3d->numR; r++)
		{
			for (phi = 0; phi < histo3d->numPhi; phi++)
			{
				for (z_bin = 0; z_bin < histo3d->numZBin; z_bin++)
				{
					det_id_t d1, d2;
					coord_t r_supp, phi_supp, z_bin_supp;
					histo3d->getDetPairFromCoords(r, phi, z_bin, d1, d2);
					histo3d->getCoordsFromDetPair(d1, d2, r_supp, phi_supp,
					                              z_bin_supp);
					det_id_t d1_supp, d2_supp;
					histo3d->getDetPairFromCoords(r_supp, phi_supp, z_bin_supp,
					                              d1_supp, d2_supp);

					bool check = (d1_supp == d1 && d2_supp == d2) ||
					             (d1_supp == d2 && d2_supp == d1);
					REQUIRE(check);
					REQUIRE(d1 < n_total_detectors);
					REQUIRE(d2 < n_total_detectors);
				}
			}
		}
	}

	SECTION("histo3d-detector-swap")
	{
		coord_t r, phi, z_bin;
		for (r = 0; r < histo3d->numR; r++)
		{
			for (phi = 0; phi < histo3d->numPhi; phi++)
			{
				for (z_bin = 0; z_bin < histo3d->numZBin; z_bin++)
				{
					det_id_t d1, d2;
					coord_t r_supp, phi_supp, z_bin_supp;
					histo3d->getDetPairFromCoords(r, phi, z_bin, d1, d2);
					histo3d->getCoordsFromDetPair(d2, d1, r_supp, phi_supp,
					                              z_bin_supp);
					CHECK(r_supp == r);
					CHECK(phi_supp == phi);
					CHECK(z_bin_supp == z_bin);
				}
			}
		}
	}


	SECTION("histo3d-coords-uniqueness")
	{
		std::vector<std::array<coord_t, 3>> all_coords;

		for (det_id_t d1 = 0; d1 < n_total_detectors; d1++)
		{
			for (det_id_t d2 = d1 + 1; d2 < n_total_detectors; d2++)
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
				int z1 = (d1 / (scanner->detsPerRing)) % (scanner->numRings);
				int z2 = (d2 / (scanner->detsPerRing)) % (scanner->numRings);
				if (std::abs(z1 - z2) > scanner->maxRingDiff)
				{
					continue;
				}

				coord_t r, phi, z_bin;
				histo3d->getCoordsFromDetPair(d1, d2, r, phi, z_bin);
				all_coords.push_back({r, phi, z_bin});
			}
		}
		std::sort(std::begin(all_coords), std::end(all_coords), compare_coords);
		auto u = std::unique(std::begin(all_coords), std::end(all_coords),
		                     check_coords);
		bool containsDuplicate = u != std::end(all_coords);
		REQUIRE(!containsDuplicate);
	}

	SECTION("histo3d-detector-pairs-uniqueness")
	{
		std::vector<det_pair_t> allDetPairs;

		const size_t numBins = histo3d->count();
		allDetPairs.resize(numBins);

		for (bin_t binId = 0; binId < numBins; binId++)
		{
			det_pair_t currPair = histo3d->getDetectorPair(binId);
			allDetPairs[binId] = currPair;
		}

		std::sort(std::begin(allDetPairs), std::end(allDetPairs),
		          compare_det_pairs);
		auto u = std::unique(std::begin(allDetPairs), std::end(allDetPairs),
		                     check_det_pairs);
		bool containsDuplicate = u != std::end(allDetPairs);
		REQUIRE(!containsDuplicate);
	}

	SECTION("histo3d-line-integrity")
	{
		auto someListMode = std::make_unique<ListModeLUTOwned>(*scanner);
		double epsilon = 1e-5;
		someListMode->allocate(3);
		someListMode->setDetectorIdsOfEvent(0, 15, 105);
		someListMode->setDetectorIdsOfEvent(1, 238, 75);
		someListMode->setDetectorIdsOfEvent(2, 215, 110);
		for (size_t lmEv = 0; lmEv < someListMode->count(); lmEv++)
		{
			auto [d1, d2] = someListMode->getDetectorPair(lmEv);
			Line3D lor = Util::getNativeLOR(*scanner, *someListMode, lmEv);
			bin_t binId = histo3d->getBinIdFromDetPair(d1, d2);
			Line3D histoLor = Util::getNativeLOR(*scanner, *histo3d, binId);

			CHECK(((std::abs(histoLor.point1.x - lor.point1.x) < epsilon &&
			        std::abs(histoLor.point1.y - lor.point1.y) < epsilon &&
			        std::abs(histoLor.point1.z - lor.point1.z) < epsilon &&
			        std::abs(histoLor.point2.x - lor.point2.x) < epsilon &&
			        std::abs(histoLor.point2.y - lor.point2.y) < epsilon &&
			        std::abs(histoLor.point2.z - lor.point2.z) < epsilon) ||
			       (std::abs(histoLor.point1.x - lor.point2.x) < epsilon &&
			        std::abs(histoLor.point1.y - lor.point2.y) < epsilon &&
			        std::abs(histoLor.point1.z - lor.point2.z) < epsilon &&
			        std::abs(histoLor.point2.x - lor.point1.x) < epsilon &&
			        std::abs(histoLor.point2.y - lor.point1.y) < epsilon &&
			        std::abs(histoLor.point2.z - lor.point1.z) < epsilon)));
		}
	}

	SECTION("histo3d-bin-iterator")
	{
		size_t num_subsets = histo3d->numPhi;  // Have a subset for every angle
		for (size_t subset = 0; subset < num_subsets; subset++)
		{
			auto binIter = histo3d->getBinIter(num_subsets, subset);
			CHECK(binIter->size() == histo3d->count() / num_subsets);
			CHECK(binIter->size() == histo3d->numR * histo3d->numZBin);
			coord_t r0, phi0, z_bin0;
			histo3d->getCoordsFromBinId(binIter->get(0), r0, phi0, z_bin0);
			for (bin_t bin = 1; bin < binIter->size(); bin++)
			{
				coord_t r, phi, z_bin;
				histo3d->getCoordsFromBinId(binIter->get(bin), r, phi, z_bin);
				REQUIRE(phi == phi0);
			}
		}
	}

	SECTION("histo3d-get-lor-id")
	{
		bin_t binId = 12;
		auto [d1_ref, d2_ref] = histo3d->getDetectorPair(binId);
		histo_bin_t lorId = histo3d->getHistogramBin(binId);
		auto newBin = std::get<bin_t>(lorId);
		det_pair_t detPair = histo3d->getDetectorPair(newBin);
		CHECK(d1_ref == detPair.d1);
		CHECK(d2_ref == detPair.d2);
	}

	SECTION("histo3d-allowed-lors")
	{
		det_id_t numDets = scanner->getNumDets();
		for (det_id_t d1 = 0; d1 < numDets; d1++)
		{
			for (det_id_t d2 = d1 + 1; d2 < numDets; d2++)
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
				if (std::min(Util::positiveModulo(posInRing_d1 - posInRing_d2,
				                                  scanner->detsPerRing),
				             Util::positiveModulo(posInRing_d2 - posInRing_d1,
				                                  scanner->detsPerRing)) <
				    static_cast<int64_t>(scanner->minAngDiff))
				{
					continue;
				}

				coord_t r, phi, z_bin;
				histo3d->getCoordsFromDetPair(d1, d2, r, phi, z_bin);

				det_pair_t detPair2;
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
