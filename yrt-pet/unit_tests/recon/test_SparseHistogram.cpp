/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "datastruct/projection/SparseHistogram.hpp"
#include "datastruct/scanner/DetRegular.hpp"
#include "test_utils.hpp"


TEST_CASE("sparsehisto", "[sparsehisto]")
{
	auto scanner = TestUtils::makeScanner();

	SECTION("from-listmodes")
	{
		auto listMode = std::make_unique<ListModeLUTOwned>(*scanner);
		listMode->allocate(10);
		listMode->setDetectorIdsOfEvent(0, 0, 15);   // 1st
		listMode->setDetectorIdsOfEvent(1, 10, 15);  // 1st
		listMode->setDetectorIdsOfEvent(2, 0, 15);   // 2nd
		listMode->setDetectorIdsOfEvent(3, 12, 78);  // 1st
		listMode->setDetectorIdsOfEvent(4, 10, 15);  // 2nd
		listMode->setDetectorIdsOfEvent(5, 0, 20);   // 1st
		listMode->setDetectorIdsOfEvent(6, 48, 21);  // 1st
		listMode->setDetectorIdsOfEvent(7, 0, 15);   // 3rd
		listMode->setDetectorIdsOfEvent(8, 10, 13);  // 1st
		listMode->setDetectorIdsOfEvent(9, 20, 0);   // 2nd
		auto sparseHisto =
		    std::make_unique<SparseHistogram>(*scanner, *listMode);
		CHECK(sparseHisto->getProjectionValueFromDetPair({0, 20}) == 2.0f);
		CHECK(sparseHisto->getProjectionValueFromDetPair({10, 13}) == 1.0f);
		CHECK(sparseHisto->getProjectionValueFromDetPair({0, 15}) == 3.0f);
		CHECK(sparseHisto->getProjectionValueFromDetPair({48, 21}) == 1.0f);
		CHECK(sparseHisto->getProjectionValueFromDetPair({10, 15}) == 2.0f);
		CHECK(sparseHisto->getProjectionValueFromDetPair({78, 12}) == 1.0f);

		listMode = std::make_unique<ListModeLUTOwned>(*scanner);
		listMode->allocate(3);
		listMode->setDetectorIdsOfEvent(0, 0, 15);   // 4th
		listMode->setDetectorIdsOfEvent(1, 10, 15);  // 3rd
		listMode->setDetectorIdsOfEvent(2, 0, 15);   // 5th

		sparseHisto->accumulate(*listMode);
		CHECK(sparseHisto->getProjectionValueFromDetPair({0, 15}) == 5.0f);
		CHECK(sparseHisto->getProjectionValueFromDetPair({10, 15}) == 3.0f);
	}

	SECTION("from-sparsehistos")
	{
		auto sparseHisto1 = std::make_unique<SparseHistogram>(*scanner);
		sparseHisto1->accumulate({15, 10}, 100.f);
		sparseHisto1->accumulate({12, 10}, 100.f);
		sparseHisto1->accumulate({10, 12}, 100.f);
		sparseHisto1->accumulate({5, 10}, 100.f);

		auto sparseHistoTotal = std::make_unique<SparseHistogram>(*scanner);
		sparseHistoTotal->accumulate(*sparseHisto1);

		auto sparseHisto2 = std::make_unique<SparseHistogram>(*scanner);
		sparseHisto2->accumulate({5, 20}, 10.0f);
		sparseHisto2->accumulate({15, 10}, 10.0f);
		sparseHisto2->accumulate({12, 10}, 10.0f);
		sparseHisto2->accumulate({120, 100}, 10.0f);

		sparseHistoTotal->accumulate(*sparseHisto2);

		REQUIRE(sparseHistoTotal->count() == 5);

		CHECK(sparseHistoTotal->getProjectionValueFromHistogramBin(
		          det_pair_t{15, 10}) == Approx(110.0f));
		CHECK(sparseHistoTotal->getProjectionValueFromHistogramBin(
		          det_pair_t{12, 10}) == Approx(210.0f));
		CHECK(sparseHistoTotal->getProjectionValueFromHistogramBin(
		          det_pair_t{5, 10}) == Approx(100.0f));
		CHECK(sparseHistoTotal->getProjectionValueFromHistogramBin(
		          det_pair_t{5, 20}) == Approx(10.0f));
		CHECK(sparseHistoTotal->getProjectionValueFromHistogramBin(
		          det_pair_t{100, 120}) == Approx(10.0f));
	}

	SECTION("from-histogram3d")
	{
		auto histo = std::make_unique<Histogram3DOwned>(*scanner);
		histo->allocate();
		const size_t numBins = histo->count();

		for (bin_t bin = 0; bin < numBins; bin++)
		{
			histo->setProjectionValue(bin, static_cast<float>(rand() % 10 + 1));
		}

		auto sparseHisto = std::make_unique<SparseHistogram>(*scanner, *histo);

		// Because Histogram3D also only has "unique" LORs
		REQUIRE(sparseHisto->count() == histo->count());

		for (bin_t i = 0; i < sparseHisto->count(); i++)
		{
			const det_pair_t detPair = sparseHisto->getDetectorPair(i);
			CHECK(sparseHisto->getProjectionValueFromDetPair(detPair) ==
			      histo->getProjectionValueFromHistogramBin(detPair));
		}
	}

	SECTION("read-write")
	{
		int random_seed = time(0);
		srand(random_seed);
		const det_id_t numDets = static_cast<det_id_t>(scanner->getNumDets());

		auto sparseHisto = std::make_unique<SparseHistogram>(*scanner);
		constexpr size_t NumBins = 100;
		sparseHisto->allocate(NumBins);
		// Generate sparse histo with random data
		for (size_t i = 0; i < NumBins; i++)
		{
			sparseHisto->accumulate(
			    det_pair_t{rand() % numDets, rand() % numDets},
			    static_cast<float>(rand() % 10 + 1));
		}
		std::string filename = "mysparsehisto.shis";

		sparseHisto->writeToFile(filename);
		auto sparseHistoFromFile =
		    std::make_unique<SparseHistogram>(*scanner, filename);

		REQUIRE(sparseHistoFromFile->count() == sparseHisto->count());
		for (bin_t i = 0; i < sparseHisto->count(); i++)
		{
			CHECK(sparseHisto->getProjectionValue(i) ==
			      sparseHistoFromFile->getProjectionValue(i));
		}

		std::remove(filename.c_str());
	}
}
