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

	SECTION("from-listmode")
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
	}

	SECTION("from-histogram3d")
	{
		auto histo = std::make_unique<Histogram3DOwned>(*scanner);
		histo->allocate();
		histo->clearProjections(1.0f);

		auto sparseHisto = std::make_unique<SparseHistogram>(*scanner, *histo);
		// Because Histogram3D also only has "unique" LORs
		REQUIRE(sparseHisto->count() == histo->count());

		for (bin_t i = 0; i < sparseHisto->count(); i++)
		{
			CHECK(sparseHisto->getProjectionValue(i) == 1.0f);
		}
	}

	SECTION("from-histogram3d-with-biniterator")
	{
		auto histo = std::make_unique<Histogram3DOwned>(*scanner);
		histo->allocate();
		auto binIter = histo->getBinIter(5, 2);

		int random_seed = time(0);
		srand(random_seed);
		for (bin_t bin = 0; bin < binIter->size(); bin++)
		{
			bin_t binId = binIter->get(bin);
			histo->setProjectionValue(binId, static_cast<float>(rand() % 100));
		}

		auto sparseHisto =
		    std::make_unique<SparseHistogram>(*scanner, *histo, binIter.get());
		// Because Histogram3D also only has "unique" LORs
		REQUIRE(sparseHisto->count() == binIter->size());

		for (bin_t bin = 0; bin < binIter->size(); bin++)
		{
			bin_t binId = binIter->get(bin);
			float origProjValue = histo->getProjectionValue(binId);
			CHECK(origProjValue == sparseHisto->getProjectionValue(bin));
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
			    static_cast<float>(rand() % 10));
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
