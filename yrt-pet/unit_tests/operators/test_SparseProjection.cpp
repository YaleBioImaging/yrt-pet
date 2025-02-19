/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/SparseHistogram.hpp"
#include "operators/OperatorProjectorDD.hpp"
#include "operators/OperatorProjectorSiddon.hpp"
#include "operators/SparseProjection.hpp"
#include "utils/ReconstructionUtils.hpp"

#include "../test_utils.hpp"

#include "catch.hpp"


TEST_CASE("sparse-projection", "[SparseProjection]")
{
	auto scanner = TestUtils::makeScanner();

	SECTION("against-dense-histogram")
	{
		ImageParams imgParams{64, 64, 16, 280.0f, 280.0f, 200.0f};
		auto image = TestUtils::makeImageWithRandomPrism(imgParams);

		// Initialize dense histogram
		auto histogram3D = std::make_unique<Histogram3DOwned>(*scanner);
		histogram3D->allocate();

		// Forward project into histogram using default settings (siddon, no
		//  subsets)
		std::cout << "Forward projecting into dense histogram..." << std::endl;
		Util::forwProject(*scanner, *image, *histogram3D,
		                  OperatorProjector::ProjectorType::DD);

		// Initialize sparse histogram
		auto sparseHistogram = std::make_unique<SparseHistogram>(*scanner);

		// Create DD projector with default settings (no PSF, no TOF)
		auto projector = std::make_unique<OperatorProjectorDD>(*scanner);

		// Forward project into sparse histogram
		std::cout << "Forward projecting into sparse histogram..." << std::endl;
		Util::forwProjectToSparseHistogram(*image, *projector,
		                                   *sparseHistogram);

		// Compare both histograms
		size_t numBins = histogram3D->count();

		std::cout << "Comparing sparse histogram with dense histogram..."
		          << std::endl;
		for (bin_t bin = 0; bin < numBins; bin++)
		{
			const float histogram3DProjValue =
			    histogram3D->getProjectionValue(bin);

			const det_pair_t detPair = histogram3D->getDetectorPair(bin);
			const float sparseHistogramProjValue =
			    sparseHistogram->getProjectionValueFromDetPair(detPair);

			CHECK(sparseHistogramProjValue == Approx(histogram3DProjValue));
		}

		// Accumulating dense histogram into another sparse histogram
		auto sparseHistogram2 = std::make_unique<SparseHistogram>(*scanner);
		sparseHistogram2->accumulate(*histogram3D);

		// Comparing both sparse histograms
		REQUIRE(sparseHistogram2->count() == sparseHistogram->count());
		numBins = sparseHistogram->count();

		std::cout << "Comparing sparse histograms..." << std::endl;
		for (bin_t bin = 0; bin < numBins; bin++)
		{
			const det_pair_t detPair = sparseHistogram->getDetectorPair(bin);

			const float sparseHistogramProjValue =
			    sparseHistogram->getProjectionValueFromDetPair(detPair);
			const float sparseHistogram2ProjValue =
			    sparseHistogram2->getProjectionValueFromDetPair(detPair);

			CHECK(sparseHistogramProjValue ==
			      Approx(sparseHistogram2ProjValue));
		}


		// Accumulating sparse histogram into another sparse histogram
		auto sparseHistogram3 = std::make_unique<SparseHistogram>(*scanner);
		sparseHistogram3->accumulate(*sparseHistogram);

		// Comparing both sparse histograms
		REQUIRE(sparseHistogram3->count() == sparseHistogram->count());
		numBins = sparseHistogram->count();

		std::cout << "Comparing sparse histograms..." << std::endl;
		for (bin_t bin = 0; bin < numBins; bin++)
		{
			const det_pair_t detPair = sparseHistogram->getDetectorPair(bin);

			const float sparseHistogramProjValue =
			    sparseHistogram->getProjectionValueFromDetPair(detPair);
			const float sparseHistogram3ProjValue =
			    sparseHistogram3->getProjectionValueFromDetPair(detPair);

			CHECK(sparseHistogramProjValue ==
			      Approx(sparseHistogram3ProjValue));
		}

		// Accumulating sparse histogram into dense histogram
		auto histogram3D2 = std::make_unique<Histogram3DOwned>(*scanner);
		histogram3D2->allocate();
		Util::convertToHistogram3D<false>(*sparseHistogram, *histogram3D2);

		// Comparing both dense histograms
		numBins = histogram3D2->count();

		std::cout << "Comparing dense histograms..." << std::endl;
		for (bin_t bin = 0; bin < numBins; bin++)
		{
			const float histogram3DProjValue =
			    histogram3D2->getProjectionValue(bin);

			const det_pair_t detPair = histogram3D2->getDetectorPair(bin);
			const float sparseHistogramProjValue =
			    sparseHistogram->getProjectionValueFromDetPair(detPair);

			CHECK(sparseHistogramProjValue == Approx(histogram3DProjValue));
		}
	}
}
