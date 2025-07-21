/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/SparseHistogram.hpp"
#include "yrt-pet/operators/OperatorProjectorDD.hpp"
#include "yrt-pet/operators/OperatorProjectorSiddon.hpp"
#include "yrt-pet/operators/SparseProjection.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#include "../test_utils.hpp"

#include "catch.hpp"


TEST_CASE("sparse-projection", "[SparseProjection]")
{
	auto scanner = yrt::util::test::makeScanner();

	SECTION("against-dense-histogram")
	{
		yrt::ImageParams imgParams{64, 64, 16, 280.0f, 280.0f, 200.0f};
		auto image = yrt::util::test::makeImageWithRandomPrism(imgParams);

		// Initialize dense histogram
		auto histogram3D = std::make_unique<yrt::Histogram3DOwned>(*scanner);
		histogram3D->allocate();

		// Forward project into histogram using default settings (siddon, no
		//  subsets)
		std::cout << "Forward projecting into dense histogram..." << std::endl;
		yrt::util::forwProject(*scanner, *image, *histogram3D,
		                  yrt::OperatorProjector::ProjectorType::DD);

		// Initialize sparse histogram
		auto sparseHistogram = std::make_unique<yrt::SparseHistogram>(*scanner);

		// Create DD projector with default settings (no PSF, no TOF)
		auto projector = std::make_unique<yrt::OperatorProjectorDD>(*scanner);

		// Forward project into sparse histogram
		std::cout << "Forward projecting into sparse histogram..." << std::endl;
		yrt::util::forwProjectToSparseHistogram(*image, *projector,
		                                   *sparseHistogram);

		// Compare both histograms
		size_t numBins = histogram3D->count();

		std::cout << "Comparing sparse histogram with dense histogram..."
		          << std::endl;
		for (yrt::bin_t bin = 0; bin < numBins; bin++)
		{
			const float histogram3DProjValue =
			    histogram3D->getProjectionValue(bin);

			const yrt::det_pair_t detPair = histogram3D->getDetectorPair(bin);
			const float sparseHistogramProjValue =
			    sparseHistogram->getProjectionValueFromDetPair(detPair);

			CHECK(sparseHistogramProjValue == Approx(histogram3DProjValue));
		}

		// Accumulating dense histogram into another sparse histogram
		auto sparseHistogram2 = std::make_unique<yrt::SparseHistogram>(*scanner);
		sparseHistogram2->accumulate(*histogram3D);

		// Comparing both sparse histograms
		REQUIRE(sparseHistogram2->count() == sparseHistogram->count());
		numBins = sparseHistogram->count();

		std::cout << "Comparing sparse histograms..." << std::endl;
		for (yrt::bin_t bin = 0; bin < numBins; bin++)
		{
			const yrt::det_pair_t detPair = sparseHistogram->getDetectorPair(bin);

			const float sparseHistogramProjValue =
			    sparseHistogram->getProjectionValueFromDetPair(detPair);
			const float sparseHistogram2ProjValue =
			    sparseHistogram2->getProjectionValueFromDetPair(detPair);

			CHECK(sparseHistogramProjValue ==
			      Approx(sparseHistogram2ProjValue));
		}


		// Accumulating sparse histogram into another sparse histogram
		auto sparseHistogram3 = std::make_unique<yrt::SparseHistogram>(*scanner);
		sparseHistogram3->accumulate(*sparseHistogram);

		// Comparing both sparse histograms
		REQUIRE(sparseHistogram3->count() == sparseHistogram->count());
		numBins = sparseHistogram->count();

		std::cout << "Comparing sparse histograms..." << std::endl;
		for (yrt::bin_t bin = 0; bin < numBins; bin++)
		{
			const yrt::det_pair_t detPair = sparseHistogram->getDetectorPair(bin);

			const float sparseHistogramProjValue =
			    sparseHistogram->getProjectionValueFromDetPair(detPair);
			const float sparseHistogram3ProjValue =
			    sparseHistogram3->getProjectionValueFromDetPair(detPair);

			CHECK(sparseHistogramProjValue ==
			      Approx(sparseHistogram3ProjValue));
		}

		// Accumulating sparse histogram into dense histogram
		auto histogram3D2 = std::make_unique<yrt::Histogram3DOwned>(*scanner);
		histogram3D2->allocate();
		yrt::util::convertToHistogram3D<false>(*sparseHistogram, *histogram3D2);

		// Comparing both dense histograms
		numBins = histogram3D2->count();

		std::cout << "Comparing dense histograms..." << std::endl;
		for (yrt::bin_t bin = 0; bin < numBins; bin++)
		{
			const float histogram3DProjValue =
			    histogram3D2->getProjectionValue(bin);

			const yrt::det_pair_t detPair = histogram3D2->getDetectorPair(bin);
			const float sparseHistogramProjValue =
			    sparseHistogram->getProjectionValueFromDetPair(detPair);

			CHECK(sparseHistogramProjValue == Approx(histogram3DProjValue));
		}
	}
}
