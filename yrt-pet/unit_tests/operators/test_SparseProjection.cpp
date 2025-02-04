/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/SparseHistogram.hpp"
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
		std::default_random_engine engine(
		    static_cast<unsigned int>(std::time(nullptr)));
		constexpr float MaxPrismValue = 10.0f;

		// Initialize image with prism inside
		ImageParams params{32, 32, 16, 280.0f, 280.0f, 200.0f};
		REQUIRE(params.nx == params.ny);

		std::uniform_int_distribution<int> prismPositionDistributionXY(
		    0, params.nx);
		std::uniform_int_distribution<int> prismPositionDistributionZ(
		    0, params.nz);
		std::uniform_real_distribution<float> prismValueDistribution(
		    0.0f, MaxPrismValue);

		auto image = std::make_unique<ImageOwned>(params);
		image->allocate();
		image->setValue(0.0f);

		bool mustTryAgain = false;

		int prismBeginX, prismBeginY, prismBeginZ, prismEndX, prismEndY,
		    prismEndZ;

		do
		{
			prismBeginX = prismPositionDistributionXY(engine);
			prismBeginY = prismPositionDistributionXY(engine);
			prismBeginZ = prismPositionDistributionZ(engine);
			prismEndX = prismPositionDistributionXY(engine);
			prismEndY = prismPositionDistributionXY(engine);
			prismEndZ = prismPositionDistributionZ(engine);

			auto [prismBeginX_n, prismEndX_n] =
			    std::minmax(prismBeginX, prismEndX);
			prismBeginX = prismBeginX_n;
			prismEndX = prismEndX_n;
			auto [prismBeginY_n, prismEndY_n] =
			    std::minmax(prismBeginY, prismEndY);
			prismBeginY = prismBeginY_n;
			prismEndY = prismEndY_n;
			auto [prismBeginZ_n, prismEndZ_n] =
			    std::minmax(prismBeginZ, prismEndZ);
			prismBeginZ = prismBeginZ_n;
			prismEndZ = prismEndZ_n;

			// In case randomness made it so that the prism is of null value
			mustTryAgain = prismBeginX == prismEndX ||
			               prismBeginY == prismEndY || prismBeginZ == prismEndZ;
		} while (mustTryAgain);

		REQUIRE(prismEndX > prismBeginX);
		REQUIRE(prismEndY > prismBeginY);
		REQUIRE(prismEndZ > prismBeginZ);

		float* image_ptr = image->getRawPointer();

		for (int i_x = prismBeginX; i_x < prismEndX; i_x++)
		{
			for (int i_y = prismBeginY; i_y < prismEndY; i_y++)
			{
				for (int i_z = prismBeginZ; i_z < prismEndZ; i_z++)
				{
					const size_t flatIdx = image->unravel(i_z, i_y, i_x);
					image_ptr[flatIdx] = 1.0f;
				}
			}
		}

		const float voxelSum = image->voxelSum();
		REQUIRE(voxelSum > 0.0f);

		// Initialize dense histogram
		auto histogram3D = std::make_unique<Histogram3DOwned>(*scanner);
		histogram3D->allocate();

		// Forward project into histogram using default settings (siddon, no
		//  subsets)
		std::cout << "Forward projecting into dense histogram..." << std::endl;
		Util::forwProject(*scanner, *image, *histogram3D);

		// Initialize sparse histogram
		auto sparseHistogram = std::make_unique<SparseHistogram>(*scanner);

		// Create Siddon projector with default settings (1 ray, no TOF)
		auto projector = std::make_unique<OperatorProjectorSiddon>(*scanner);

		// Forward project into sparse histogram
		std::cout << "Forward projecting into sparse histogram..." << std::endl;
		Util::forwProjectToSparseHistogram(*image, *projector,
		                                   *sparseHistogram);

		// Compare both histograms
		size_t numBins = histogram3D->count();

		std::cout << "Comparing histograms..." << std::endl;
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
	}
}
