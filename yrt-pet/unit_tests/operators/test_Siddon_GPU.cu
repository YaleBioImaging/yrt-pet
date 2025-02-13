/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "datastruct/image/Image.hpp"
#include "datastruct/projection/ListMode.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "operators/OperatorProjectorSiddon.hpp"
#include "utils/Assert.hpp"
#include "utils/ReconstructionUtils.hpp"

TEST_CASE("siddon_gpu_vs_cpu", "[siddon-gpu]")
{
	// Create Scanner
	const auto scanner = TestUtils::makeScanner();

	const size_t numDets = scanner->getTheoreticalNumDets();

	// Create some image
	// Setup image
	constexpr int nx = 256;
	constexpr int ny = 256;
	constexpr int nz = 128;
	const float sx = scanner->scannerRadius * 2.0f / sqrt(2.0f);
	const float sy = scanner->scannerRadius * 2.0f / sqrt(2.0f);
	const float sz = scanner->axialFOV;
	constexpr float ox = 3.0f;
	constexpr float oy = 0.0f;
	constexpr float oz = -10.0f;
	ImageParams imgParams{nx, ny, nz, sx, sy, sz, ox, oy, oz};
	auto img = std::make_unique<ImageOwned>(imgParams);
	img->allocate();

	auto data = std::make_unique<ListModeLUTOwned>(*scanner);
	constexpr size_t numEvents = 1000;
	data->allocate(numEvents);

	for (bin_t binId = 0; binId < numEvents; binId++)
	{
		const det_id_t d1 = rand() % numDets;
		const det_id_t d2 = rand() % numDets;
		data->setDetectorIdsOfEvent(binId, d1, d2);
	}

	auto img_cpu = std::make_unique<ImageOwned>(imgParams);
	img_cpu->allocate();

	auto img_gpu = std::make_unique<ImageOwned>(imgParams);
	img_gpu->allocate();

	// TODO NOW: Do backprojection and compare
}
