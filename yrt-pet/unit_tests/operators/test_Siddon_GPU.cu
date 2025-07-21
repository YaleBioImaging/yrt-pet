/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

TEST_CASE("siddon_gpu_vs_cpu", "[siddon-gpu]")
{
	// Create Scanner
	const auto scanner = yrt::util::test::makeScanner();

	const size_t numDets = scanner->getNumDets();

	// Setup image
	constexpr int nx = 256;
	constexpr int ny = 256;
	constexpr int nz = 128;
	constexpr float ox = 3.0f;
	constexpr float oy = 10.0f;
	constexpr float oz = -15.0f;
	const float sx = scanner->scannerRadius * 2.0f / sqrt(2.0f);
	const float sy = scanner->scannerRadius * 2.0f / sqrt(2.0f) - oy * 2.0f;
	const float sz = scanner->axialFOV;
	yrt::ImageParams imgParams{nx, ny, nz, sx, sy, sz, ox, oy, oz};

	auto data = std::make_unique<yrt::ListModeLUTOwned>(*scanner);
	constexpr size_t numEvents = 10000;
	data->allocate(numEvents);

	for (yrt::bin_t binId = 0; binId < numEvents; binId++)
	{
		const yrt::det_id_t d1 = rand() % numDets;
		const yrt::det_id_t d2 = rand() % numDets;
		data->setDetectorIdsOfEvent(binId, d1, d2);
	}

	SECTION("bwd-project")
	{
		auto img_cpu = std::make_unique<yrt::ImageOwned>(imgParams);
		img_cpu->allocate();
		img_cpu->setValue(0.0);
		yrt::util::backProject(*scanner, *img_cpu, *data, yrt::OperatorProjector::SIDDON,
		                  false);

		REQUIRE(img_cpu->voxelSum() > 0.0f);

		auto img_gpu = std::make_unique<yrt::ImageOwned>(imgParams);
		img_gpu->allocate();
		img_gpu->setValue(0.0);
		yrt::util::backProject(*scanner, *img_gpu, *data, yrt::OperatorProjector::SIDDON,
		                  true);

		double rmseCpuGpu = yrt::util::test::getRMSE(*img_gpu, *img_cpu);

		REQUIRE(img_gpu->voxelSum() > 0.0f);
		CHECK(rmseCpuGpu < 0.000007);
	}

	auto imgToFwdProj = yrt::util::test::makeImageWithRandomPrism(imgParams);

	SECTION("fwd-project")
	{
		auto projList_cpu = std::make_unique<yrt::ProjectionListOwned>(data.get());
		projList_cpu->allocate();
		projList_cpu->clearProjections(0.0f);
		yrt::util::forwProject(*scanner, *imgToFwdProj, *projList_cpu,
		                  yrt::OperatorProjector::SIDDON, false);

		auto projList_gpu = std::make_unique<yrt::ProjectionListOwned>(data.get());
		projList_gpu->allocate();
		projList_gpu->clearProjections(0.0f);
		yrt::util::forwProject(*scanner, *imgToFwdProj, *projList_gpu,
		                  yrt::OperatorProjector::SIDDON, true);

		double rmseCpuGpu = yrt::util::test::getRMSE(*projList_cpu, *projList_gpu);

		CHECK(rmseCpuGpu < 0.0003);
	}
}
