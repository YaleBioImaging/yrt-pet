/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "datastruct/image/Image.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "utils/ReconstructionUtils.hpp"

TEST_CASE("siddon_gpu_vs_cpu", "[siddon-gpu]")
{
	// Create Scanner
	const auto scanner = TestUtils::makeScanner();

	const size_t numDets = scanner->getTheoreticalNumDets();

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
	img_cpu->setValue(0.0);
	Util::backProject(*scanner, *img_cpu, *data, OperatorProjector::SIDDON,
	                  false);

	REQUIRE(img_cpu->voxelSum() > 0.0f);

	auto img_gpu = std::make_unique<ImageOwned>(imgParams);
	img_gpu->allocate();
	img_gpu->setValue(0.0);
	Util::backProject(*scanner, *img_gpu, *data, OperatorProjector::SIDDON,
	                  true);

	double rmseCpuGpu = TestUtils::getRMSE(*img_gpu, *img_cpu);

	REQUIRE(img_gpu->voxelSum() > 0.0f);

	CHECK(rmseCpuGpu < 0.000005);

	const Image& imgToFwdProj = *img_cpu;

	auto projList_cpu = std::make_unique<ProjectionListOwned>(data.get());
	projList_cpu->allocate();
	projList_cpu->clearProjections(0.0f);
	Util::forwProject(*scanner, imgToFwdProj, *projList_cpu,
	                  OperatorProjector::SIDDON, false);

	auto projList_gpu = std::make_unique<ProjectionListOwned>(data.get());
	projList_gpu->allocate();
	projList_gpu->clearProjections(0.0f);
	Util::forwProject(*scanner, imgToFwdProj, *projList_gpu,
	                  OperatorProjector::SIDDON, true);

	rmseCpuGpu = TestUtils::getRMSE(*projList_cpu, *projList_gpu);

	CHECK(rmseCpuGpu < 0.0001);
}
