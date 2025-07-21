/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/operators/OperatorProjectorDD.hpp"
#include "yrt-pet/operators/OperatorProjectorSiddon.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#include <algorithm>
#include <utility>

#if BUILD_CUDA
#include "yrt-pet/recon/OSEM_GPU.cuh"
#endif

TEST_CASE("DD-simple", "[dd]")
{
	SECTION("get_overlap")
	{
		CHECK(yrt::OperatorProjectorDD::get_overlap(1.1, 4.1, 2.1, 3.1) ==
		      Approx(1.0));
		CHECK(yrt::OperatorProjectorDD::get_overlap(4, 1, 2, 3) == Approx(0.0));
		CHECK(yrt::OperatorProjectorDD::get_overlap(4.5, 2.3, 1.6, 3.2) ==
		      Approx(0.0));
		CHECK(yrt::OperatorProjectorDD::get_overlap(1.1, 1.2, 1.3, 1.4) ==
		      Approx(0.0));
		CHECK(yrt::OperatorProjectorDD::get_overlap(1.4, 1.3, 1.1, 1.2) ==
		      Approx(0.0));
		CHECK(yrt::OperatorProjectorDD::get_overlap(9.2, 10.9, 8.3, 10.0) ==
		      Approx(10.0 - 9.2));
		CHECK(yrt::OperatorProjectorDD::get_overlap(9.2, 9.9, 8.3, 10.0) ==
		      Approx(9.9 - 9.2));
	}
}

TEST_CASE("DD", "[dd]")
{
#if BUILD_CUDA
	srand(13);

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
	auto img = std::make_unique<yrt::ImageOwned>(imgParams);
	img->allocate();

	auto data = std::make_unique<yrt::ListModeLUTOwned>(*scanner);
	constexpr size_t numEvents = 10000;
	data->allocate(numEvents);
	for (yrt::bin_t binId = 0; binId < numEvents; binId++)
	{
		const yrt::det_id_t d1 = rand() % numDets;
		const yrt::det_id_t d2 = rand() % numDets;
		data->setDetectorIdsOfEvent(binId, d1, d2);
	}

	// Helper aliases
	using ImageSharedPTR = std::shared_ptr<yrt::Image>;
	const auto toOwned = [](const ImageSharedPTR& i)
	{ return reinterpret_cast<yrt::ImageOwned*>(i.get()); };

	const ImageSharedPTR img_cpu = std::make_shared<yrt::ImageOwned>(imgParams);
	toOwned(img_cpu)->allocate();
	img_cpu->setValue(0.0);
	yrt::util::backProject(*scanner, *img_cpu, *data, yrt::OperatorProjector::DD, false);

	REQUIRE(img_cpu->voxelSum() > 0.0f);

	const ImageSharedPTR img_gpu = std::make_shared<yrt::ImageOwned>(imgParams);
	toOwned(img_gpu)->allocate();
	img_gpu->setValue(0.0);
	yrt::util::backProject(*scanner, *img_gpu, *data, yrt::OperatorProjector::DD, true);

	REQUIRE(img_gpu->voxelSum() > 0.0f);

	double rmseCpuGpu = yrt::util::test::getRMSE(*img_gpu, *img_cpu);

	CHECK(rmseCpuGpu < 0.000005);

	const yrt::Image& imgToFwdProj = *img_cpu;

	auto projList_cpu = std::make_unique<yrt::ProjectionListOwned>(data.get());
	projList_cpu->allocate();
	projList_cpu->clearProjections(0.0f);
	yrt::util::forwProject(*scanner, imgToFwdProj, *projList_cpu,
	                  yrt::OperatorProjector::DD, false);

	auto projList_gpu = std::make_unique<yrt::ProjectionListOwned>(data.get());
	projList_gpu->allocate();
	projList_gpu->clearProjections(0.0f);
	yrt::util::forwProject(*scanner, imgToFwdProj, *projList_gpu,
	                  yrt::OperatorProjector::DD, true);

	rmseCpuGpu = yrt::util::test::getRMSE(*projList_cpu, *projList_gpu);

	CHECK(rmseCpuGpu < 0.0004);

#endif
}
