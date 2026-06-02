/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/operators/SiddonKernels.cuh"
#include "yrt-pet/recon/CUParameters.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include <limits>
#include <memory>
#include <vector_types.h>

TEST_CASE("siddon_gpu_vs_cpu", "[siddon-gpu]")
{
	// Create Scanner
	const auto scanner = yrt::util::test::makeFakeScanner();

	const size_t numDets = scanner->getNumDets();

	// Setup image
	constexpr ssize_t nx = 256;
	constexpr ssize_t ny = 256;
	constexpr ssize_t nz = 128;
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
		img_cpu->fill(0.0);
		yrt::util::backProject(*scanner, *img_cpu, *data,
		                       yrt::ProjectorType::SIDDON, false);

		REQUIRE(img_cpu->voxelSum() > 0.0f);

		auto img_gpu = std::make_unique<yrt::ImageOwned>(imgParams);
		img_gpu->allocate();
		img_gpu->fill(0.0);
		yrt::util::backProject(*scanner, *img_gpu, *data,
		                       yrt::ProjectorType::SIDDON, true);

		double rmseCpuGpu = yrt::util::test::getRMSE(*img_gpu, *img_cpu);

		REQUIRE(img_gpu->voxelSum() > 0.0f);
		CHECK(rmseCpuGpu < 0.0001);
	}

	auto imgToFwdProj = yrt::util::test::makeImageWithRandomPrism(imgParams);

	SECTION("fwd-project")
	{
		auto projList_cpu =
		    std::make_unique<yrt::ProjectionListOwned>(data.get());
		projList_cpu->allocate();
		projList_cpu->clearProjections(0.0f);
		yrt::util::forwProject(*scanner, *imgToFwdProj, *projList_cpu,
		                       yrt::ProjectorType::SIDDON, false);

		auto projList_gpu =
		    std::make_unique<yrt::ProjectionListOwned>(data.get());
		projList_gpu->allocate();
		projList_gpu->clearProjections(0.0f);
		yrt::util::forwProject(*scanner, *imgToFwdProj, *projList_gpu,
		                       yrt::ProjectorType::SIDDON, true);

		double rmseCpuGpu =
		    yrt::util::test::getRMSE(*projList_cpu, *projList_gpu);

		CHECK(rmseCpuGpu < 0.0008);
	}
}

__global__ void getMultiRayPos(float3 p1Init, float3 p2Init, float3 n1,
                               float3 n2, int numRays,
                               yrt::CUScannerParams scannerParams, float3* pos1,
                               float3* pos2)
{
	float3 parallelToTrans1, parallelToTrans2;
	curandState state =
	    yrt::setupMultiRays(13, 0, n1, n2, parallelToTrans1, parallelToTrans2);
	for (int i_line = 0; i_line < numRays; i_line++)
	{
		float3 p1 = p1Init;
		float3 p2 = p2Init;
		yrt::moveLineToRandomOffset<false, true>(
		    state, p1, p2, parallelToTrans1, parallelToTrans2, n1, n2,
		    scannerParams);
		pos1[i_line] = p1;
		pos2[i_line] = p2;
	}
}

TEST_CASE("multiray-gpu", "[siddon-gpu]")
{
	const auto scanner = yrt::util::test::makeFakeScanner();
	float3 p1Init{10, 0, 0};
	float3 p2Init{-10, 0, 0};
	float3 n1{-1, 0, 0};
	float3 n2{1, 0, 0};
	yrt::CUScannerParams scannerParams;
	scannerParams.crystalSize_trans = scanner->crystalSize_trans;
	scannerParams.crystalSize_z = scanner->crystalSize_z;
	scannerParams.crystalDepth = scanner->crystalDepth;
	scannerParams.numDets = 0;

	int numRays = 1000;
	yrt::GPULaunchConfig launchConfig;
	float3* pos1Device = nullptr;
	float3* pos2Device = nullptr;
	yrt::util::allocateDevice(&pos1Device, numRays, launchConfig);
	yrt::util::allocateDevice(&pos2Device, numRays, launchConfig);

	getMultiRayPos<<<1, 1>>>(p1Init, p2Init, n1, n2, numRays, scannerParams,
	                         pos1Device, pos2Device);

	auto pos1 = std::make_unique<float3[]>(numRays);
	auto pos2 = std::make_unique<float3[]>(numRays);
	yrt::util::copyDeviceToHost(pos1.get(), pos1Device, numRays, launchConfig);
	yrt::util::copyDeviceToHost(pos2.get(), pos2Device, numRays, launchConfig);

	float minX1 = std::numeric_limits<float>::infinity();
	float maxX1 = std::numeric_limits<float>::lowest();
	float minY1 = std::numeric_limits<float>::infinity();
	float maxY1 = std::numeric_limits<float>::lowest();
	float minZ1 = std::numeric_limits<float>::infinity();
	float maxZ1 = std::numeric_limits<float>::lowest();
	float minX2 = std::numeric_limits<float>::infinity();
	float maxX2 = std::numeric_limits<float>::lowest();
	float minY2 = std::numeric_limits<float>::infinity();
	float maxY2 = std::numeric_limits<float>::lowest();
	float minZ2 = std::numeric_limits<float>::infinity();
	float maxZ2 = std::numeric_limits<float>::lowest();
	for (int ri = 0; ri < numRays; ri++)
	{
		minX1 = std::min(pos1[ri].x, minX1);
		maxX1 = std::max(pos1[ri].x, maxX1);
		minY1 = std::min(pos1[ri].y, minY1);
		maxY1 = std::max(pos1[ri].y, maxY1);
		minZ1 = std::min(pos1[ri].z, minZ1);
		maxZ1 = std::max(pos1[ri].z, maxZ1);
		minX2 = std::min(pos2[ri].x, minX2);
		maxX2 = std::max(pos2[ri].x, maxX2);
		minY2 = std::min(pos2[ri].y, minY2);
		maxY2 = std::max(pos2[ri].y, maxY2);
		minZ2 = std::min(pos2[ri].z, minZ2);
		maxZ2 = std::max(pos2[ri].z, maxZ2);
	}

	// Check bounding box
	CHECK(((minX1 >= p1Init.x - scanner->crystalDepth / 2) &&
	       (minX1 <= p1Init.x + scanner->crystalDepth / 2)));
	CHECK(((maxX1 >= p1Init.x - scanner->crystalDepth / 2) &&
	       (maxX1 <= p1Init.x + scanner->crystalDepth / 2)));
	CHECK(((minY1 >= p1Init.y - scanner->crystalSize_trans / 2) &&
	       (minY1 <= p1Init.y + scanner->crystalSize_trans / 2)));
	CHECK(((maxY1 >= p1Init.y - scanner->crystalSize_trans / 2) &&
	       (maxY1 <= p1Init.y + scanner->crystalSize_trans / 2)));
	CHECK(((minZ1 >= p1Init.z - scanner->crystalSize_z / 2) &&
	       (minZ1 <= p1Init.z + scanner->crystalSize_z / 2)));
	CHECK(((maxZ1 >= p1Init.z - scanner->crystalSize_z / 2) &&
	       (maxZ1 <= p1Init.z + scanner->crystalSize_z / 2)));
	CHECK(((minX2 >= p2Init.x - scanner->crystalDepth / 2) &&
	       (minX2 <= p2Init.x + scanner->crystalDepth / 2)));
	CHECK(((maxX2 >= p2Init.x - scanner->crystalDepth / 2) &&
	       (maxX2 <= p2Init.x + scanner->crystalDepth / 2)));
	CHECK(((minY2 >= p2Init.y - scanner->crystalSize_trans / 2) &&
	       (minY2 <= p2Init.y + scanner->crystalSize_trans / 2)));
	CHECK(((maxY2 >= p2Init.y - scanner->crystalSize_trans / 2) &&
	       (maxY2 <= p2Init.y + scanner->crystalSize_trans / 2)));
	CHECK(((minZ2 >= p2Init.z - scanner->crystalSize_z / 2) &&
	       (minZ2 <= p2Init.z + scanner->crystalSize_z / 2)));
	CHECK(((maxZ2 >= p2Init.z - scanner->crystalSize_z / 2) &&
	       (maxZ2 <= p2Init.z + scanner->crystalSize_z / 2)));

	// Check extent
	CHECK(((minX1 < p1Init.x) && (maxX1 > p1Init.x)));
	CHECK(((minY1 < p1Init.y) && (maxY1 > p1Init.y)));
	CHECK(((minZ1 < p1Init.z) && (maxZ1 > p1Init.z)));
	CHECK(((minX2 < p2Init.x) && (maxX2 > p2Init.x)));
	CHECK(((minY2 < p2Init.y) && (maxY2 > p2Init.y)));
	CHECK(((minZ2 < p2Init.z) && (maxZ2 > p2Init.z)));
}
