/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "datastruct/image/Image.hpp"
#include "datastruct/projection/ListMode.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "datastruct/scanner/DetRegular.hpp"
#include "operators/OperatorProjectorDD.hpp"
#include "operators/OperatorProjectorSiddon.hpp"
#include "utils/ReconstructionUtils.hpp"

#include <algorithm>
#include <utility>

#if BUILD_CUDA
#include "recon/OSEM_GPU.cuh"
#endif


double get_rmse(const Image* img_ref, const Image* img)
{
	const ImageParams& params = img_ref->getParams();
	const size_t numPixels =
	    static_cast<size_t>(params.nx * params.ny * params.nz);
	const double* ptr_ref = img_ref->getData().getRawPointer();
	const double* ptr = img->getData().getRawPointer();
	double rmse = 0.0;

	for (size_t i = 0; i < numPixels; i++)
	{
		rmse += std::pow(ptr_ref[i] - ptr[i], 2.0);
	}

	rmse = std::sqrt(rmse / static_cast<double>(numPixels));

	return rmse;
}

void dd(const Scanner& scanner, ListMode* proj, std::shared_ptr<Image>& out,
        const bool flag_cuda)
{
	const auto osem = Util::createOSEM(scanner, flag_cuda);
	osem->setImageParams(out->getParams());
	osem->num_OSEM_subsets = 1;
	osem->setSensDataInput(proj);
	if (flag_cuda)
	{
		osem->projectorType = OperatorProjector::DD_GPU;
	}
	else
	{
		osem->projectorType = OperatorProjector::DD;
	}
	std::vector<std::unique_ptr<Image>> sensImages;
	osem->generateSensitivityImages(sensImages, "");
	out = std::move(sensImages[0]);
}

TEST_CASE("DD-simple", "[dd]")
{
	SECTION("get_overlap")
	{
		CHECK(OperatorProjectorDD::get_overlap(1.1, 4.1, 2.1, 3.1) ==
		      Approx(1.0));
		CHECK(OperatorProjectorDD::get_overlap(4, 1, 2, 3) == Approx(0.0));
		CHECK(OperatorProjectorDD::get_overlap(4.5, 2.3, 1.6, 3.2) ==
		      Approx(0.0));
		CHECK(OperatorProjectorDD::get_overlap(1.1, 1.2, 1.3, 1.4) ==
		      Approx(0.0));
		CHECK(OperatorProjectorDD::get_overlap(1.4, 1.3, 1.1, 1.2) ==
		      Approx(0.0));
		CHECK(OperatorProjectorDD::get_overlap(9.2, 10.9, 8.3, 10.0) ==
		      Approx(10.0 - 9.2));
		CHECK(OperatorProjectorDD::get_overlap(9.2, 9.9, 8.3, 10.0) ==
		      Approx(9.9 - 9.2));
	}
}

TEST_CASE("DD", "[dd]")
{
#if BUILD_CUDA
	srand(13);

	// Create Scanner
	const auto scanner = TestUtils::makeScanner();

	const size_t numDets = scanner->getTheoreticalNumDets();

	// Create some image
	// Setup image
	const int nx = 100;
	const int ny = 100;
	const int nz = 100;
	const double sx = 256.0;
	const double sy = 256.0;
	const double sz = 96.0;
	const double ox = 0.0;
	const double oy = 0.0;
	const double oz = 0.0;
	ImageParams img_params(nx, ny, nz, sx, sy, sz, ox, oy, oz);
	auto img = std::make_unique<ImageOwned>(img_params);
	img->allocate();

	auto data = std::make_unique<ListModeLUTOwned>(*scanner);
	const size_t numEvents = 500;
	data->allocate(numEvents);
	for (bin_t binId = 0; binId < numEvents; binId++)
	{
		const det_id_t d1 = rand() % numDets;
		const det_id_t d2 = rand() % numDets;
		data->setDetectorIdsOfEvent(binId, d1, d2);
	}

	// Helpter aliases
	using ImageSharedPTR = std::shared_ptr<Image>;
	const auto toOwned = [](const ImageSharedPTR& i)
	{ return reinterpret_cast<ImageOwned*>(i.get()); };

	ImageSharedPTR img_cpu = std::make_shared<ImageOwned>(img_params);
	toOwned(img_cpu)->allocate();
	img_cpu->setValue(0.0);
	dd(*scanner, data.get(), img_cpu, false);

	ImageSharedPTR img_gpu = std::make_shared<ImageOwned>(img_params);
	toOwned(img_gpu)->allocate();
	img_gpu->setValue(0.0);
	dd(*scanner, data.get(), img_gpu, true);

	const double rmseCpuGpu = get_rmse(img_gpu.get(), img_cpu.get());

	CHECK(rmseCpuGpu < 0.01);
#endif
}
