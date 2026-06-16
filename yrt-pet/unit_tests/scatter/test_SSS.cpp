/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#if BUILD_CUDA

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/scatter/ScatterSpace.hpp"
#include "yrt-pet/scatter/SingleScatterSimulator.hpp"
#include "yrt-pet/utils/ImageUtils.hpp"

#include <cmath>
#include <memory>

TEST_CASE("sss-gpu-vs-cpu", "[sss-gpu]")
{
	const auto scanner =
	    yrt::util::test::makeFakeScanner(200.0f,  // axialFOV (mm)
	                                     4.0f,    // crystalSize_z (mm)
	                                     4.0f,    // crystalSize_trans (mm)
	                                     20.0f,   // crystalDepth (mm)
	                                     200.0f,  // scannerRadius (mm)
	                                     48,      // detsPerRing
	                                     12,      // numRings
	                                     2,       // numDOI
	                                     8,       // maxRingDiff
	                                     6,       // minAngDiff
	                                     4        // detsPerBlock
	    );

	// Image covering the scanner FOV
	constexpr ssize_t nx = 64;
	constexpr ssize_t ny = 64;
	constexpr ssize_t nz = 32;
	constexpr float lengthX = 400.0f;
	constexpr float lengthY = 400.0f;
	constexpr float lengthZ = 200.0f;
	const yrt::ImageParams imgParams(nx, ny, nz, lengthX, lengthY, lengthZ);

	// Mu image: water sphere (0.096 1/cm)
	auto mu = std::make_unique<yrt::ImageOwned>(imgParams);
	mu->allocate();
	mu->fill(0.0f);
	yrt::util::fillSphere(*mu, 0.096f, 0.0f, 0.0f, 0.0f, 80.0f);

	// Lambda image: two hot spheres (Set activity to 1.0)
	auto lambda = std::make_unique<yrt::ImageOwned>(imgParams);
	lambda->allocate();
	lambda->fill(0.0f);
	yrt::util::fillSphere(*lambda, 1.0f, -30.0f, 0.0f, 0.0f, 25.0f);
	yrt::util::fillSphere(*lambda, 1.0f, 30.0f, 0.0f, 0.0f, 25.0f);

	// SSS (same seed)
	yrt::scatter::SingleScatterSimulator sss(
	    *scanner, *mu, *lambda, yrt::scatter::CrystalMaterial::LYSO, 13);

	// Initialize scatter spaces
	constexpr size_t nTOF = 1;  // TODO: Change this once SSS supports TOF
	constexpr size_t nPlanes = 6;
	constexpr size_t nAngles = 20;

	yrt::ScatterSpace cpuOut(*scanner, nTOF, nPlanes, nAngles);
	cpuOut.allocate();
	cpuOut.fill(0.0f);

	yrt::ScatterSpace gpuOut(*scanner, nTOF, nPlanes, nAngles);
	gpuOut.allocate();
	gpuOut.fill(0.0f);

	// CPU run
	sss.runSSS(cpuOut);

	// GPU run
	sss.runSSSDevice(gpuOut);

	// Compare
	const size_t numBins = cpuOut.count();
	REQUIRE(numBins == static_cast<size_t>(gpuOut.count()));
	const double numBins_double = static_cast<double>(numBins);

	double sqSum = 0.0, sumCPU = 0.0, sumGPU = 0.0;
	size_t nNonZero = 0;

	for (yrt::bin_t i = 0; i < numBins; ++i)
	{
		const float cpuVal = static_cast<double>(cpuOut.getProjectionValue(i));
		const float gpuVal = static_cast<double>(gpuOut.getProjectionValue(i));
		const float diff = static_cast<double>(cpuVal - gpuVal);
		sqSum += diff * diff;
		sumCPU += cpuVal;
		sumGPU += gpuVal;
		if (cpuVal > 0.0f || gpuVal > 0.0f)
		{
			++nNonZero;
		}
	}

	const double meanCPU = sumCPU / numBins_double;
	const double meanGPU = sumGPU / numBins_double;
	REQUIRE(sumCPU > 0.0);
	REQUIRE(sumGPU > 0.0);

	INFO("mean CPU: " << meanCPU << ", mean GPU: " << meanGPU
	                  << ", non-zero bins: " << nNonZero);

	const double nrmse = std::sqrt(sqSum / numBins_double) / meanCPU;
	INFO("NRMSE: " << nrmse);

	CHECK(nrmse < 1e-4);
}

#endif
