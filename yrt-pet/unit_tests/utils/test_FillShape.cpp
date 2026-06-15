/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#include <random>

namespace
{

using namespace yrt;
using namespace yrt::util;

void checkFilled(
    const Image& image, const ImageParams& params,
    const std::function<bool(ssize_t, ssize_t, ssize_t)>& predicate,
    frame_t dynamicFrame, float expectedInside, float expectedOutside)
{
	for (ssize_t iz = 0; iz < params.nz; iz++)
	{
		for (ssize_t iy = 0; iy < params.ny; iy++)
		{
			for (ssize_t ix = 0; ix < params.nx; ix++)
			{
				const float val = image.getData().get(
				    {static_cast<size_t>(dynamicFrame), static_cast<size_t>(iz),
				     static_cast<size_t>(iy), static_cast<size_t>(ix)});
				if (predicate(ix, iy, iz))
				{
					REQUIRE(val == expectedInside);
				}
				else
				{
					REQUIRE(val == expectedOutside);
				}
			}
		}
	}
}

bool insideCircle(const ImageParams& params, ssize_t ix, ssize_t iy,
                  float centerX, float centerY, float radius)
{
	const Vector3D pos = params.indexToPosition(ix, iy, 0);
	const float dx = pos.x - centerX;
	const float dy = pos.y - centerY;
	return dx * dx + dy * dy <= radius * radius;
}

bool insideEllipse(const ImageParams& params, ssize_t ix, ssize_t iy,
                   float centerX, float centerY, float semiAxisX,
                   float semiAxisY, float angle)
{
	const Vector3D pos = params.indexToPosition(ix, iy, 0);
	const float dx = pos.x - centerX;
	const float dy = pos.y - centerY;
	const float cosA = std::cos(angle);
	const float sinA = std::sin(angle);
	const float dxRot = dx * cosA + dy * sinA;
	const float dyRot = -dx * sinA + dy * cosA;
	return (dxRot * dxRot) / (semiAxisX * semiAxisX) +
	           (dyRot * dyRot) / (semiAxisY * semiAxisY) <=
	       1.0f;
}

bool insideSphere(const ImageParams& params, ssize_t ix, ssize_t iy, ssize_t iz,
                  float centerX, float centerY, float centerZ, float radius)
{
	const Vector3D pos = params.indexToPosition(ix, iy, iz);
	const float dx = pos.x - centerX;
	const float dy = pos.y - centerY;
	const float dz = pos.z - centerZ;
	return dx * dx + dy * dy + dz * dz <= radius * radius;
}

bool insideEllipsoid(const ImageParams& params, ssize_t ix, ssize_t iy,
                     ssize_t iz, float centerX, float centerY, float centerZ,
                     float semiAxisX, float semiAxisY, float semiAxisZ)
{
	const Vector3D pos = params.indexToPosition(ix, iy, iz);
	const float dx = pos.x - centerX;
	const float dy = pos.y - centerY;
	const float dz = pos.z - centerZ;
	return (dx * dx) / (semiAxisX * semiAxisX) +
	           (dy * dy) / (semiAxisY * semiAxisY) +
	           (dz * dz) / (semiAxisZ * semiAxisZ) <=
	       1.0f;
}

}  // namespace

TEST_CASE("fillCircle-single", "[fill][circle]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<ssize_t> sizeDist(10, 30);
	std::uniform_real_distribution<float> lengthDist(20.0f, 100.0f);
	std::uniform_real_distribution<float> offsetDist(-20.0f, 20.0f);

	const ssize_t nx = sizeDist(engine);
	const ssize_t ny = sizeDist(engine);
	const float lengthX = lengthDist(engine);
	const float lengthY = lengthDist(engine);
	const float offsetX = offsetDist(engine);
	const float offsetY = offsetDist(engine);

	ImageParams params(nx, ny, 1, lengthX, lengthY, 10.0f, offsetX, offsetY,
	                   0.0f);

	const float centerX = std::uniform_real_distribution<float>(
	    -lengthX / 4, lengthX / 4)(engine);
	const float centerY = std::uniform_real_distribution<float>(
	    -lengthY / 4, lengthY / 4)(engine);
	const float radius = std::uniform_real_distribution<float>(
	    3.0f, std::min(lengthX, lengthY) / 6.0f)(engine);

	ImageOwned image(params);
	image.allocate();
	SECTION("fill frame 0 z=0")
	{
		image.fill(0.0f);
		fillCircle(image, 1.0f, centerX, centerY, radius);

		auto pred = [&](ssize_t ix, ssize_t iy, ssize_t)
		{ return insideCircle(params, ix, iy, centerX, centerY, radius); };
		checkFilled(image, params, pred, 0, 1.0f, 0.0f);
	}
	SECTION("fill specific z-slice")
	{
		image.fill(0.0f);
		const ssize_t zSlice = 0;
		fillCircle(image, 1.0f, centerX, centerY, radius, zSlice);

		auto pred = [&](ssize_t ix, ssize_t iy, ssize_t iz)
		{
			return iz == zSlice &&
			       insideCircle(params, ix, iy, centerX, centerY, radius);
		};
		checkFilled(image, params, pred, 0, 1.0f, 0.0f);
	}
	SECTION("fill specific dynamic frame")
	{
		ImageParams paramsMultiFrame(nx, ny, 1, lengthX, lengthY, 10.0f,
		                             offsetX, offsetY, 0.0f, 3);
		ImageOwned imgMulti(paramsMultiFrame);
		imgMulti.allocate();
		imgMulti.fill(0.0f);

		const frame_t targetFrame = 1;
		fillCircle(imgMulti, 1.0f, centerX, centerY, radius, 0, targetFrame);

		auto pred = [&](ssize_t ix, ssize_t iy, ssize_t)
		{ return insideCircle(params, ix, iy, centerX, centerY, radius); };
		checkFilled(imgMulti, paramsMultiFrame, pred, targetFrame, 1.0f, 0.0f);

		// Other frames should be all zeros
		for (frame_t ft = 0; ft < paramsMultiFrame.nt; ft++)
		{
			if (ft != targetFrame)
			{
				checkFilled(
				    imgMulti, paramsMultiFrame, [](ssize_t, ssize_t, ssize_t)
				    { return true; }, ft, 0.0f, 0.0f);
			}
		}
	}
}

TEST_CASE("fillEllipse-single", "[fill][ellipse]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<ssize_t> sizeDist(10, 30);
	std::uniform_real_distribution<float> lengthDist(20.0f, 100.0f);
	std::uniform_real_distribution<float> offsetDist(-20.0f, 20.0f);
	std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);

	const ssize_t nx = sizeDist(engine);
	const ssize_t ny = sizeDist(engine);
	const float lengthX = lengthDist(engine);
	const float lengthY = lengthDist(engine);
	const float offsetX = offsetDist(engine);
	const float offsetY = offsetDist(engine);

	ImageParams params(nx, ny, 1, lengthX, lengthY, 10.0f, offsetX, offsetY,
	                   0.0f);

	const float centerX = std::uniform_real_distribution<float>(
	    -lengthX / 4, lengthX / 4)(engine);
	const float centerY = std::uniform_real_distribution<float>(
	    -lengthY / 4, lengthY / 4)(engine);
	const float semiAxisX = std::uniform_real_distribution<float>(
	    3.0f, std::min(lengthX, lengthY) / 6.0f)(engine);
	const float semiAxisY = std::uniform_real_distribution<float>(
	    3.0f, std::min(lengthX, lengthY) / 6.0f)(engine);
	const float angle = angleDist(engine);

	ImageOwned image(params);
	image.allocate();
	image.fill(0.0f);

	fillEllipse(image, 1.0f, centerX, centerY, semiAxisX, semiAxisY, angle);

	auto pred = [&](ssize_t ix, ssize_t iy, ssize_t)
	{
		return insideEllipse(params, ix, iy, centerX, centerY, semiAxisX,
		                     semiAxisY, angle);
	};
	checkFilled(image, params, pred, 0, 1.0f, 0.0f);
}

TEST_CASE("fillSphere-single", "[fill][sphere]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<ssize_t> sizeDist(10, 20);
	std::uniform_real_distribution<float> lengthDist(20.0f, 80.0f);
	std::uniform_real_distribution<float> offsetDist(-10.0f, 10.0f);

	const ssize_t nx = sizeDist(engine);
	const ssize_t ny = sizeDist(engine);
	const ssize_t nz = sizeDist(engine);
	const float lengthX = lengthDist(engine);
	const float lengthY = lengthDist(engine);
	const float lengthZ = lengthDist(engine);
	const float offsetX = offsetDist(engine);
	const float offsetY = offsetDist(engine);
	const float offsetZ = offsetDist(engine);

	const ImageParams params(nx, ny, nz, lengthX, lengthY, lengthZ, offsetX,
	                         offsetY, offsetZ);

	const float centerX = std::uniform_real_distribution<float>(
	    -lengthX / 4, lengthX / 4)(engine);
	const float centerY = std::uniform_real_distribution<float>(
	    -lengthY / 4, lengthY / 4)(engine);
	const float centerZ = std::uniform_real_distribution<float>(
	    -lengthZ / 4, lengthZ / 4)(engine);
	const float radius = std::uniform_real_distribution<float>(
	    3.0f, std::min({lengthX, lengthY, lengthZ}) / 6.0f)(engine);

	ImageOwned image(params);
	image.allocate();
	image.fill(0.0f);

	fillSphere(image, 1.0f, centerX, centerY, centerZ, radius);

	auto pred = [&](ssize_t ix, ssize_t iy, ssize_t iz)
	{
		return insideSphere(params, ix, iy, iz, centerX, centerY, centerZ,
		                    radius);
	};
	checkFilled(image, params, pred, 0, 1.0f, 0.0f);
}

TEST_CASE("fillEllipsoid-single", "[fill][ellipsoid]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<ssize_t> sizeDist(10, 20);
	std::uniform_real_distribution<float> lengthDist(20.0f, 80.0f);
	std::uniform_real_distribution<float> offsetDist(-10.0f, 10.0f);

	const ssize_t nx = sizeDist(engine);
	const ssize_t ny = sizeDist(engine);
	const ssize_t nz = sizeDist(engine);
	const float lengthX = lengthDist(engine);
	const float lengthY = lengthDist(engine);
	const float lengthZ = lengthDist(engine);
	const float offsetX = offsetDist(engine);
	const float offsetY = offsetDist(engine);
	const float offsetZ = offsetDist(engine);

	ImageParams params(nx, ny, nz, lengthX, lengthY, lengthZ, offsetX, offsetY,
	                   offsetZ);

	const float centerX = std::uniform_real_distribution<float>(
	    -lengthX / 4, lengthX / 4)(engine);
	const float centerY = std::uniform_real_distribution<float>(
	    -lengthY / 4, lengthY / 4)(engine);
	const float centerZ = std::uniform_real_distribution<float>(
	    -lengthZ / 4, lengthZ / 4)(engine);
	const float semiAxisX = std::uniform_real_distribution<float>(
	    3.0f, std::min({lengthX, lengthY, lengthZ}) / 6.0f)(engine);
	const float semiAxisY = std::uniform_real_distribution<float>(
	    3.0f, std::min({lengthX, lengthY, lengthZ}) / 6.0f)(engine);
	const float semiAxisZ = std::uniform_real_distribution<float>(
	    3.0f, std::min({lengthX, lengthY, lengthZ}) / 6.0f)(engine);

	ImageOwned image(params);
	image.allocate();
	image.fill(0.0f);

	fillEllipsoid(image, 1.0f, centerX, centerY, centerZ, semiAxisX, semiAxisY,
	              semiAxisZ);

	auto pred = [&](ssize_t ix, ssize_t iy, ssize_t iz)
	{
		return insideEllipsoid(params, ix, iy, iz, centerX, centerY, centerZ,
		                       semiAxisX, semiAxisY, semiAxisZ);
	};
	checkFilled(image, params, pred, 0, 1.0f, 0.0f);
}

TEST_CASE("fill-overlapping", "[fill][overlap]")
{
	ImageParams params(20, 20, 1, 40.0f, 40.0f, 10.0f, 0.0f, 0.0f, 0.0f);

	auto getVal = [&](const Image& img, ssize_t ix, ssize_t iy, frame_t ft = 0)
	{
		return img.getData().get({static_cast<size_t>(ft), 0ull,
		                          static_cast<size_t>(iy),
		                          static_cast<size_t>(ix)});
	};

	SECTION("two circles overlapping")
	{
		ImageOwned image(params);
		image.allocate();
		image.fill(0.0f);

		constexpr float radius = 10.0f;
		fillCircle(image, 1.0f, -5.0f, 0.0f, radius);
		fillCircle(image, 1.0f, 5.0f, 0.0f, radius);

		// Check specific known voxels
		// Left circle only: near (-5, 0): voxel (7, 10)
		REQUIRE(getVal(image, 7, 10) == 1.0f);
		REQUIRE(insideCircle(params, 7, 10, -5.0f, 0.0f, radius));
		REQUIRE(!insideCircle(params, 7, 10, 5.0f, 0.0f, radius));

		// Right circle only: near (5, 0): voxel (12, 10)
		REQUIRE(getVal(image, 12, 10) == 1.0f);
		REQUIRE(!insideCircle(params, 12, 10, -5.0f, 0.0f, radius));
		REQUIRE(insideCircle(params, 12, 10, 5.0f, 0.0f, radius));

		// Both: near center (1, 1): voxel (10, 10)
		// Value accumulates: 1.0 (left) + 1.0 (right) = 2.0
		REQUIRE(getVal(image, 10, 10) == 2.0f);
		REQUIRE(insideCircle(params, 10, 10, -5.0f, 0.0f, radius));
		REQUIRE(insideCircle(params, 10, 10, 5.0f, 0.0f, radius));

		// Outside both: far corner voxel (0, 0)
		REQUIRE(getVal(image, 0, 0) == 0.0f);
	}

	SECTION("circle on negative background")
	{
		ImageOwned image(params);
		image.allocate();
		image.fill(-5.0f);

		constexpr float centerX = 0.0f;
		constexpr float centerY = 0.0f;
		constexpr float radius = 8.0f;
		fillCircle(image, 1.0f, centerX, centerY, radius);

		auto pred = [&](ssize_t ix, ssize_t iy, ssize_t)
		{ return insideCircle(params, ix, iy, centerX, centerY, radius); };
		checkFilled(image, params, pred, 0, -4.0f, -5.0f);
	}

	SECTION("ellipse on top of circle")
	{
		ImageOwned image(params);
		image.allocate();
		image.fill(0.0f);

		// Big circle centered at origin, then smaller ellipse offset to the
		//  right
		fillCircle(image, 1.0f, 0.0f, 0.0f, 15.0f);
		// Ellipse extends beyond the circle on the right side
		fillEllipse(image, 1.0f, 5.0f, 0.0f, 8.0f, 4.0f, 0.0f);

		// Voxel in both circle and ellipse: near (5, 0): voxel (12, 10)
		// Value accumulates: 1.0 (circle) + 1.0 (ellipse) = 2.0
		REQUIRE(getVal(image, 12, 10) == 2.0f);
		REQUIRE(insideCircle(params, 12, 10, 0.0f, 0.0f, 15.0f));
		REQUIRE(insideEllipse(params, 12, 10, 5.0f, 0.0f, 8.0f, 4.0f, 0.0f));

		// Circle-only voxel: near (-9, 1): voxel (5, 10)
		REQUIRE(getVal(image, 5, 10) == 1.0f);
		REQUIRE(insideCircle(params, 5, 10, 0.0f, 0.0f, 15.0f));
		REQUIRE_FALSE(insideEllipse(params, 5, 10, 5.0f, 0.0f, 8.0f, 4.0f, 0.0f));

		// Outside both: far corner voxel (0, 0)
		REQUIRE(getVal(image, 0, 0) == 0.0f);
	}
}

TEST_CASE("fill-getImage", "[fill][get]")
{
	std::default_random_engine engine(
	    static_cast<unsigned int>(std::time(nullptr)));

	std::uniform_int_distribution<ssize_t> sizeDist(10, 20);
	std::uniform_real_distribution<float> lengthDist(20.0f, 60.0f);
	std::uniform_real_distribution<float> offsetDist(-10.0f, 10.0f);

	const ssize_t nx = sizeDist(engine);
	const ssize_t ny = sizeDist(engine);
	const float lengthX = lengthDist(engine);
	const float lengthY = lengthDist(engine);
	const float offsetX = offsetDist(engine);
	const float offsetY = offsetDist(engine);

	const ImageParams params(nx, ny, 3, lengthX, lengthY, 15.0f, offsetX,
	                         offsetY, 0.0f, 2);

	const float centerX = std::uniform_real_distribution<float>(
	    -lengthX / 4, lengthX / 4)(engine);
	const float centerY = std::uniform_real_distribution<float>(
	    -lengthY / 4, lengthY / 4)(engine);
	const float radius = std::uniform_real_distribution<float>(
	    3.0f, std::min(lengthX, lengthY) / 6.0f)(engine);

	const auto image = getCircleImage(params, 3.0f, centerX, centerY, radius);

	// All frames and all z-slices should have the circle
	for (frame_t dynamicFrame = 0; dynamicFrame < params.nt; dynamicFrame++)
	{
		for (ssize_t iz = 0; iz < params.nz; iz++)
		{
			auto pred = [&](ssize_t ix, ssize_t iy, ssize_t)
			{ return insideCircle(params, ix, iy, centerX, centerY, radius); };
			checkFilled(*image, params, pred, dynamicFrame, 3.0f, 0.0f);
		}
	}
}
