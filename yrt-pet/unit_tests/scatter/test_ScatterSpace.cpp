/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/scatter/ScatterSpace.hpp"

#include <algorithm>
#include <cmath>
#include <random>

TEST_CASE("scatterspace", "[scatterspace]")
{
	SECTION("scatterspace-construction", "[scatterspace]")
	{
		auto scanner = yrt::util::test::makeScanner();

		// valid-parameters
		{
			// Require that this does not throw a
			REQUIRE_NOTHROW(yrt::ScatterSpace(*scanner, 10, 20, 30));

			yrt::ScatterSpace space(*scanner, 10, 20, 30);
			REQUIRE(space.getNumTOFBins() == 10);
			REQUIRE(space.getNumPlanes() == 20);
			REQUIRE(space.getNumAngles() == 30);
		}

		// invalid-parameters
		{
			// Zero TOF bins - check that it throws
			REQUIRE_THROWS(yrt::ScatterSpace(*scanner, 0, 20, 30));

			// Zero planes - check that it throws
			REQUIRE_THROWS(yrt::ScatterSpace(*scanner, 10, 0, 30));

			// Need more than one angle - check that it throws
			REQUIRE_THROWS(yrt::ScatterSpace(*scanner, 10, 20, 1));
		}
	}

	SECTION("scatterspace-properties", "[scatterspace]")
	{
		auto scanner = yrt::util::test::makeScanner();

		yrt::ScatterSpace space(*scanner, 10, 20, 30);

		// radius-diameter
		{
			REQUIRE(space.getRadius() == scanner->scannerRadius);
			REQUIRE(space.getDiameter() == scanner->scannerRadius * 2.0f);
		}

		// axialfov
		{
			REQUIRE(space.getAxialFOV() == scanner->axialFOV);
		}

		// maxtof
		{
			// "scanner diameter"/"speed of light"
			const float expectedMaxTOF_ps =
			    scanner->scannerRadius * 2.0f / yrt::SPEED_OF_LIGHT_MM_PS;
			REQUIRE(space.getMaxTOF_ps() == Approx(expectedMaxTOF_ps));
		}
	}

	SECTION("scatterspace-step-size", "[scatterspace]")
	{
		auto scanner = yrt::util::test::makeScanner();

		// 10-tof-bins
		{
			yrt::ScatterSpace space(*scanner, 10, 20, 30);

			REQUIRE(space.getTOFBinStep_ps() ==
			        Approx(space.getMaxTOF_ps() / 10.0f));
			REQUIRE(space.getAngleStep() == Approx(yrt::TWOPI_FLT / 30.0f));
			REQUIRE(space.getPlaneStep() == Approx(scanner->axialFOV / 20.0f));
		}

		// difference-grid
		{
			yrt::ScatterSpace space(*scanner, 5, 10, 15);

			REQUIRE(space.getTOFBinStep_ps() ==
			        Approx(space.getMaxTOF_ps() / 5.0f));
			REQUIRE(space.getAngleStep() == Approx(yrt::TWOPI_FLT / 15.0f));
			REQUIRE(space.getPlaneStep() == Approx(scanner->axialFOV / 10.0f));
		}
	}

	SECTION("scatterspace-index-bidirectionality", "[scatterspace]")
	{
		auto scanner = yrt::util::test::makeScanner();

		// tof-bin-tof-value
		{
			yrt::ScatterSpace space(*scanner, 10, 20, 30);
			float tof_step = space.getTOFBinStep_ps();

			for (size_t bin = 0; bin < 10; ++bin)
			{
				float expected_tof =
				    tof_step * (static_cast<float>(bin) + 0.5f);
				REQUIRE(space.getTOF_ps(bin) == Approx(expected_tof));

				// Test that getTOF_ps and nearest neighbor are consistent
				yrt::ScatterSpace::ScatterSpacePosition pos;
				pos.tof_ps = expected_tof;
				pos.planePosition1 = 0.0f;
				pos.angle1 = 0.0f;
				pos.planePosition2 = 0.0f;
				pos.angle2 = 0.0f;

				auto idx = space.getNearestNeighborIndex(pos);
				REQUIRE(idx.tofBin == bin);
			}
		}

		// plane-index-plane-pos
		{
			yrt::ScatterSpace space(*scanner, 10, 20, 30);
			float plane_step = space.getPlaneStep();
			float axial_fov = space.getAxialFOV();
			float plane_start = -axial_fov / 2.0f;

			for (size_t plane = 0; plane < 20; ++plane)
			{
				float expected_pos =
				    plane_start +
				    plane_step * (static_cast<float>(plane) + 0.5f);
				REQUIRE(space.getPlanePosition(plane) == Approx(expected_pos));

				// Test consistency with nearest neighbor
				yrt::ScatterSpace::ScatterSpacePosition pos;
				pos.tof_ps = space.getTOF_ps(5);  // Middle TOF bin
				pos.planePosition1 = expected_pos;
				pos.angle1 = 0.0f;
				pos.planePosition2 = expected_pos;
				pos.angle2 = 0.0f;

				auto idx = space.getNearestNeighborIndex(pos);
				REQUIRE(idx.planeIndex1 == plane);
				REQUIRE(idx.planeIndex2 == plane);
			}
		}

		// angle-index-angle-value
		{
			yrt::ScatterSpace space(*scanner, 10, 20, 30);
			float angle_step = space.getAngleStep();

			for (size_t angle = 0; angle < 30; ++angle)
			{
				float expected_angle =
				    angle_step * (static_cast<float>(angle) + 0.5f);
				REQUIRE(space.getAngle(angle) == Approx(expected_angle));

				// Test consistency with nearest neighbor
				yrt::ScatterSpace::ScatterSpacePosition pos;
				pos.tof_ps = space.getTOF_ps(5);
				pos.planePosition1 = 0.0f;
				pos.angle1 = expected_angle;
				pos.planePosition2 = 0.0f;
				pos.angle2 = expected_angle;

				auto idx = space.getNearestNeighborIndex(pos);
				REQUIRE(idx.angleIndex1 == angle);
				REQUIRE(idx.angleIndex2 == angle);
			}
		}
	}

	SECTION("scatterspace-wrapangle", "[scatterspace]")
	{

		// between-0-and-2pi
		{
			REQUIRE(yrt::ScatterSpace::wrapAngle(0.0f) == Approx(0.0f));
			REQUIRE(yrt::ScatterSpace::wrapAngle(M_PI) == Approx(M_PI));
			REQUIRE(yrt::ScatterSpace::wrapAngle(2.0f * M_PI - 0.001f) ==
			        Approx(2.0f * M_PI - 0.001f));
		}

		// more-than-2pi
		{
			REQUIRE(yrt::ScatterSpace::wrapAngle(2.0f * M_PI + 0.5f) ==
			        Approx(0.5f));
			REQUIRE(yrt::ScatterSpace::wrapAngle(4.0f * M_PI + 1.0f) ==
			        Approx(1.0f));
		}

		// negative-angles
		{
			REQUIRE(yrt::ScatterSpace::wrapAngle(-0.5f) ==
			        Approx(2.0f * M_PI - 0.5f));
			REQUIRE(yrt::ScatterSpace::wrapAngle(-M_PI) == Approx(M_PI));
			REQUIRE(yrt::ScatterSpace::wrapAngle(-2.0f * M_PI - 0.3f) ==
			        Approx(2.0f * M_PI - 0.3f));
		}
	}

	SECTION("scatterspace-clamping", "[scatterspace]")
	{
		auto scanner = yrt::util::test::makeScanner();
		yrt::ScatterSpace space(*scanner, 10, 20, 30);

		// clamp-axialfov
		{
			float half_fov = space.getAxialFOV() / 2.0f;
			float plane_step = space.getPlaneStep();

			REQUIRE(space.clampPlanePosition(0.0f) == Approx(0.0f));
			REQUIRE(space.clampPlanePosition(half_fov - plane_step / 2) ==
			        Approx(half_fov - plane_step / 2));
			REQUIRE(space.clampPlanePosition(-half_fov + plane_step / 2) ==
			        Approx(-half_fov + plane_step / 2));

			// Test clamping at boundaries
			REQUIRE(space.clampPlanePosition(half_fov) ==
			        Approx(half_fov - plane_step / 2));
			REQUIRE(space.clampPlanePosition(-half_fov) ==
			        Approx(-half_fov + plane_step / 2));

			// Test beyond boundaries
			REQUIRE(space.clampPlanePosition(half_fov * 2.0f) ==
			        Approx(half_fov - plane_step / 2));
			REQUIRE(space.clampPlanePosition(-half_fov * 2.0f) ==
			        Approx(-half_fov + plane_step / 2));
		}

		// clamp-tof
		{
			float max_tof = space.getMaxTOF_ps();
			float tof_step = space.getTOFBinStep_ps();
			float half_step = tof_step / 2.0f;

			REQUIRE(space.clampTOF(half_step) == Approx(half_step));
			REQUIRE(space.clampTOF(max_tof - half_step) ==
			        Approx(max_tof - half_step));

			// Test clamping at boundaries
			REQUIRE(space.clampTOF(0.0f) == Approx(half_step));
			REQUIRE(space.clampTOF(max_tof) == Approx(max_tof - half_step));

			// Test beyond boundaries
			REQUIRE(space.clampTOF(-100.0f) == Approx(half_step));
			REQUIRE(space.clampTOF(max_tof * 2.0f) ==
			        Approx(max_tof - half_step));
		}
	}

	SECTION("scatterspace-nearestneighbor", "[scatterspace]")
	{
		auto scanner = yrt::util::test::makeScanner();

		yrt::ScatterSpace space(*scanner, 10, 20, 30);

		// Fill with pattern for testing
		for (size_t t = 0; t < 10; ++t)
		{
			for (size_t p1 = 0; p1 < 20; ++p1)
			{
				for (size_t a1 = 0; a1 < 30; ++a1)
				{
					for (size_t p2 = 0; p2 < 20; ++p2)
					{
						for (size_t a2 = 0; a2 < 30; ++a2)
						{
							float value =
							    static_cast<float>(t + p1 + a1 + p2 + a2);
							space.setValue(t, p1, a1, p2, a2, value);
						}
					}
				}
			}
		}

		// nearestneighbor-exact-centers
		{
			for (size_t t = 0; t < 10; ++t)
			{
				for (size_t p = 0; p < 20; ++p)
				{
					for (size_t a = 0; a < 30; ++a)
					{
						yrt::ScatterSpace::ScatterSpacePosition pos;
						pos.tof_ps = space.getTOF_ps(t);
						pos.planePosition1 = space.getPlanePosition(p);
						pos.angle1 = space.getAngle(a);
						pos.planePosition2 = space.getPlanePosition(p);
						pos.angle2 = space.getAngle(a);

						auto idx = space.getNearestNeighborIndex(pos);

						REQUIRE(idx.tofBin == t);
						REQUIRE(idx.planeIndex1 == p);
						REQUIRE(idx.angleIndex1 == a);
						REQUIRE(idx.planeIndex2 == p);
						REQUIRE(idx.angleIndex2 == a);
					}
				}
			}
		}

		// nearestneighbor-boundaries
		{
			float tof_step = space.getTOFBinStep_ps();
			float plane_step = space.getPlaneStep();
			float angle_step = space.getAngleStep();

			// Test near lower TOF boundary
			yrt::ScatterSpace::ScatterSpacePosition pos1;
			pos1.tof_ps = tof_step * 0.49f;  // Just below first bin center
			pos1.planePosition1 = 0.0f;
			pos1.angle1 = 0.0f;
			pos1.planePosition2 = 0.0f;
			pos1.angle2 = 0.0f;

			auto idx1 = space.getNearestNeighborIndex(pos1);
			REQUIRE(idx1.tofBin == 0);

			// Test near upper TOF boundary
			yrt::ScatterSpace::ScatterSpacePosition pos2;
			pos2.tof_ps = space.getMaxTOF_ps() -
			              tof_step * 0.49f;  // Just below last bin center
			pos2.planePosition1 = 0.0f;
			pos1.angle1 = 0.0f;
			pos1.planePosition2 = 0.0f;
			pos1.angle2 = 0.0f;

			auto idx2 = space.getNearestNeighborIndex(pos2);
			REQUIRE(idx2.tofBin == 9);
		}

		// nearestneighbor-wrapped-angles
		{
			float angle_step = space.getAngleStep();

			// Test angle slightly below 0 (should wrap to near 2π)
			yrt::ScatterSpace::ScatterSpacePosition pos;
			pos.tof_ps = space.getTOF_ps(5);
			pos.planePosition1 = 0.0f;
			pos.angle1 =
			    -angle_step * 0.3f;  // Should wrap to ~2π - 0.3*angle_step
			pos.planePosition2 = 0.0f;
			pos.angle2 = 0.0f;

			auto idx = space.getNearestNeighborIndex(pos);
			REQUIRE(idx.angleIndex1 == 29);  // Last angle bin

			// Test angle slightly above 2π
			pos.angle1 = 2.0f * M_PI + angle_step * 0.3f;
			idx = space.getNearestNeighborIndex(pos);
			REQUIRE(idx.angleIndex1 == 0);  // First angle bin
		}
	}

	SECTION("scatterspace-linearinterpolation", "[scatterspace]")
	{
		auto scanner = yrt::util::test::makeScanner();
		yrt::ScatterSpace space(*scanner, 2, 2, 2);  // Small grid for testing

		// linearinterpolation-exact-centers
		{
			// Interpolation at exact grid points returns same value
			// TODO NOW: This
		}

		// linearinterpolation-middle-of-grid
		{
			// Interpolation in middle of grid is weighted average

			// TODO NOW: This
			// Create a simple test pattern
			// For a 2x2x2x2x2 grid, test interpolation at the center
			// This would be the average of all 32 points if all values are
			// different Skip for now without setValue
		}
	}

	SECTION("scatterspace-edgecases", "[scatterspace]")
	{
		auto scanner = yrt::util::test::makeScanner();

		// minimum-num-bins
		{
			yrt::ScatterSpace space(*scanner, 1, 1,
			                        2);  // Need at least 2 angles

			// Should still work without crashing
			REQUIRE(space.getNumTOFBins() == 1);
			REQUIRE(space.getNumPlanes() == 1);
			REQUIRE(space.getNumAngles() == 2);

			// Test nearest neighbor at various positions
			yrt::ScatterSpace::ScatterSpacePosition pos;
			pos.tof_ps = space.getTOF_ps(0);
			pos.planePosition1 = space.getPlanePosition(0);
			pos.angle1 = space.getAngle(0);
			pos.planePosition2 = space.getPlanePosition(0);
			pos.angle2 = space.getAngle(1);

			auto idx = space.getNearestNeighborIndex(pos);
			REQUIRE(idx.tofBin == 0);
			REQUIRE(idx.planeIndex1 == 0);
			REQUIRE(idx.planeIndex2 == 0);
		}

		// small-angle-step
		{
			yrt::ScatterSpace space(*scanner, 10, 10, 360);  // 1 degree steps

			REQUIRE(space.getAngleStep() == Approx(2.0f * M_PI / 360.0f));

			// Test angle wrapping still works
			yrt::ScatterSpace::ScatterSpacePosition pos;
			pos.tof_ps = space.getTOF_ps(5);
			pos.planePosition1 = 0.0f;
			pos.angle1 = 2.0f * M_PI + 0.01f;  // Slightly above 2π
			pos.planePosition2 = 0.0f;
			pos.angle2 = 0.0f;

			auto idx = space.getNearestNeighborIndex(pos);
			REQUIRE(idx.angleIndex1 < 360);  // Should be valid
		}
	}

	SECTION("scatterspace-random-grid-sizes", "[scatterspace]")
	{
		std::random_device rd;
		std::default_random_engine gen(rd());
		std::uniform_int_distribution<> size_dist(2, 50);

		auto scanner = yrt::util::test::makeScanner();

		constexpr int NumTrials = 10;
		for (int trial = 0; trial < NumTrials; trial++)
		{
			size_t n_tof = size_dist(gen);
			size_t n_planes = size_dist(gen);
			size_t n_angles = size_dist(gen);

			SECTION("random-grid-size-" + std::to_string(trial))
			{
				REQUIRE_NOTHROW(
				    yrt::ScatterSpace(*scanner, n_tof, n_planes, n_angles));

				yrt::ScatterSpace space(*scanner, n_tof, n_planes, n_angles);

				// Verify basic properties
				REQUIRE(space.getNumTOFBins() == n_tof);
				REQUIRE(space.getNumPlanes() == n_planes);
				REQUIRE(space.getNumAngles() == n_angles);

				// Verify step sizes are consistent
				float tof_step = space.getTOFBinStep_ps();
				float angle_step = space.getAngleStep();
				float plane_step = space.getPlaneStep();

				REQUIRE(tof_step > 0.0f);
				REQUIRE(angle_step > 0.0f);
				REQUIRE(plane_step > 0.0f);

				// Test a few random positions
				std::uniform_real_distribution<float> tof_dist(
				    0.0f, space.getMaxTOF_ps());
				std::uniform_real_distribution<float> plane_dist(
				    -200.0f, 200.0f);  // Within ±axialFOV/2
				std::uniform_real_distribution<float> angle_dist(
				    0.0f, 4.0f * M_PI);  // Test wrapping

				for (int j = 0; j < 5; ++j)
				{
					yrt::ScatterSpace::ScatterSpacePosition pos;
					pos.tof_ps = tof_dist(gen);
					pos.planePosition1 = plane_dist(gen);
					pos.planePosition2 = plane_dist(gen);
					pos.angle1 = angle_dist(gen);
					pos.angle2 = angle_dist(gen);

					// Nearest neighbor should not crash
					REQUIRE_NOTHROW(space.getNearestNeighborIndex(pos));

					auto idx = space.getNearestNeighborIndex(pos);

					// Verify indices are within bounds
					REQUIRE(idx.tofBin < n_tof);
					REQUIRE(idx.planeIndex1 < n_planes);
					REQUIRE(idx.planeIndex2 < n_planes);
					REQUIRE(idx.angleIndex1 < n_angles);
					REQUIRE(idx.angleIndex2 < n_angles);
				}
			}
		}
	}

	// Add this if you want to test the actual interpolation with values
	// We need a way to set values in the ScatterSpace
	SECTION("scatterspace-getters-and-setters", "[scatterspace]")
	{
		// This test assumes we have a way to set values
		// You'll need to add a setValue method to your ScatterSpace class

		auto scanner = yrt::util::test::makeScanner();
		yrt::ScatterSpace space(*scanner, 2, 2, 2);

		// Set a known pattern
		for (size_t t = 0; t < 2; ++t)
		{
			for (size_t p1 = 0; p1 < 2; ++p1)
			{
				for (size_t a1 = 0; a1 < 2; ++a1)
				{
					for (size_t p2 = 0; p2 < 2; ++p2)
					{
						for (size_t a2 = 0; a2 < 2; ++a2)
						{
							const float value = static_cast<float>(
							    t * 1000 + p1 * 100 + a1 * 10 + p2 + a2);
							space.setValue({t, p1, a1, p2, a2}, value);
						}
					}
				}
			}
		}

		// Verify we can retrieve values
		for (size_t t = 0; t < 2; ++t)
		{
			for (size_t p1 = 0; p1 < 2; ++p1)
			{
				for (size_t a1 = 0; a1 < 2; ++a1)
				{
					for (size_t p2 = 0; p2 < 2; ++p2)
					{
						for (size_t a2 = 0; a2 < 2; ++a2)
						{
							const float expected = static_cast<float>(
							    t * 1000 + p1 * 100 + a1 * 10 + p2 + a2);
							REQUIRE(space.getValue(t, p1, a1, p2, a2) ==
							        Approx(expected));
						}
					}
				}
			}
		}
	}

	// TODO NOW: Add unit test for usage of getProjectionValueFromHistogramBin
}
