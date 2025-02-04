/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../../include/datastruct/image/Image.hpp"
#include "../../include/operators/OperatorProjectorSiddon.hpp"

#include "../../../build-cuda-pybind11/external/Catch/single_include/catch2/catch.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <memory>

/** Helper function for adjoint test
 *
 * Compute the backprojection of the line of response into img_bp, and return
 * the dot product between img and img_bp.
 */
double bp_dot(const Line3D& lor, Image* img_bp, const Image* img,
              float proj_val)
{
	img_bp->setValue(0.0);
	OperatorProjectorSiddon::singleBackProjection(img_bp, lor, proj_val);
	return img->dotProduct(*img_bp);
}

double bp_dot_slow(const Line3D& lor, Image* img_bp, const Image* img,
                   float proj_val)
{
	img_bp->setValue(0.0);
	OperatorProjectorSiddon::project_helper<false, false, false>(img_bp, lor,
	                                                             proj_val);
	return img->dotProduct(*img_bp);
}

TEST_CASE("Siddon-simple", "[siddon]")
{
	int random_seed = time(0);
	srand(random_seed);
	std::string rseed_str = "random_seed=" + std::to_string(random_seed);

	// Setup image
	int nx = 5;
	int ny = 5;
	int nz = 6;
	float sx = 1.1;
	float sy = 1.1;
	float sz = 1.2;
	float ox = 0.0;
	float oy = 0.0;
	float oz = 0.0;
	ImageParams img_params(nx, ny, nz, sx, sy, sz, ox, oy, oz);
	auto img = std::make_unique<ImageOwned>(img_params);
	img->allocate();
	img->setValue(1.0);
	auto img_bp = std::make_unique<ImageOwned>(img_params);
	img_bp->allocate();
	img_bp->setValue(0.0);
	float fov_radius = img->getRadius();

	SECTION("planar_isocenter_ray")
	{
		int num_tests = 10;
		for (int i = 0; i < num_tests; i++)
		{
			double beta = 2 * M_PI * i / (double)(num_tests - 1);
			// Single line of response (through isocenter)
			Vector3D p1{-sx * cosf(beta), -sx * sinf(beta), oz};
			Vector3D p2{sx * cosf(beta), sx * sinf(beta), oz};
			Line3D lor{p1, p2};
			INFO(rseed_str + " i=" + std::to_string(i));
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			REQUIRE(proj_val == Approx(2 * fov_radius));

			// Adjoint
			double proj_val_t = rand() / (double)RAND_MAX * proj_val;
			double dot_Ax_y = proj_val * proj_val_t;
			double dot_x_Aty = bp_dot(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_Ax_y == Approx(dot_x_Aty));

			// Slow version of ray tracing
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
			double dot_x_Aty_slow =
			    bp_dot_slow(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_x_Aty == Approx(dot_x_Aty_slow));
		}
	}

	SECTION("within_fov_ray")
	{
		int num_tests = 10;
		for (int i = 0; i < num_tests; i++)
		{
			float beta_1 = rand() / (double)RAND_MAX * 2 * M_PI;
			float beta_2 = rand() / (double)RAND_MAX * 2 * M_PI;
			float rad_1 = rand() / (double)RAND_MAX * 0.8 * fov_radius;
			float rad_2 = rand() / (double)RAND_MAX * 0.8 * fov_radius;
			// Single line of response (through isocenter)
			Vector3D p1{rad_1 * cosf(beta_1), rad_1 * sinf(beta_1), oz};
			Vector3D p2{rad_2 * cosf(beta_2), rad_2 * sinf(beta_2), oz};
			Line3D lor{p1, p2};
			INFO(rseed_str + " i=" + std::to_string(i));
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			REQUIRE(proj_val == Approx((p1 - p2).getNorm()));

			// Adjoint
			double proj_val_t = rand() / (double)RAND_MAX * proj_val;
			double dot_Ax_y = proj_val * proj_val_t;
			double dot_x_Aty = bp_dot(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_Ax_y == Approx(dot_x_Aty));

			// Slow version of ray tracing
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
			double dot_x_Aty_slow =
			    bp_dot_slow(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_x_Aty == Approx(dot_x_Aty_slow));
		}
	}

	SECTION("planar_y_ray")
	{
		int num_tests = 10;
		for (int i = 0; i < num_tests; i++)
		{
			float y0 = sy * i / static_cast<float>(num_tests - 1) - sy / 2;
			// Single line of response (parallel to y-axis)
			Vector3D p1{-sx, y0, oz};
			Vector3D p2{sx, y0, p1.z};
			Line3D lor{p1, p2};
			float integral_ref =
			    2 * sqrtf(std::max(0.0f, fov_radius * fov_radius - y0 * y0));
			INFO(rseed_str + " i=" + std::to_string(i));
			float proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			REQUIRE(proj_val == Approx(integral_ref));
			// Adjoint
			double proj_val_t = rand() / (double)RAND_MAX * proj_val;
			double dot_Ax_y = proj_val * proj_val_t;
			double dot_x_Aty = bp_dot(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_Ax_y == Approx(dot_x_Aty));

			// Slow version of ray tracing
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
			double dot_x_Aty_slow =
			    bp_dot_slow(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_x_Aty == Approx(dot_x_Aty_slow));
		}
	}

	SECTION("outside_ray")
	{
		// Lines of response outside of the field of view
		{
			Vector3D p1{sx, oy, oz};
			Vector3D p2{2 * sx, p1.y, p1.z};
			Line3D lor{p1, p2};
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			REQUIRE(proj_val == Approx(0.f));
			// Adjoint
			double proj_val_t = rand() / (double)RAND_MAX * proj_val;
			double dot_Ax_y = proj_val * proj_val_t;
			double dot_x_Aty = bp_dot(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_Ax_y == Approx(dot_x_Aty));

			// Slow version of ray tracing
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
			double dot_x_Aty_slow =
			    bp_dot_slow(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_x_Aty == Approx(dot_x_Aty_slow));
		}
		{
			Vector3D p1{2 * sx, oy, oz};
			Vector3D p2{2 * sx, sy, p1.z};
			Line3D lor{p1, p2};
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			REQUIRE(proj_val == Approx(0.f));
			// Adjoint
			double proj_val_t = rand() / (double)RAND_MAX * proj_val;
			double dot_Ax_y = proj_val * proj_val_t;
			double dot_x_Aty = bp_dot(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_Ax_y == Approx(dot_x_Aty));

			// Slow version of ray tracing
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
			double dot_x_Aty_slow =
			    bp_dot_slow(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_x_Aty == Approx(dot_x_Aty_slow));
		}
		for (int i = 0; i < 2; i++)
		{
			float delta_z = (i == 0) ? 0 : rand() / (double)RAND_MAX * 0.00001;
			Vector3D p1{-sx, 0.0f, 1.0001f * sz / 2.0f};
			Vector3D p2{sx, 0.0f, p1.z + delta_z};
			Line3D lor{p1, p2};
			float proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			REQUIRE(proj_val == Approx(0.f));
			// Adjoint
			double proj_val_t = rand() / (double)RAND_MAX * proj_val;
			double dot_Ax_y = proj_val * proj_val_t;
			double dot_x_Aty = bp_dot(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_Ax_y == Approx(dot_x_Aty));

			// Slow version of ray tracing
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
			double dot_x_Aty_slow =
			    bp_dot_slow(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_x_Aty == Approx(dot_x_Aty_slow));
		}
	}

	SECTION("z_ray")
	{
		int num_tests = 10;
		for (int i = 0; i < num_tests; i++)
		{
			// Line of response along diameter of FOV (varying z)
			float z1 = rand() / (double)RAND_MAX * sz - sz / 2;
			float z2 = rand() / (double)RAND_MAX * sz - sz / 2;
			Vector3D p1{0.0f, -fov_radius, z1};
			Vector3D p2{0.0f, fov_radius, z2};
			Line3D lor{p1, p2};
			double integral_ref =
			    sqrtf(4.f * fov_radius * fov_radius + (z2 - z1) * (z2 - z1));
			INFO(rseed_str + " i=" + std::to_string(i));
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			REQUIRE(proj_val == Approx(integral_ref));
			// Adjoint
			double proj_val_t = rand() / (double)RAND_MAX * proj_val;
			double dot_Ax_y = proj_val * proj_val_t;
			double dot_x_Aty = bp_dot(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_Ax_y == Approx(dot_x_Aty));

			// Slow version of ray tracing
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
			double dot_x_Aty_slow =
			    bp_dot_slow(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_x_Aty == Approx(dot_x_Aty_slow));
		}
		for (int i = 0; i < 4; i++)
		{
			Vector3D p1;
			Vector3D p2;
			double l_ref;
			switch (i)
			{
			case 0:
				p1.x = 0.0;
				p1.y = 0.0;
				p1.z = sz;
				p2.x = 0.0;
				p2.y = 0.0;
				p2.z = -sz;
				l_ref = sz;
				break;
			case 1:
				p1.x = 0.0;
				p1.y = -sy;
				p1.z = 0.0;
				p2.x = 0.0;
				p2.y = sy;
				p2.z = 0.0;
				l_ref = 2 * fov_radius;
				break;
			case 2:
				p1.x = -sx;
				p1.y = 0.0;
				p1.z = 0.0;
				p2.x = sx;
				p2.y = 0.0;
				p2.z = 0.0;
				l_ref = 2 * fov_radius;
				break;
			case 3:
				p1.x = -sx;
				p1.y = -sy;
				p1.z = 0.0;
				p2.x = sx;
				p2.y = sy;
				p2.z = 0.0;
				l_ref = 2 * fov_radius;
				break;
			}
			Line3D lor{p1, p2};
			INFO("axis i=" + std::to_string(i));
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			REQUIRE(proj_val == Approx(l_ref));
			// Adjoint
			double proj_val_t = rand() / (double)RAND_MAX * proj_val;
			double dot_Ax_y = proj_val * proj_val_t;
			double dot_x_Aty = bp_dot(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_Ax_y == Approx(dot_x_Aty));

			// Slow version of ray tracing
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
			double dot_x_Aty_slow =
			    bp_dot_slow(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_x_Aty == Approx(dot_x_Aty_slow));
		}
	}
}

TEST_CASE("Siddon-random", "[siddon]")
{
	int random_seed = time(0);
	srand(random_seed);
	std::string rseed_str = "random_seed=" + std::to_string(random_seed);

	// Setup image
	size_t nx = 1 + (rand() % 30);
	size_t ny = 1 + (rand() % 30);
	size_t nz = 1 + (rand() % 20);
	double sx = 0.01 + (rand() / (double)RAND_MAX * 5.0);
	double sy = 0.01 + (rand() / (double)RAND_MAX * 10.0);
	double sz = 0.01 + (rand() / (double)RAND_MAX * 10.0);
	double ox = 0.0;
	double oy = 0.0;
	double oz = 0.0;
	ImageParams img_params(nx, ny, nz, sx, sy, sz, ox, oy, oz);
	auto img = std::make_unique<ImageOwned>(img_params);
	img->allocate();
	img->setValue(1.0);
	// Randomize image content
	Array3DAlias<float> img_arr = img->getArray();
	for (size_t k = 0; k < nz; k++)
	{
		for (size_t j = 0; j < ny; j++)
		{
			for (size_t i = 0; i < nx; i++)
			{
				img_arr[k][j][i] =
				    rand() / static_cast<float>(RAND_MAX) * 10.0f - 5.0f;
			}
		}
	}
	auto img_bp = std::make_unique<ImageOwned>(img_params);
	img_bp->allocate();
	img_bp->setValue(0.0);
	double fov_radius = img->getRadius();
	double dx = sx / nx;
	double dy = sy / ny;
	double dz = sz / nz;

	SECTION("sampling_check")
	{
		int num_tests = 100;
		for (int i = 0; i < num_tests; i++)
		{
			// Line of response
			float x1 = rand() / static_cast<float>(RAND_MAX) * 2.0 * sx - sx;
			float x2 = rand() / static_cast<float>(RAND_MAX) * 2.0 * sx - sx;
			float y1 = rand() / static_cast<float>(RAND_MAX) * 2.0 * sy - sy;
			float y2 = rand() / static_cast<float>(RAND_MAX) * 2.0 * sy - sy;
			float z1 = rand() / static_cast<float>(RAND_MAX) * 2.0 * sz - sz;
			float z2 = rand() / static_cast<float>(RAND_MAX) * 2.0 * sz - sz;

			Vector3D p1{x1, y1, z1};
			Vector3D p2{x2, y2, z2};
			Line3D lor{p1, p2};

			// Use Siddon implementation to compute projection
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			// Compute reference
			double proj_ref = 0.0;
			double t1;
			double t2;
			{
				// Intersection with (centered) FOV cylinder
				double A = (p2.x - p1.x) * (p2.x - p1.x) +
				           (p2.y - p1.y) * (p2.y - p1.y);
				double B = 2.0 * ((p2.x - p1.x) * p1.x + (p2.y - p1.y) * p1.y);
				double C = p1.x * p1.x + p1.y * p1.y - fov_radius * fov_radius;
				if (A != 0.0)
				{
					double Delta = B * B - 4 * A * C;
					if (Delta <= 0.0)
					{
						t1 = 1.0;
						t2 = 0.0;
					}
					else
					{
						t1 = (-B - sqrt(Delta)) / (2 * A);
						t2 = (-B + sqrt(Delta)) / (2 * A);
					}
				}
				else
				{
					t1 = 0.0;
					t2 = 1.0;
				}
			}
			// Clip to ray range
			t1 = std::max(0.0, t1);
			t2 = std::min(1.0, t2);
			if ((p2 - p1).getNorm() > 0.0 && t1 < t2)
			{
				for (size_t k = 0; k < nz; k++)
				{
					for (size_t j = 0; j < ny; j++)
					{
						for (size_t i = 0; i < nx; i++)
						{
							double x0 = -sx / 2 + i * dx;
							double x1 = -sx / 2 + (i + 1) * dx;
							double y0 = -sy / 2 + j * dy;
							double y1 = -sy / 2 + (j + 1) * dy;
							double z0 = -sz / 2 + k * dz;
							double z1 = -sz / 2 + (k + 1) * dz;
							double ax0 = (x0 - p1.x) / (p2.x - p1.x);
							double ax1 = (x1 - p1.x) / (p2.x - p1.x);
							double ay0 = (y0 - p1.y) / (p2.y - p1.y);
							double ay1 = (y1 - p1.y) / (p2.y - p1.y);
							double az0 = (z0 - p1.z) / (p2.z - p1.z);
							double az1 = (z1 - p1.z) / (p2.z - p1.z);
							double amin = std::max({t1, std::min(ax0, ax1),
							                        std::min(ay0, ay1),
							                        std::min(az0, az1)});
							double amax = std::min({t2, std::max(ax0, ax1),
							                        std::max(ay0, ay1),
							                        std::max(az0, az1)});
							if (amin < amax)
							{
								double weight =
								    (amax - amin) * (p2 - p1).getNorm();
								proj_ref += weight * img_arr[k][j][i];
							}
						}
					}
				}
			}
			INFO(rseed_str + " i=" + std::to_string(i) +
			     " p1=" + std::to_string(p1.x) + ", " + std::to_string(p1.y) +
			     ", " + std::to_string(p1.z) + " p2=" + std::to_string(p2.x) +
			     ", " + std::to_string(p2.y) + ", " + std::to_string(p2.z));
			REQUIRE(proj_val == Approx(proj_ref).epsilon(0.02));
			// Adjoint
			double proj_val_t = rand() / (double)RAND_MAX * proj_val;
			double dot_Ax_y = proj_val * proj_val_t;
			double dot_x_Aty = bp_dot(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_Ax_y == Approx(dot_x_Aty).epsilon(0.001));

			// Slow version of ray tracing
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow).epsilon(0.001));
			double dot_x_Aty_slow =
			    bp_dot_slow(lor, img_bp.get(), img.get(), proj_val_t);
			REQUIRE(dot_x_Aty == Approx(dot_x_Aty_slow).epsilon(0.001));
		}
	}
}

TEST_CASE("Siddon-bugs", "[siddon]")
{
	SECTION("check_bug_offset")
	{
		// Fix bug in Siddon fast implementation (pixel offset) causing offset
		// in pixel indices

		// Setup image
		size_t nx = 1;
		size_t ny = 1;
		size_t nz = 89;
		double sx = 38.4;
		double sy = 38.4;
		double sz = 25;
		ImageParams img_params(nx, ny, nz, sx, sy, sz);
		auto img = std::make_unique<ImageOwned>(img_params);
		img->allocate();
		double v = rand() / (double)RAND_MAX * 1000.0;
		img->setValue(v);

		Vector3D p1{0, 0, 26.4843};
		Vector3D p2{0, 0, -26.4292};
		Line3D lor{p1, p2};
		double proj_val =
		    OperatorProjectorSiddon::singleForwardProjection(img.get(), lor);
		REQUIRE(proj_val == Approx(v * sz));
	}

	SECTION("check_bug_zext")
	{
		// Fix bug in Siddon implementation causing segfault and caused by
		// numerical precision

		// Setup image
		size_t nx = 500;
		size_t ny = 500;
		size_t nz = 118;
		double sx = 25.0;
		double sy = 25.0;
		double sz = 23.5;
		ImageParams img_params(nx, ny, nz, sx, sy, sz);
		auto img = std::make_unique<ImageOwned>(img_params);
		img->allocate();
		double v = rand() / (double)RAND_MAX * 1000.0;
		img->setValue(v);

		Vector3D p1{-15.998346, -11.563760, 10.800007};
		Vector3D p2{19.74, 0.0, 13.200009};
		Line3D lor{p1, p2};
		double proj_val =
		    OperatorProjectorSiddon::singleForwardProjection(img.get(), lor);
		REQUIRE(proj_val > 0.0f);

		float proj_val_slow;
		OperatorProjectorSiddon::project_helper<true, false, false>(
		    img.get(), lor, proj_val_slow);
		REQUIRE(proj_val == Approx(proj_val_slow));
	}

	SECTION("check_bug_fast_multi_intersection")
	{
		// Fix bug in Siddon (fast mode FLAG_INCR) caused by crossing of line of
		// response with more than one pixel edge

		// Setup image
		size_t nx = 4;
		size_t ny = 4;
		size_t nz = 4;
		double sx = 4.0;
		double sy = 4.0;
		double sz = 4.0;
		ImageParams img_params(nx, ny, nz, sx, sy, sz);
		auto img = std::make_unique<ImageOwned>(img_params);
		img->allocate();
		// Randomize image content
		Array3DAlias<float> img_arr = img->getArray();
		for (size_t k = 0; k < nz; k++)
		{
			for (size_t j = 0; j < ny; j++)
			{
				for (size_t i = 0; i < nx; i++)
				{
					img_arr[k][j][i] = rand() / (double)RAND_MAX * 10 - 5.0;
				}
			}
		}

		// xy
		{
			Vector3D p1{-2.0, -1.0, 0.0};
			Vector3D p2{2.0, 1.0, 0.0};
			Line3D lor{p1, p2};
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
		}

		// xz
		{
			Vector3D p1{-2.0, 0.0, -1.0};
			Vector3D p2{2.0, 0.0, 1.0};
			Line3D lor{p1, p2};
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
		}

		// yz
		{
			Vector3D p1{0.0, -2.0, -1.0};
			Vector3D p2{0.0, 2.0, 1.0};
			Line3D lor{p1, p2};
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
		}

		// xyz
		{
			Vector3D p1{-2.0, -2.0, -2.0};
			Vector3D p2{2.0, 2.0, 2.0};
			Line3D lor{p1, p2};
			double proj_val = OperatorProjectorSiddon::singleForwardProjection(
			    img.get(), lor);
			float proj_val_slow;
			OperatorProjectorSiddon::project_helper<true, false, false>(
			    img.get(), lor, proj_val_slow);
			REQUIRE(proj_val == Approx(proj_val_slow));
		}
	}
}
