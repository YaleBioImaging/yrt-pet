/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorDD.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/ProjectorUtils.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#include <algorithm>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace py::literals;

namespace yrt
{


void py_setup_operatorprojectordd(py::module& m)
{
	auto c = py::class_<OperatorProjectorDD, OperatorProjector>(
	    m, "OperatorProjectorDD");

	c.def(py::init<const Scanner&, float, int, const std::string&>(),
	      "scanner"_a, "tofWidth_ps"_a = 0.f, "tofNumStd"_a = -1,
	      "projPsf_fname"_a = "");
	c.def(py::init<const OperatorProjectorParams&>(), "projParams"_a);

	c.def(
	    "forwardProjection",
	    [](const OperatorProjectorDD& self, const Image* in_image,
	       const Line3D& lor, const Vector3D& n1, const Vector3D& n2,
	       const TimeOfFlightHelper* tofHelper, float tofValue) -> float
	    {
		    return self.forwardProjection(in_image, lor, n1, n2, 0, tofHelper,
		                                  tofValue, nullptr);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("n1"), py::arg("n2"),
	    py::arg("tofHelper") = nullptr, py::arg("tofValue") = 0.0f);
	c.def(
	    "backProjection",
	    [](const OperatorProjectorDD& self, Image* in_image, const Line3D& lor,
	       const Vector3D& n1, const Vector3D& n2, float proj_value,
	       const TimeOfFlightHelper* tofHelper, float tofValue)
	    {
		    self.backProjection(in_image, lor, n1, n2, proj_value, 0, tofHelper,
		                        tofValue, nullptr);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("n1"), py::arg("n2"),
	    py::arg("proj_value"), py::arg("tofHelper") = nullptr,
	    py::arg("tofValue") = 0.0f);

	c.def_static("get_overlap", &OperatorProjectorDD::get_overlap);
}
}  // namespace yrt

#endif

namespace yrt
{
OperatorProjectorDD::OperatorProjectorDD(const Scanner& pr_scanner,
                                         float tofWidth_ps, int tofNumStd,
                                         const std::string& projPsf_fname)
    : OperatorProjector{pr_scanner, tofWidth_ps, tofNumStd, projPsf_fname}
{
}

OperatorProjectorDD::OperatorProjectorDD(
    const OperatorProjectorParams& p_projParams)
    : OperatorProjector{p_projParams}
{
}

float OperatorProjectorDD::forwardProjection(
    const Image* img, const ProjectionProperties& projectionProperties,
    int tid) const
{
	return forwardProjection(
	    img, projectionProperties.lor, projectionProperties.detOrient.d1,
	    projectionProperties.detOrient.d2, mp_tofHelper.get(),
	    projectionProperties.tofValue, mp_projPsfManager.get());
}

void OperatorProjectorDD::backProjection(
    Image* img, const ProjectionProperties& projectionProperties,
    float projValue, int tid) const
{
	backProjection(
	    img, projectionProperties.lor, projectionProperties.detOrient.d1,
	    projectionProperties.detOrient.d2, projValue, mp_tofHelper.get(),
	    projectionProperties.tofValue, mp_projPsfManager.get());
}

float OperatorProjectorDD::forwardProjection(
    const Image* in_image, const Line3D& lor, const Vector3D& n1,
    const Vector3D& n2, int /*tid*/, const TimeOfFlightHelper* tofHelper,
    float tofValue, const ProjectionPsfManager* psfManager) const
{
	float v = 0;
	if (tofHelper != nullptr)
	{
		dd_project_ref<true, true>(const_cast<Image*>(in_image), lor, n1, n2, v,
		                           tofHelper, tofValue, psfManager);
	}
	else
	{
		dd_project_ref<true, false>(const_cast<Image*>(in_image), lor, n1, n2,
		                            v, nullptr, tofValue, psfManager);
	}
	return v;
}

void OperatorProjectorDD::backProjection(
    Image* in_image, const Line3D& lor, const Vector3D& n1, const Vector3D& n2,
    float proj_value, int /*tid*/, const TimeOfFlightHelper* tofHelper,
    float tofValue, const ProjectionPsfManager* psfManager) const
{
	if (tofHelper != nullptr)
	{
		dd_project_ref<false, true>(in_image, lor, n1, n2, proj_value,
		                            tofHelper, tofValue, psfManager);
	}
	else
	{
		dd_project_ref<false, false>(in_image, lor, n1, n2, proj_value,
		                             tofHelper, tofValue, psfManager);
	}
}

float OperatorProjectorDD::get_overlap_safe(float p0, float p1, float d0,
                                            float d1)
{
	return std::min(p1, d1) - std::max(p0, d0);
}

float OperatorProjectorDD::get_overlap_safe(
    const float p0, const float p1, const float d0, const float d1,
    const ProjectionPsfManager* psfManager, const float* psfKernel)
{
	if (psfManager != nullptr)
	{
		return psfManager->getWeight(psfKernel, p0 - d1, p1 - d0);
	}
	return get_overlap_safe(p0, p1, d0, d1);
}

float OperatorProjectorDD::get_overlap(const float p0, const float p1,
                                       const float d0, const float d1,
                                       const ProjectionPsfManager* psfManager,
                                       const float* psfKernel)
{
	return std::max(0.f,
	                get_overlap_safe(p0, p1, d0, d1, psfManager, psfKernel));
}

template <bool IS_FWD, bool FLAG_TOF>
void OperatorProjectorDD::dd_project_ref(
    Image* in_image, const Line3D& lor, const Vector3D& n1, const Vector3D& n2,
    float& proj_value, const TimeOfFlightHelper* tofHelper, float tofValue,
    const ProjectionPsfManager* psfManager) const
{
	if constexpr (IS_FWD)
	{
		proj_value = 0.0f;
	}
	const ImageParams& params = in_image->getParams();
	const Vector3D offsetVec = {params.off_x, params.off_y, params.off_z};
	Line3D lorWithoffset = lor;
	lorWithoffset.point1 = lorWithoffset.point1 - offsetVec;
	lorWithoffset.point2 = lorWithoffset.point2 - offsetVec;

	const Vector3D& d1 = lorWithoffset.point1;
	const Vector3D& d2 = lorWithoffset.point2;
	const Vector3D d1_minus_d2 = d1 - d2;
	const bool flag_y = std::abs(d1_minus_d2.y) > std::abs(d1_minus_d2.x);
	const float d_norm = d1_minus_d2.getNorm();

	// Scanner size
	const float thickness_z = scanner.crystalSize_z;
	const float thickness_trans = scanner.crystalSize_trans;

	// PSF
	const float* psfKernel = nullptr;
	float detFootprintExt = 0.f;
	if (psfManager != nullptr)
	{
		psfKernel = psfManager->getKernel(lorWithoffset, !IS_FWD);
		detFootprintExt = psfManager->getHalfWidth_mm();
	}
	// Pixel limits (ignore detector width)
	float* raw_img_ptr = in_image->getRawPointer();
	const int num_xy = params.nx * params.ny;
	const float dx = params.vx;
	const float dy = params.vy;
	const float dz = params.vz;

	const float inv_d12_x = (d1.x == d2.x) ? 0.0f : 1.0f / (d2.x - d1.x);
	const float inv_d12_y = (d1.y == d2.y) ? 0.0f : 1.0f / (d2.y - d1.y);
	const float inv_d12_z = (d1.z == d2.z) ? 0.0f : 1.0f / (d2.z - d1.z);

	float ax_min, ax_max, ay_min, ay_max, az_min, az_max;
	util::get_alpha(-0.5f * (params.length_x - dx),
	                0.5f * (params.length_x - dx), d1.x, d2.x, inv_d12_x,
	                ax_min, ax_max);
	util::get_alpha(-0.5f * (params.length_y - dy),
	                0.5f * (params.length_y - dy), d1.y, d2.y, inv_d12_y,
	                ay_min, ay_max);
	util::get_alpha(-0.5f * (params.length_z - dz),
	                0.5f * (params.length_z - dz), d1.z, d2.z, inv_d12_z,
	                az_min, az_max);
	float amin = std::max({0.0f, ax_min, ay_min, az_min});
	float amax = std::min({1.0f, ax_max, ay_max, az_max});
	if constexpr (FLAG_TOF)
	{
		float amin_tof, amax_tof;
		tofHelper->getAlphaRange(amin_tof, amax_tof, d_norm, tofValue);
		amin = std::max(amin, amin_tof);
		amax = std::min(amax, amax_tof);
	}

	const float x_0 = d1.x + amin * (d2.x - d1.x);
	const float y_0 = d1.y + amin * (d2.y - d1.y);
	const float x_1 = d1.x + amax * (d2.x - d1.x);
	const float y_1 = d1.y + amax * (d2.y - d1.y);
	const int x_i_0 = std::floor(x_0 / dx + 0.5f * (params.nx - 1) + 0.5f);
	const int y_i_0 = std::floor(y_0 / dy + 0.5f * (params.ny - 1) + 0.5f);
	const int x_i_1 = std::floor(x_1 / dx + 0.5f * (params.nx - 1) + 0.5f);
	const int y_i_1 = std::floor(y_1 / dy + 0.5f * (params.ny - 1) + 0.5f);

	float d1_i, d2_i, n1_i, n2_i;
	if (flag_y)
	{
		d1_i = d1.y;
		d2_i = d2.y;
		n1_i = n1.y;
		n2_i = n2.y;
	}
	else
	{
		d1_i = d1.x;
		d2_i = d2.x;
		n1_i = n1.x;
		n2_i = n2.x;
	}

	// Normal vectors (in-plane and through-plane)
	const float n1_xy_norm2 = n1.x * n1.x + n1.y * n1.y;
	const float n1_xy_norm = std::sqrt(n1_xy_norm2);
	const float n1_p_x = n1.y / n1_xy_norm;
	const float n1_p_y = -n1.x / n1_xy_norm;
	const float n1_z_norm =
	    std::sqrt((n1.x * n1.z) * (n1.x * n1.z) +
	              (n1.y * n1.z) * (n1.y * n1.z) + n1_xy_norm2);
	const float n1_p_i = (n1_i * n1.z) / n1_z_norm;
	const float n1_p_z = -n1_xy_norm2 / n1_z_norm;
	const float n2_xy_norm2 = n2.x * n2.x + n2.y * n2.y;
	const float n2_xy_norm = std::sqrt(n2_xy_norm2);
	const float n2_p_x = n2.y / n2_xy_norm;
	const float n2_p_y = -n2.x / n2_xy_norm;
	const float n2_z_norm =
	    std::sqrt((n2.x * n2.z) * (n2.x * n2.z) +
	              (n2.y * n2.z) * (n2.y * n2.z) + n2_xy_norm2);
	const float n2_p_i = (n2_i * n2.z) / n2_z_norm;
	const float n2_p_z = -n2_xy_norm2 / n2_z_norm;

	// In-plane detector edges
	const float half_thickness_trans = thickness_trans * 0.5f;
	const float d1_xy_lo_x = d1.x - half_thickness_trans * n1_p_x;
	const float d1_xy_lo_y = d1.y - half_thickness_trans * n1_p_y;
	const float d1_xy_hi_x = d1.x + half_thickness_trans * n1_p_x;
	const float d1_xy_hi_y = d1.y + half_thickness_trans * n1_p_y;
	const float d2_xy_lo_x = d2.x - half_thickness_trans * n2_p_x;
	const float d2_xy_lo_y = d2.y - half_thickness_trans * n2_p_y;
	const float d2_xy_hi_x = d2.x + half_thickness_trans * n2_p_x;
	const float d2_xy_hi_y = d2.y + half_thickness_trans * n2_p_y;

	// Through-plane detector edges
	const float half_thickness_z = thickness_z * 0.5f;
	const float d1_z_lo_i = d1_i - half_thickness_z * n1_p_i;
	const float d1_z_lo_z = d1.z - half_thickness_z * n1_p_z;
	const float d1_z_hi_i = d1_i + half_thickness_z * n1_p_i;
	const float d1_z_hi_z = d1.z + half_thickness_z * n1_p_z;
	const float d2_z_lo_i = d2_i - half_thickness_z * n2_p_i;
	const float d2_z_lo_z = d2.z - half_thickness_z * n2_p_z;
	const float d2_z_hi_i = d2_i + half_thickness_z * n2_p_i;
	const float d2_z_hi_z = d2.z + half_thickness_z * n2_p_z;

	float xy_i_0, xy_i_1;
	float lxy, lyx, dxy, dyx;
	int nyx;
	float d1_xy_lo, d1_xy_hi, d2_xy_lo, d2_xy_hi;
	float d1_yx_lo, d1_yx_hi, d2_yx_lo, d2_yx_hi;
	if (flag_y)
	{
		xy_i_0 = std::max(0, std::min(y_i_0, y_i_1));
		xy_i_1 = std::min(params.ny - 1, std::max(y_i_0, y_i_1));
		lxy = params.length_y;
		dxy = params.vy;
		lyx = params.length_x;
		dyx = params.vx;
		nyx = params.nx;
		d1_xy_lo = d1_xy_lo_y;
		d1_xy_hi = d1_xy_hi_y;
		d2_xy_lo = d2_xy_lo_y;
		d2_xy_hi = d2_xy_hi_y;
		d1_yx_lo = d1_xy_lo_x;
		d1_yx_hi = d1_xy_hi_x;
		d2_yx_lo = d2_xy_lo_x;
		d2_yx_hi = d2_xy_hi_x;
	}
	else
	{
		xy_i_0 = std::max(0, std::min(x_i_0, x_i_1));
		xy_i_1 = std::min(params.nx - 1, std::max(x_i_0, x_i_1));
		lxy = params.length_x;
		dxy = params.vx;
		lyx = params.length_y;
		dyx = params.vy;
		nyx = params.ny;
		d1_xy_lo = d1_xy_lo_x;
		d1_xy_hi = d1_xy_hi_x;
		d2_xy_lo = d2_xy_lo_x;
		d2_xy_hi = d2_xy_hi_x;
		d1_yx_lo = d1_xy_lo_y;
		d1_yx_hi = d1_xy_hi_y;
		d2_yx_lo = d2_xy_lo_y;
		d2_yx_hi = d2_xy_hi_y;
	}
	float dxy_cos_theta;
	if (d1_i != d2_i)
	{
		dxy_cos_theta = dxy / (std::abs(d1_i - d2_i) / d_norm);
	}
	else
	{
		dxy_cos_theta = dxy;
	}

	for (int xyi = xy_i_0; xyi <= xy_i_1; xyi++)
	{
		const float pix_xy = -0.5f * lxy + (xyi + 0.5f) * dxy;
		const float a_xy_lo = (pix_xy - d1_xy_lo) / (d2_xy_hi - d1_xy_lo);
		const float a_xy_hi = (pix_xy - d1_xy_hi) / (d2_xy_lo - d1_xy_hi);
		const float a_z_lo = (pix_xy - d1_z_lo_i) / (d2_z_lo_i - d1_z_lo_i);
		const float a_z_hi = (pix_xy - d1_z_hi_i) / (d2_z_hi_i - d1_z_hi_i);
		float dd_yx_r_0 = d1_yx_lo + a_xy_lo * (d2_yx_hi - d1_yx_lo);
		float dd_yx_r_1 = d1_yx_hi + a_xy_hi * (d2_yx_lo - d1_yx_hi);
		if (dd_yx_r_0 > dd_yx_r_1)
		{
			std::swap(dd_yx_r_0, dd_yx_r_1);
		}
		const float widthFrac_yx = dd_yx_r_1 - dd_yx_r_0;
		// Save bounds without extension for overlap calculation
		const float dd_yx_r_0_ov = dd_yx_r_0;
		const float dd_yx_r_1_ov = dd_yx_r_1;
		dd_yx_r_0 -= detFootprintExt;
		dd_yx_r_1 += detFootprintExt;
		const float dd_yx_i_offset = (nyx - 1) / 2.f;
		const float inv_dyx = 1.0f / dyx;
		const int dd_yx_i_0 = std::max(
		    0,
		    static_cast<int>(std::rintf(dd_yx_r_0 * inv_dyx + dd_yx_i_offset)));
		const int dd_yx_i_1 = std::min(
		    nyx - 1,
		    static_cast<int>(std::rintf(dd_yx_r_1 * inv_dyx + dd_yx_i_offset)));
		for (int yxi = dd_yx_i_0; yxi <= dd_yx_i_1; yxi++)
		{
			const float pix_yx = -0.5f * lyx + (yxi + 0.5f) * dyx;
			float dd_z_r_0 = d1_z_lo_z + a_z_lo * (d2_z_lo_z - d1_z_lo_z);
			float dd_z_r_1 = d1_z_hi_z + a_z_hi * (d2_z_hi_z - d1_z_hi_z);
			if (dd_z_r_0 > dd_z_r_1)
			{
				std::swap(dd_z_r_0, dd_z_r_1);
			}
			const float widthFrac_z = dd_z_r_1 - dd_z_r_0;
			const float half_dyx = dyx * 0.5f;
			const float dd_yx_p_0 = pix_yx - half_dyx;
			const float dd_yx_p_1 = pix_yx + half_dyx;
			if (dd_yx_r_1 >= dd_yx_p_0 && dd_yx_r_0 < dd_yx_p_1)
			{
				const float weight_xy =
				    get_overlap_safe(dd_yx_p_0, dd_yx_p_1, dd_yx_r_0_ov,
				                     dd_yx_r_1_ov, psfManager, psfKernel);
				const float weight_xy_s = weight_xy / widthFrac_yx;
				const float dd_z_i_offset = (params.nz - 1) * 0.5f;
				const float inv_dz = 1.0f / dz;
				const int dd_z_i_0 =
				    std::max(0, static_cast<int>(std::rintf(dd_z_r_0 * inv_dz +
				                                            dd_z_i_offset)));
				const int dd_z_i_1 = std::min(
				    params.nz - 1, static_cast<int>(std::rintf(
				                       dd_z_r_1 * inv_dz + dd_z_i_offset)));
				for (int zi = dd_z_i_0; zi <= dd_z_i_1; zi++)
				{
					const float pix_z =
					    -0.5f * params.length_z + (zi + 0.5f) * params.vz;
					float tof_weight = 1.f;
					if constexpr (FLAG_TOF)
					{
						const float a_lo =
						    (pix_xy - d1_i - 0.5f * dxy) / (d2_i - d1_i);
						const float a_hi =
						    (pix_xy - d1_i + 0.5f * dxy) / (d2_i - d1_i);
						tof_weight = tofHelper->getWeight(
						    d_norm, tofValue, a_lo * d_norm, a_hi * d_norm);
					}

					const float dd_z_p_0 = pix_z - params.vz * 0.5f;
					const float dd_z_p_1 = pix_z + params.vz * 0.5f;
					if (dd_z_r_1 >= dd_z_p_0 && dd_z_r_0 < dd_z_p_1)
					{
						const float weight_z = get_overlap_safe(
						    dd_z_p_0, dd_z_p_1, dd_z_r_0, dd_z_r_1);
						const float weight_z_s = weight_z / widthFrac_z;
						size_t idx = zi * num_xy;
						if (flag_y)
						{
							idx += params.nx * xyi + yxi;
						}
						else
						{
							idx += params.nx * yxi + xyi;
						}
						float weight = weight_xy_s * weight_z_s * dxy_cos_theta;
						if constexpr (FLAG_TOF)
						{
							weight *= tof_weight;
						}

						if constexpr (IS_FWD)
						{
							float* ptr = raw_img_ptr + idx;
							proj_value += (*ptr) * weight;
						}
						else
						{
							std::atomic_ref<float> atomic_elem(
							    raw_img_ptr[idx]);
							atomic_elem.fetch_add(proj_value * weight);
						}
					}
				}
			}
		}
	}
}

template void OperatorProjectorDD::dd_project_ref<true, false>(
    Image*, const Line3D&, const Vector3D&, const Vector3D&, float&,
    const TimeOfFlightHelper*, float, const ProjectionPsfManager*) const;
template void OperatorProjectorDD::dd_project_ref<false, false>(
    Image*, const Line3D&, const Vector3D&, const Vector3D&, float&,
    const TimeOfFlightHelper*, float, const ProjectionPsfManager*) const;
template void OperatorProjectorDD::dd_project_ref<true, true>(
    Image*, const Line3D&, const Vector3D&, const Vector3D&, float&,
    const TimeOfFlightHelper*, float, const ProjectionPsfManager*) const;
template void OperatorProjectorDD::dd_project_ref<false, true>(
    Image*, const Line3D&, const Vector3D&, const Vector3D&, float&,
    const TimeOfFlightHelper*, float, const ProjectionPsfManager*) const;

}  // namespace yrt
