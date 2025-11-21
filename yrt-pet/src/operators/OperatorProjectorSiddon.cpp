/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorSiddon.hpp"
#include "yrt-pet/operators/OperatorProjectorUpdater.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/BinFilter.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/geometry/ProjectorUtils.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/utils/Types.hpp"

#include <algorithm>
#include <vector>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_operatorprojectorsiddon(py::module& m)
{
	auto c = py::class_<OperatorProjectorSiddon, OperatorProjector>(
	    m, "OperatorProjectorSiddon");
	c.def(py::init<const OperatorProjectorParams&, std::vector<Constraint*>>(),
	      "projParams"_a, "constraints"_a);
	c.def(py::init<const OperatorProjectorParams&>(), "projParams"_a);
	c.def_property("num_rays", &OperatorProjectorSiddon::getNumRays,
	               &OperatorProjectorSiddon::setNumRays);
	c.def(
	    "forwardProjection",
	    [](const OperatorProjectorSiddon& self, const Image* in_image,
	       const Line3D& lor, const Vector3D& n1, const Vector3D& n2,
	       const int tid, frame_t dynamicFrame,
	       const TimeOfFlightHelper* tofHelper, float tofValue) -> float
	    {
		    return self.forwardProjection(in_image, lor, n1, n2, tid,
		                                  dynamicFrame, tofHelper, tofValue);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("n1"), py::arg("n2"),
	    py::arg("tid") = 0, py::arg("dynamic_frame") = 0,
	    py::arg("tofHelper") = nullptr, py::arg("tofValue") = 0.0f);
	c.def(
	    "backProjection",
	    [](const OperatorProjectorSiddon& self, Image* in_image,
	       const Line3D& lor, const Vector3D& n1, const Vector3D& n2,
	       float proj_value, int tid, frame_t dynamicFrame,
	       const TimeOfFlightHelper* tofHelper, float tofValue) -> void
	    {
		    self.backProjection(in_image, lor, n1, n2, proj_value, tid,
		                        dynamicFrame, tofHelper, tofValue);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("n1"), py::arg("n2"),
	    py::arg("proj_value"), py::arg("tid") = 0, py::arg("dynamic_rame") = 0,
	    py::arg("tofHelper") = nullptr, py::arg("tofValue") = 0.0f);
	c.def_static(
	    "singleBackProjection",
	    [](Image* in_image, const Line3D& lor, float proj_value,
	       const TimeOfFlightHelper* tofHelper, float tofValue) -> void
	    {
		    OperatorProjectorSiddon::singleBackProjection(
		        in_image, lor, proj_value, tofHelper, tofValue);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("proj_value"),
	    py::arg("tofHelper") = nullptr, py::arg("tofValue") = 0.0f);
	c.def_static(
	    "singleBackProjection",
	    [](Image* in_image, const Line3D& lor, float proj_value,
	       OperatorProjectorUpdater& updater, frame_t dynamicFrame,
	       const TimeOfFlightHelper* tofHelper, float tofValue) -> void
	    {
		    OperatorProjectorSiddon::singleBackProjection(
		        in_image, lor, proj_value, updater, dynamicFrame, tofHelper,
		        tofValue);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("proj_value"),
	    py::arg("updater"), py::arg("dynamicFrame") = 0,
	    py::arg("tofHelper") = nullptr, py::arg("tofValue") = 0.0f);
	c.def_static(
	    "singleForwardProjection",
	    [](const Image* in_image, const Line3D& lor,
	       const TimeOfFlightHelper* tofHelper, float tofValue) -> float
	    {
		    return OperatorProjectorSiddon::singleForwardProjection(
		        in_image, lor, tofHelper, tofValue);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("tofHelper") = nullptr,
	    py::arg("tofValue") = 0.0f);
	c.def_static(
	    "singleForwardProjection",
	    [](const Image* in_image, const Line3D& lor,
	       OperatorProjectorUpdater& updater, frame_t dynamicFrame,
	       const TimeOfFlightHelper* tofHelper, float tofValue) -> float
	    {
		    return OperatorProjectorSiddon::singleForwardProjection(
		        in_image, lor, updater, dynamicFrame, tofHelper, tofValue);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("updater"),
	    py::arg("dynamicFrame"), py::arg("tofHelper") = nullptr,
	    py::arg("tofValue") = 0.0f);
}
}  // namespace yrt

#endif

namespace yrt
{

OperatorProjectorSiddon::OperatorProjectorSiddon(
    const OperatorProjectorParams& pr_projParams,
    const std::vector<Constraint*>& pr_constraints)
    : OperatorProjector(pr_projParams, pr_constraints),
      m_numRays(pr_projParams.numRays)
{
	if (m_numRays > 1)
	{
		mp_lineGen = std::make_unique<std::vector<MultiRayGenerator>>(
		    m_numThreads, MultiRayGenerator{scanner.crystalSize_z,
		                                    scanner.crystalSize_trans});
	}
	ASSERT_MSG_WARNING(
	    mp_projPsfManager == nullptr,
	    "Siddon does not support Projection space PSF. It will be ignored.");
	OperatorProjectorBase::initBinFilter(pr_projParams.projPropertyTypesExtra,
	                                     pr_projParams.numThreads);
}

int OperatorProjectorSiddon::getNumRays() const
{
	return m_numRays;
}

void OperatorProjectorSiddon::setNumRays(int n)
{
	m_numRays = n;
}

std::set<ProjectionPropertyType>
    OperatorProjectorSiddon::getProjectionPropertyTypes() const
{
	std::set<ProjectionPropertyType> projProperties;
	projProperties.insert(ProjectionPropertyType::LOR);
	if (m_numRays > 1)
	{
		projProperties.insert(ProjectionPropertyType::DET_ORIENT);
	}
	// TODO NOW: Make sure this is not done unnecessarily
	projProperties.insert(ProjectionPropertyType::DYNAMIC_FRAME);
	return projProperties;
}

float OperatorProjectorSiddon::forwardProjection(
    const Image* img, const ProjectionProperties& projectionProperties,
    size_t pos) const
{
	auto projPropManager = mp_binFilter->getPropertyManager();
	det_orient_t detOrient;
	if (m_numRays > 1)
	{
		ASSERT(projPropManager.has(ProjectionPropertyType::DET_ORIENT));
		detOrient = projPropManager.getDataValue<det_orient_t>(
		    projectionProperties, pos, ProjectionPropertyType::DET_ORIENT);
	}
	float tofValue = 0.f;
	if (projPropManager.has(ProjectionPropertyType::TOF))
	{
		tofValue = projPropManager.getDataValue<float>(
		    projectionProperties, pos, ProjectionPropertyType::TOF);
	}
	frame_t dynamicFrame = 0;
	if (projPropManager.has(ProjectionPropertyType::DYNAMIC_FRAME))
	{
		dynamicFrame = projPropManager.getDataValue<frame_t>(
		    projectionProperties, pos, ProjectionPropertyType::DYNAMIC_FRAME);
	}
	return forwardProjection(
	    img,
	    projPropManager.getDataValue<Line3D>(projectionProperties, pos,
	                                         ProjectionPropertyType::LOR),
	    detOrient.d1, detOrient.d2, pos, dynamicFrame, mp_tofHelper.get(),
	    tofValue);
}

void OperatorProjectorSiddon::backProjection(
    Image* img, const ProjectionProperties& projectionProperties,
    float projValue, size_t pos) const
{
	auto projPropManager = mp_binFilter->getPropertyManager();
	det_orient_t detOrient;
	if (m_numRays > 1)
	{
		ASSERT(projPropManager.has(ProjectionPropertyType::DET_ORIENT));
		detOrient = projPropManager.getDataValue<det_orient_t>(
		    projectionProperties, pos, ProjectionPropertyType::DET_ORIENT);
	}
	float tofValue = 0.f;
	if (projPropManager.has(ProjectionPropertyType::TOF))
	{
		tofValue = projPropManager.getDataValue<float>(
		    projectionProperties, pos, ProjectionPropertyType::TOF);
	}
	frame_t dynamicFrame = 0;
	if (projPropManager.has(ProjectionPropertyType::DYNAMIC_FRAME))
	{
		dynamicFrame = projPropManager.getDataValue<frame_t>(
		    projectionProperties, pos, ProjectionPropertyType::DYNAMIC_FRAME);
	}
	backProjection(img,
	               projPropManager.getDataValue<Line3D>(
	                   projectionProperties, pos, ProjectionPropertyType::LOR),
	               detOrient.d1, detOrient.d2, projValue, pos, dynamicFrame,
	               mp_tofHelper.get(), tofValue);
}

float OperatorProjectorSiddon::forwardProjection(
    const Image* img, const Line3D& lor, const Vector3D& n1, const Vector3D& n2,
    int tid, frame_t dynamicFrame, const TimeOfFlightHelper* tofHelper,
    float tofValue) const
{
	const ImageParams& params = img->getParams();
	const Vector3D offsetVec = {params.off_x, params.off_y, params.off_z};

	float imProj = 0.;

	int currThread = 0;
	if (m_numRays > 1)
	{
		currThread = tid;
		ASSERT(mp_lineGen != nullptr);
		mp_lineGen->at(currThread).setupGenerator(lor, n1, n2);
	}

	unsigned int seed = 13;
	for (int i_line = 0; i_line < m_numRays; i_line++)
	{
		Line3D randLine = (i_line == 0) ?
		                      lor :
		                      mp_lineGen->at(currThread).getRandomLine(seed);
		randLine.point1 = randLine.point1 - offsetVec;
		randLine.point2 = randLine.point2 - offsetVec;

		float currentProjValue = 0.0;
		if (tofHelper != nullptr)
		{
			project_helper<true, true, true>(
			    const_cast<Image*>(img), randLine, currentProjValue,
			    *mp_updater.get(), dynamicFrame, tid, tofHelper, tofValue);
		}
		else
		{
			project_helper<true, true, false>(
			    const_cast<Image*>(img), randLine, currentProjValue,
			    *mp_updater.get(), dynamicFrame, tid, nullptr, 0);
		}
		imProj += currentProjValue;
	}

	if (m_numRays > 1)
	{
		imProj = imProj / static_cast<float>(m_numRays);
	}

	return imProj;
}

void OperatorProjectorSiddon::backProjection(
    Image* img, const Line3D& lor, const Vector3D& n1, const Vector3D& n2,
    float projValue, int tid, frame_t dynamicFrame,
    const TimeOfFlightHelper* tofHelper, float tofValue) const
{
	const ImageParams& params = img->getParams();
	const Vector3D offsetVec = {params.off_x, params.off_y, params.off_z};


	int currThread = 0;
	float projValuePerLor = projValue;
	if (m_numRays > 1)
	{
		ASSERT(mp_lineGen != nullptr);
		currThread = tid;
		mp_lineGen->at(currThread).setupGenerator(lor, n1, n2);
		projValuePerLor = projValue / static_cast<float>(m_numRays);
	}

	unsigned int seed = 13;
	for (int i_line = 0; i_line < m_numRays; i_line++)
	{
		Line3D randLine = (i_line == 0) ?
		                      lor :
		                      mp_lineGen->at(currThread).getRandomLine(seed);
		randLine.point1 = randLine.point1 - offsetVec;
		randLine.point2 = randLine.point2 - offsetVec;
		if (tofHelper != nullptr)
		{
			project_helper<false, true, true>(img, randLine, projValuePerLor,
			                                  *mp_updater.get(), dynamicFrame,
			                                  tid, tofHelper, tofValue);
		}
		else
		{
			project_helper<false, true, false>(img, randLine, projValuePerLor,
			                                   *mp_updater.get(), dynamicFrame,
			                                   tid, nullptr, 0);
		}
	}
}

float OperatorProjectorSiddon::singleForwardProjection(
    const Image* img, const Line3D& lor, const TimeOfFlightHelper* tofHelper,
    float tofValue)
{
	OperatorProjectorUpdaterDefault3D updater;
	return singleForwardProjection(img, lor, updater, 0, tofHelper, tofValue);
}


float OperatorProjectorSiddon::singleForwardProjection(
    const Image* img, const Line3D& lor, OperatorProjectorUpdater& updater,
    frame_t dynamicFrame, const TimeOfFlightHelper* tofHelper, float tofValue)
{
	float v;
	if (tofHelper != nullptr)
	{
		project_helper<true, true, true>(const_cast<Image*>(img), lor, v,
		                                 updater, dynamicFrame, 0, tofHelper,
		                                 tofValue);
	}
	else
	{
		project_helper<true, true, false>(const_cast<Image*>(img), lor, v,
		                                  updater, dynamicFrame, 0, tofHelper,
		                                  tofValue);
	}
	return v;
}


void OperatorProjectorSiddon::singleBackProjection(
    Image* img, const Line3D& lor, float projValue,
    const TimeOfFlightHelper* tofHelper, float tofValue)
{
	OperatorProjectorUpdaterDefault3D updater;
	singleBackProjection(img, lor, projValue, updater, 0, tofHelper, tofValue);
}

void OperatorProjectorSiddon::singleBackProjection(
    Image* img, const Line3D& lor, float projValue,
    OperatorProjectorUpdater& updater, frame_t dynamicFrame,
    const TimeOfFlightHelper* tofHelper, float tofValue)
{
	if (tofHelper != nullptr)
	{
		project_helper<false, true, true>(img, lor, projValue, updater,
		                                  dynamicFrame, 0, tofHelper, tofValue);
	}
	else
	{
		project_helper<false, true, false>(
		    img, lor, projValue, updater, dynamicFrame, 0, tofHelper, tofValue);
	}
}


enum SIDDON_DIR
{
	DIR_X = 0b001,
	DIR_Y = 0b010,
	DIR_Z = 0b100
};

// Note: FLAG_INCR skips the conversion from physical to logical coordinates by
// moving from pixel to pixel as the ray parameter is updated.  This may cause
// issues near the last intersection, which must therefore be handled with extra
// care.  Speedups around 20% were measured with FLAG_INCR=true.  Both versions
// are compared in tests, the "faster" version (FLAG_INCR=true) is used by
// default.
template <bool IS_FWD, bool FLAG_INCR, bool FLAG_TOF>
void OperatorProjectorSiddon::project_helper(
    Image* img, const Line3D& lor, float& value,
    OperatorProjectorUpdater& updater, frame_t dynamicFrame, int tid,
    const TimeOfFlightHelper* tofHelper, float tofValue)
{
	if (IS_FWD)
	{
		value = 0.0f;
	}

	ImageParams params = img->getParams();

	const Vector3D& p1 = lor.point1;
	const Vector3D& p2 = lor.point2;
	// 1. Intersection with FOV
	float t0;
	float t1;
	// Intersection with (centered) FOV cylinder
	float A = (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y);
	float B = 2.0f * ((p2.x - p1.x) * p1.x + (p2.y - p1.y) * p1.y);
	float C = p1.x * p1.x + p1.y * p1.y - params.fovRadius * params.fovRadius;
	float Delta = B * B - 4.0f * A * C;
	if (A != 0.0f)
	{
		if (Delta <= 0.0f)
		{
			t0 = 1.0f;
			t1 = 0.0f;
			return;
		}
		t0 = (-B - sqrt(Delta)) / (2.0f * A);
		t1 = (-B + sqrt(Delta)) / (2.0f * A);
	}
	else
	{
		t0 = 0.0f;
		t1 = 1.0f;
	}

	float d_norm = (p1 - p2).getNorm();
	bool flat_x = (p1.x == p2.x);
	bool flat_y = (p1.y == p2.y);
	bool flat_z = (p1.z == p2.z);
	float inv_p12_x = flat_x ? 0.0f : 1.0f / (p2.x - p1.x);
	float inv_p12_y = flat_y ? 0.0f : 1.0f / (p2.y - p1.y);
	float inv_p12_z = flat_z ? 0.0f : 1.0f / (p2.z - p1.z);
	int dir_x = (inv_p12_x >= 0.0f) ? 1 : -1;
	int dir_y = (inv_p12_y >= 0.0f) ? 1 : -1;
	int dir_z = (inv_p12_z >= 0.0f) ? 1 : -1;

	// 2. Intersection with volume
	float dx = params.vx;
	float dy = params.vy;
	float dz = params.vz;
	float inv_dx = 1.0f / dx;
	float inv_dy = 1.0f / dy;
	float inv_dz = 1.0f / dz;

	float x0 = -params.length_x * 0.5f;
	float x1 = params.length_x * 0.5f;
	float y0 = -params.length_y * 0.5f;
	float y1 = params.length_y * 0.5f;
	float z0 = -params.length_z * 0.5f;
	float z1 = params.length_z * 0.5f;
	float ax_min, ax_max, ay_min, ay_max, az_min, az_max;
	util::get_alpha(-0.5f * params.length_x, 0.5f * params.length_x, p1.x, p2.x,
	                inv_p12_x, ax_min, ax_max);
	util::get_alpha(-0.5f * params.length_y, 0.5f * params.length_y, p1.y, p2.y,
	                inv_p12_y, ay_min, ay_max);
	util::get_alpha(-0.5f * params.length_z, 0.5f * params.length_z, p1.z, p2.z,
	                inv_p12_z, az_min, az_max);
	float amin = std::max({0.0f, t0, ax_min, ay_min, az_min});
	float amax = std::min({1.0f, t1, ax_max, ay_max, az_max});
	if (FLAG_TOF)
	{
		float amin_tof, amax_tof;
		tofHelper->getAlphaRange(amin_tof, amax_tof, d_norm, tofValue);
		amin = std::max(amin, amin_tof);
		amax = std::min(amax, amax_tof);
	}

	float a_cur = amin;
	float a_next = -1.0f;
	float x_cur = (inv_p12_x > 0.0f) ? x0 : x1;
	float y_cur = (inv_p12_y > 0.0f) ? y0 : y1;
	float z_cur = (inv_p12_z > 0.0f) ? z0 : z1;
	if ((inv_p12_x >= 0.0f && p1.x > x1) || (inv_p12_x < 0.0f && p1.x < x0) ||
	    (inv_p12_y >= 0.0f && p1.y > y1) || (inv_p12_y < 0.0f && p1.y < y0) ||
	    (inv_p12_z >= 0.0f && p1.z > z1) || (inv_p12_z < 0.0f && p1.z < z0))
	{
		return;
	}
	// Move starting point inside FOV
	float ax_next = flat_x ? std::numeric_limits<float>::max() : ax_min;
	if (!flat_x)
	{
		int kx = (int)ceil(dir_x * (a_cur * (p2.x - p1.x) - x_cur + p1.x) / dx);
		x_cur += kx * dir_x * dx;
		ax_next = (x_cur - p1.x) * inv_p12_x;
	}
	float ay_next = flat_y ? std::numeric_limits<float>::max() : ay_min;
	if (!flat_y)
	{
		int ky = (int)ceil(dir_y * (a_cur * (p2.y - p1.y) - y_cur + p1.y) / dy);
		y_cur += ky * dir_y * dy;
		ay_next = (y_cur - p1.y) * inv_p12_y;
	}
	float az_next = flat_z ? std::numeric_limits<float>::max() : az_min;
	if (!flat_z)
	{
		int kz = (int)ceil(dir_z * (a_cur * (p2.z - p1.z) - z_cur + p1.z) / dz);
		z_cur += kz * dir_z * dz;
		az_next = (z_cur - p1.z) * inv_p12_z;
	}
	// Pixel location (move pixel to pixel instead of calculating position for
	// each intersection)
	bool flag_first = true;
	int vx = -1;
	int vy = -1;
	int vz = -1;
	// The dir variables operate as binary bit-flags to determine in which
	// direction the current pixel should move: format 0bzyx (where z, y and x
	// are bits set to 1 when the pixel should move in the corresponding
	// direction, e.g. 0b101 moves in the z and x directions)
	short dir_prev = -1;
	short dir_next = -1;

	// Prepare data pointer (this assumes that the data is stored as a
	// contiguous array)
	float* raw_img_ptr = img->getRawPointer();
	// float* cur_img_ptr = nullptr;
	size_t offset_img_ptr = 0;
	int num_x = params.nx;
	int num_xy = params.nx * params.ny;

	float ax_next_prev = ax_next;
	float ay_next_prev = ay_next;
	float az_next_prev = az_next;

	// 3. Integrate along ray
	bool flag_done = false;
	while (a_cur < amax && !flag_done)
	{
		// Find next intersection (along x, y or z)
		dir_next = 0b000;
		if (ax_next_prev <= ay_next_prev && ax_next_prev <= az_next_prev)
		{
			a_next = ax_next;
			x_cur += dir_x * dx;
			ax_next = (x_cur - p1.x) * inv_p12_x;
			dir_next |= SIDDON_DIR::DIR_X;
		}
		if (ay_next_prev <= ax_next_prev && ay_next_prev <= az_next_prev)
		{
			a_next = ay_next;
			y_cur += dir_y * dy;
			ay_next = (y_cur - p1.y) * inv_p12_y;
			dir_next |= SIDDON_DIR::DIR_Y;
		}
		if (az_next_prev <= ax_next_prev && az_next_prev <= ay_next_prev)
		{
			a_next = az_next;
			z_cur += dir_z * dz;
			az_next = (z_cur - p1.z) * inv_p12_z;
			dir_next |= SIDDON_DIR::DIR_Z;
		}
		// Clip to FOV range
		if (a_next > amax)
		{
			a_next = amax;
		}
		if (a_cur >= a_next)
		{
			ax_next_prev = ax_next;
			ay_next_prev = ay_next;
			az_next_prev = az_next;
			continue;
		}
		// Determine pixel location
		float tof_weight = 1.0f;
		float a_mid = 0.5f * (a_cur + a_next);
		if (FLAG_TOF)
		{
			tof_weight = tofHelper->getWeight(d_norm, tofValue, a_cur * d_norm,
			                                  a_next * d_norm);
		}
		if (!FLAG_INCR || flag_first)
		{
			vx = (int)((p1.x + a_mid * (p2.x - p1.x) + params.length_x / 2.0f) *
			           inv_dx);
			vy = (int)((p1.y + a_mid * (p2.y - p1.y) + params.length_y / 2.0f) *
			           inv_dy);
			vz = (int)((p1.z + a_mid * (p2.z - p1.z) + params.length_z / 2.0f) *
			           inv_dz);
			offset_img_ptr = vz * num_xy + vy * num_x;
			flag_first = false;
			if (vx < 0 || vx >= params.nx || vy < 0 || vy >= params.ny ||
			    vz < 0 || vz >= params.nz)
			{
				flag_done = true;
			}
		}
		else
		{
			if (dir_prev & SIDDON_DIR::DIR_X)
			{
				vx += dir_x;
				if (vx < 0 || vx >= params.nx)
				{
					flag_done = true;
				}
			}
			if (dir_prev & SIDDON_DIR::DIR_Y)
			{
				vy += dir_y;
				if (vy < 0 || vy >= params.ny)
				{
					flag_done = true;
				}
				else
				{
					offset_img_ptr += dir_y * num_x;
				}
			}
			if (dir_prev & SIDDON_DIR::DIR_Z)
			{
				vz += dir_z;
				if (vz < 0 || vz >= params.nz)
				{
					flag_done = true;
				}
				else
				{
					offset_img_ptr += dir_z * num_xy;
				}
			}
		}
		if (flag_done)
		{
			continue;
		}
		dir_prev = dir_next;
		float weight = (a_next - a_cur) * d_norm;
		const size_t numVoxelsPerFrame = params.nx * params.ny * params.nz;
		const size_t totalOffset = vx + offset_img_ptr;
		if (FLAG_TOF)
		{
			weight *= tof_weight;
		}
		if (IS_FWD)
		{
			value += updater.forwardUpdate(weight, raw_img_ptr, totalOffset,
			                               dynamicFrame, numVoxelsPerFrame);
		}
		else
		{
			updater.backUpdate(value, weight, raw_img_ptr, totalOffset,
			                   dynamicFrame, numVoxelsPerFrame, tid);
		}
		a_cur = a_next;
		ax_next_prev = ax_next;
		ay_next_prev = ay_next;
		az_next_prev = az_next;
	}
}


// Explicit instantiation of slow version used in tests
template void OperatorProjectorSiddon::project_helper<true, false, true>(
    Image* img, const Line3D&, float&, OperatorProjectorUpdater&, int, int,
    const TimeOfFlightHelper*, float);
template void OperatorProjectorSiddon::project_helper<false, false, true>(
    Image* img, const Line3D&, float&, OperatorProjectorUpdater&, int, int,
    const TimeOfFlightHelper*, float);
template void OperatorProjectorSiddon::project_helper<true, false, false>(
    Image* img, const Line3D&, float&, OperatorProjectorUpdater&, int, int,
    const TimeOfFlightHelper*, float);
template void OperatorProjectorSiddon::project_helper<false, false, false>(
    Image* img, const Line3D&, float&, OperatorProjectorUpdater&, int, int,
    const TimeOfFlightHelper*, float);
}  // namespace yrt
