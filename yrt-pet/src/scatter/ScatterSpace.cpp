/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/ScatterSpace.hpp"

#include "yrt-pet/geometry/Constants.hpp"

#define FORBID_INPUT_ERROR_MESSAGE                                          \
	"The scatter space should not be used to gather LORs, do projections, " \
	"or reconstructions. It is meant only for corrections."

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_scatterspace(py::module& m)
{
	// ScatterSpaceIndex struct
	py::class_<ScatterSpace::ScatterSpaceIndex>(m, "ScatterSpaceIndex")
	    .def(py::init<>())
	    .def_readwrite("tofBin", &ScatterSpace::ScatterSpaceIndex::tofBin)
	    .def_readwrite("planeIndex1",
	                   &ScatterSpace::ScatterSpaceIndex::planeIndex1)
	    .def_readwrite("angleIndex1",
	                   &ScatterSpace::ScatterSpaceIndex::angleIndex1)
	    .def_readwrite("planeIndex2",
	                   &ScatterSpace::ScatterSpaceIndex::planeIndex2)
	    .def_readwrite("angleIndex2",
	                   &ScatterSpace::ScatterSpaceIndex::angleIndex2);

	// ScatterSpacePosition struct
	py::class_<ScatterSpace::ScatterSpacePosition>(m, "ScatterSpacePosition")
	    .def(py::init<>())
	    .def_readwrite("tof_ps", &ScatterSpace::ScatterSpacePosition::tof_ps)
	    .def_readwrite("planePosition1",
	                   &ScatterSpace::ScatterSpacePosition::planePosition1)
	    .def_readwrite("angle1", &ScatterSpace::ScatterSpacePosition::angle1)
	    .def_readwrite("planePosition2",
	                   &ScatterSpace::ScatterSpacePosition::planePosition2)
	    .def_readwrite("angle2", &ScatterSpace::ScatterSpacePosition::angle2);

	// ScatterSpace class
	auto c = py::class_<ScatterSpace, std::shared_ptr<ScatterSpace>>(
	    m, "ScatterSpace");
	c.def(py::init<const Scanner&, const std::string&>());
	c.def(py::init<const Scanner&, size_t, size_t, size_t>());

	// I/O
	c.def("readFromFile", &ScatterSpace::readFromFile);
	c.def("writeToFile", &ScatterSpace::writeToFile);

	// Access methods
	c.def("getNearestNeighborIndex", &ScatterSpace::getNearestNeighborIndex);
	c.def("getNearestNeighborValue", &ScatterSpace::getNearestNeighborValue);
	c.def("getLinearInterpolationValue",
	      &ScatterSpace::getLinearInterpolationValue);

	// Index ↔ Position conversions
	c.def("getTOF_ps", &ScatterSpace::getTOF_ps);
	c.def("getPlanePosition", &ScatterSpace::getPlanePosition);
	c.def("getAngle", &ScatterSpace::getAngle);
	c.def("getTOFBin", &ScatterSpace::getTOFBin);
	c.def("getPlaneIndex", &ScatterSpace::getPlaneIndex);
	c.def("getAngleIndex", &ScatterSpace::getAngleIndex);

	// Get/Set values
	c.def("getValue", static_cast<float (ScatterSpace::*)(
	                      const ScatterSpace::ScatterSpaceIndex& idx) const>(
	                      &ScatterSpace::getValue));
	c.def("getValue", static_cast<float (ScatterSpace::*)(
	                      size_t tofBin, size_t planeIndex1, size_t angleIndex1,
	                      size_t planeIndex2, size_t angleIndex2) const>(
	                      &ScatterSpace::getValue));
	c.def("setValue",
	      py::overload_cast<const ScatterSpace::ScatterSpaceIndex&, float>(
	          &ScatterSpace::setValue));
	c.def("setValue",
	      py::overload_cast<size_t, size_t, size_t, size_t, size_t, float>(
	          &ScatterSpace::setValue));

	// Utility functions
	c.def("symmetrize", &ScatterSpace::symmetrize);
	c.def("clampTOF", &ScatterSpace::clampTOF);
	c.def("clampPlanePosition", &ScatterSpace::clampPlanePosition);
	c.def_static("wrapAngle", &ScatterSpace::wrapAngle);
	c.def("wrapAngleIndex", &ScatterSpace::wrapAngleIndex);

	// Size information
	c.def("getNumTOFBins", &ScatterSpace::getNumTOFBins);
	c.def("getNumPlanes", &ScatterSpace::getNumPlanes);
	c.def("getNumAngles", &ScatterSpace::getNumAngles);
	c.def("count", &ScatterSpace::count);

	// Step sizes
	c.def("getTOFBinStep_ps", &ScatterSpace::getTOFBinStep_ps);
	c.def("getPlaneStep", &ScatterSpace::getPlaneStep);
	c.def("getAngleStep", &ScatterSpace::getAngleStep);

	// Scanner properties
	c.def("getAxialFOV", &ScatterSpace::getAxialFOV);
	c.def("getRadius", &ScatterSpace::getRadius);
	c.def("getDiameter", &ScatterSpace::getDiameter);
	c.def("getMaxTOF_ps", &ScatterSpace::getMaxTOF_ps);

	// Compute cylindrical coordinates from two points
	c.def_static(
	    "computeCylindricalCoordinates",
	    [](const Line3D& lor)
	    {
		    float planePos1, angle1, planePos2, angle2;
		    ScatterSpace::computeCylindricalCoordinates(lor, planePos1, angle1,
		                                                planePos2, angle2);

		    // Return as tuple
		    return py::make_tuple(planePos1, angle1, planePos2, angle2);
	    },
	    "Compute cylindrical coordinates from two 3D points");
}
}  // namespace yrt
#endif

namespace yrt
{

ScatterSpace::ScatterSpace(const Scanner& pr_scanner, const std::string& fname)
    : Histogram(pr_scanner), mp_values(nullptr)
{
	readFromFile(fname);

	const auto dims = mp_values->getDims();

	m_numTOFBins = dims[0];
	m_numPlanes = dims[1];
	m_numAngles = dims[2];

	initStepSizes();
}

ScatterSpace::ScatterSpace(const Scanner& pr_scanner, size_t p_numTOFBins,
                           size_t p_numPlanes, size_t p_numAngles)
    : Histogram(pr_scanner),
      m_numTOFBins(p_numTOFBins),
      m_numPlanes(p_numPlanes),
      m_numAngles(p_numAngles)
{
	ASSERT_MSG(m_numTOFBins > 0, "Number of TOF bins must be non-null");
	ASSERT_MSG(m_numPlanes > 0, "Number of planes must be non-null");
	ASSERT_MSG(m_numAngles > 1, "Number of angles must be more than 1");

	initStepSizes();

	// Allocate the scatter space
	mp_values = std::make_unique<Array5D<float>>();
	mp_values->allocate(m_numTOFBins, m_numPlanes, m_numAngles, m_numPlanes,
	                    m_numAngles);
	mp_values->fill(0.0f);
}

void ScatterSpace::readFromFile(const std::string& fname)
{
	if (mp_values == nullptr)
	{
		mp_values = std::make_unique<Array5D<float>>();
	}
	mp_values->readFromFile(fname);
}

void ScatterSpace::writeToFile(const std::string& fname) const
{
	mp_values->writeToFile(fname);
}

ScatterSpace::ScatterSpaceIndex
    ScatterSpace::getNearestNeighborIndex(const ScatterSpacePosition& pos) const
{
	return {getTOFBin(pos.tof_ps), getPlaneIndex(pos.planePosition1),
	        getAngleIndex(pos.angle1), getPlaneIndex(pos.planePosition2),
	        getAngleIndex(pos.angle2)};
}

float ScatterSpace::getNearestNeighborValue(
    const ScatterSpacePosition& pos) const
{
	return getValue(getNearestNeighborIndex(pos));
}

float ScatterSpace::getLinearInterpolationValue(
    const ScatterSpacePosition& pos) const
{
	// TODO NOW: Rename these local variables

	// Clamp and wrap input
	const float clampedTOF = clampTOF(pos.tof_ps);
	const float clampedPlane1 = clampPlanePosition(pos.planePosition1);
	const float wrappedAngle1 = wrapAngle(pos.angle1);
	const float clampedPlane2 = clampPlanePosition(pos.planePosition2);
	const float wrappedAngle2 = wrapAngle(pos.angle2);

	// Get the logical indices but in float
	const float tofIndex_f = clampedTOF / m_TOFBinStep_ps - 0.5f;
	const float planeStart_f = -getAxialFOV() / 2.0f;
	const float plane1Index_f =
	    (clampedPlane1 - planeStart_f) / m_planeStep - 0.5f;
	const float angle1Index_f = wrappedAngle1 / m_angleStep - 0.5f;
	const float plane2Index_f =
	    (clampedPlane2 - planeStart_f) / m_planeStep - 0.5f;
	const float angle2Index_f = wrappedAngle2 / m_angleStep - 0.5f;

	// Logical indices but in signed int
	const int tofIndex = static_cast<int>(std::floor(tofIndex_f));
	const int plane1Index = static_cast<int>(std::floor(plane1Index_f));
	const int angle1Index = static_cast<int>(std::floor(angle1Index_f));
	const int plane2Index = static_cast<int>(std::floor(plane2Index_f));
	const int angle2Index = static_cast<int>(std::floor(angle2Index_f));

	// Fraction within the sample
	float tofFrac = tofIndex_f - tofIndex;
	float plane1Frac = plane1Index_f - plane1Index;
	const float angle1Frac = angle1Index_f - angle1Index;
	float plane2Frac = plane2Index_f - plane2Index;
	const float angle2Frac = angle2Index_f - angle2Index;

	// From the logical indices, gather the neighbor sample (in each dimension)
	const int tofIndex0 =
	    std::clamp(tofIndex, 0, static_cast<int>(m_numTOFBins) - 1);
	const int tofIndex1 =
	    std::clamp(tofIndex + 1, 0, static_cast<int>(m_numTOFBins) - 1);
	const int plane1Index0 =
	    std::clamp(plane1Index, 0, static_cast<int>(m_numPlanes) - 1);
	const int plane1Index1 =
	    std::clamp(plane1Index + 1, 0, static_cast<int>(m_numPlanes) - 1);
	const int plane2Index0 =
	    std::clamp(plane2Index, 0, static_cast<int>(m_numPlanes) - 1);
	const int plane2Index1 =
	    std::clamp(plane2Index + 1, 0, static_cast<int>(m_numPlanes) - 1);

	// Wrap angle indices and get the neighboring sample
	// Edge case:
	//  Might have to do a linear interpolation between the first angle sample
	//  and the last one
	const size_t angle1Index0 = wrapAngleIndex(angle1Index);
	const size_t angle1Index1 = wrapAngleIndex(angle1Index + 1);
	const size_t angle2Index0 = wrapAngleIndex(angle2Index);
	const size_t angle2Index1 = wrapAngleIndex(angle2Index + 1);

	// Edge case for boundaries (if the clamping happened)
	if (tofIndex0 == tofIndex1)
	{
		tofFrac = 0.0f;
	}
	if (plane1Index0 == plane1Index1)
	{
		plane1Frac = 0.0f;
	}
	if (plane2Index0 == plane2Index1)
	{
		plane2Frac = 0.0f;
	}

	// Precompute strides for each dimension
	const size_t tofStride =
	    m_numPlanes * m_numAngles * m_numPlanes * m_numAngles;
	const size_t plane1Stride = m_numAngles * m_numPlanes * m_numAngles;
	const size_t angle1Stride = m_numPlanes * m_numAngles;
	const size_t plane2Stride = m_numAngles;
	const size_t angle2Stride = 1;

	// Precompute weights for all 32 corners using a 5D tensor product
	float weights[32];
	size_t indices[32];

	int idx = 0;
	for (int t = 0; t < 2; ++t)
	{
		const float tofWeight = (t == 0) ? (1.0f - tofFrac) : tofFrac;
		const size_t tofIndex_ull = (t == 0) ? tofIndex0 : tofIndex1;

		for (int p1 = 0; p1 < 2; ++p1)
		{
			const float plane1Weight =
			    (p1 == 0) ? (1.0f - plane1Frac) : plane1Frac;
			const size_t plane1Index_ull =
			    (p1 == 0) ? plane1Index0 : plane1Index1;

			for (int a1 = 0; a1 < 2; ++a1)
			{
				const float angle1Weight =
				    (a1 == 0) ? (1.0f - angle1Frac) : angle1Frac;
				const size_t angle1Index_ull =
				    (a1 == 0) ? angle1Index0 : angle1Index1;

				for (int p2 = 0; p2 < 2; ++p2)
				{
					const float plane2Weight =
					    (p2 == 0) ? (1.0f - plane2Frac) : plane2Frac;
					const size_t plane2Index_ull =
					    (p2 == 0) ? plane2Index0 : plane2Index1;

					for (int a2 = 0; a2 < 2; ++a2)
					{
						const float angle2Weight =
						    (a2 == 0) ? (1.0f - angle2Frac) : angle2Frac;
						const size_t angle2Index_ull =
						    (a2 == 0) ? angle2Index0 : angle2Index1;

						// Compute linear index
						const size_t index = tofIndex_ull * tofStride +
						                     plane1Index_ull * plane1Stride +
						                     angle1Index_ull * angle1Stride +
						                     plane2Index_ull * plane2Stride +
						                     angle2Index_ull * angle2Stride;

						indices[idx] = index;
						weights[idx] = tofWeight * plane1Weight * angle1Weight *
						               plane2Weight * angle2Weight;
						idx++;
					}
				}
			}
		}
	}

	// Get the raw data pointer
	const float* data = mp_values->getRawPointer();

	// Sum weighted contributions
	float result = 0.0f;
	for (int i = 0; i < 32; ++i)
	{
		result += data[indices[i]] * weights[i];
	}

	return result;
}

void ScatterSpace::computeCylindricalCoordinates(const Line3D& lor,
                                                 float& planePosition1,
                                                 float& angle1,
                                                 float& planePosition2,
                                                 float& angle2)
{
	// Plane position is simply the Z coordinates of the LOR
	planePosition1 = lor.point1.z;
	planePosition2 = lor.point2.z;

	// Angle is atan2(y, x) wrapped to [0, 2pi[
	angle1 = std::atan2(lor.point1.y, lor.point1.x);
	angle2 = std::atan2(lor.point2.y, lor.point2.x);
	angle1 = wrapAngle(angle1);
	angle2 = wrapAngle(angle2);
}

float ScatterSpace::getTOF_ps(size_t TOFBin) const
{
	return m_TOFBinStep_ps * (static_cast<float>(TOFBin) + 0.5f);
}

float ScatterSpace::getPlanePosition(size_t planeIndex) const
{
	return getMinSampledPlanePosition() +
	       m_planeStep * static_cast<float>(planeIndex);
}

float ScatterSpace::getAngle(size_t angleIndex) const
{
	return getMinSampledAngle() + m_angleStep * static_cast<float>(angleIndex);
}

size_t ScatterSpace::getTOFBin(float tof_ps) const
{
	// This function assumes that tof_ps is between 0 and getMaxTOF_ps()
	const float clampedTOF_ps = clampTOF(tof_ps);
	const float tofBin_flt = clampedTOF_ps / m_TOFBinStep_ps - 0.5f;
	return static_cast<size_t>(std::round(tofBin_flt));
}

size_t ScatterSpace::getPlaneIndex(float planePosition) const
{
	const float clampedPlanePosition = clampPlanePosition(planePosition);
	const float startingPlane = -getAxialFOV() / 2.0f;
	const float planeIndex_flt =
	    (clampedPlanePosition - startingPlane) / m_planeStep - 0.5f;
	return static_cast<size_t>(std::round(planeIndex_flt));
}

size_t ScatterSpace::getAngleIndex(float angle) const
{
	const float wrappedAngle = wrapAngle(angle);
	const float angleIndex_flt = wrappedAngle / m_angleStep - 0.5f;
	return static_cast<size_t>(std::round(angleIndex_flt));
}

float ScatterSpace::getValue(const ScatterSpaceIndex& idx) const
{
	return getValue(idx.tofBin, idx.planeIndex1, idx.angleIndex1,
	                idx.planeIndex2, idx.angleIndex2);
}

float ScatterSpace::getValue(size_t tofBin, size_t planeIndex1,
                             size_t angleIndex1, size_t planeIndex2,
                             size_t angleIndex2) const
{
	return mp_values->get(
	    {tofBin, planeIndex1, angleIndex1, planeIndex2, angleIndex2});
}

void ScatterSpace::setValue(const ScatterSpaceIndex& idx, float value)
{
	setValue(idx.tofBin, idx.planeIndex1, idx.angleIndex1, idx.planeIndex2,
	         idx.angleIndex2, value);
}

void ScatterSpace::setValue(size_t tofBin, size_t planeIndex1,
                            size_t angleIndex1, size_t planeIndex2,
                            size_t angleIndex2, float value)
{
	mp_values->set({tofBin, planeIndex1, angleIndex1, planeIndex2, angleIndex2},
	               value);
}

void ScatterSpace::symmetrize()
{
	// TODO NOW: Here, make it so that for all elements of this scatter space,
	//  the value at ((a1,z1), (a2,z2), tof) is the same as for
	//  ((a2,z2), (a1,z1), tof) when m_numTOFBins==1
}

float ScatterSpace::clampTOF(float tof_ps) const
{
	const float halfTOFBinStep_ps = getTOFBinStep_ps() / 2;
	const float maxTOFSample = getMaxTOF_ps() - halfTOFBinStep_ps;
	const float minTOFSample = halfTOFBinStep_ps;  // ps
	return std::clamp(tof_ps, minTOFSample, maxTOFSample);
}

float ScatterSpace::clampPlanePosition(float planePosition) const
{
	const float maxPlanePosition = getAxialFOV() / 2 - getPlaneStep() / 2;
	const float minPlanePosition = -maxPlanePosition;
	return std::clamp(planePosition, minPlanePosition, maxPlanePosition);
}

float ScatterSpace::wrapAngle(float angle)
{
	// Wrap to [0, 2pi[
	angle = std::fmod(angle, TWOPI_FLT);
	if (angle < 0)
	{
		angle += TWOPI_FLT;
	}
	return angle;
}

size_t ScatterSpace::wrapAngleIndex(int angleIndex) const
{
	const int numAngles_int = static_cast<int>(m_numAngles);

	// In case we get a negative angleIndex
	if (angleIndex < 0)
	{
		angleIndex += numAngles_int;
	}

	return static_cast<size_t>(angleIndex % numAngles_int);
}

size_t ScatterSpace::getNumTOFBins() const
{
	return m_numTOFBins;
}

size_t ScatterSpace::getNumPlanes() const
{
	return m_numPlanes;
}

size_t ScatterSpace::getNumAngles() const
{
	return m_numAngles;
}

float ScatterSpace::getTOFBinStep_ps() const
{
	return m_TOFBinStep_ps;
}

float ScatterSpace::getPlaneStep() const
{
	return m_planeStep;
}

float ScatterSpace::getAngleStep() const
{
	return m_angleStep;
}

float ScatterSpace::getMinSampledPlanePosition() const
{
	return -getAxialFOV() / 2.0f + getPlaneStep() / 2.0f;
}

float ScatterSpace::getMaxSampledPlanePosition() const
{
	return getAxialFOV() / 2.0f - getPlaneStep() / 2.0f;
}

float ScatterSpace::getMinSampledAngle() const
{
	return 0.0f + m_angleStep / 2.0f;
}

float ScatterSpace::getMaxSampledAngle() const
{
	return TWOPI_FLT - m_angleStep / 2.0f;
}

float ScatterSpace::getAxialFOV() const
{
	return mr_scanner.axialFOV;
}

float ScatterSpace::getRadius() const
{
	return mr_scanner.scannerRadius;
}

float ScatterSpace::getDiameter() const
{
	return getRadius() * 2;
}

float ScatterSpace::getMaxTOF_ps() const
{
	return getDiameter() / SPEED_OF_LIGHT_MM_PS_FLT;
}

size_t ScatterSpace::count() const
{
	return mp_values->getSizeTotal();
}

float ScatterSpace::getProjectionValue(bin_t id) const
{
	return mp_values->getFlat(id);
}

void ScatterSpace::setProjectionValue(bin_t id, float val)
{
	mp_values->setFlat(id, val);
}

void ScatterSpace::clearProjections(float value)
{
	mp_values->fill(value);
}

float ScatterSpace::getProjectionValueFromHistogramBin(
    histo_bin_t histoBinId) const
{
	// TODO NOW:
	//  - Gather the LOR associated with the given detector pair
	//  - Compute the properties of the LOR
	//  - Also Gather the TOF property in case it's available (from histo_bin_t)
	//  - Do linear interpolation and return the value
	return 0.0f;
}

det_id_t ScatterSpace::getDetector1(bin_t /*id*/) const
{
	throw std::runtime_error(FORBID_INPUT_ERROR_MESSAGE);
}

det_id_t ScatterSpace::getDetector2(bin_t /*id*/) const
{
	throw std::runtime_error(FORBID_INPUT_ERROR_MESSAGE);
}

det_pair_t ScatterSpace::getDetectorPair(bin_t /*id*/) const
{
	throw std::runtime_error(FORBID_INPUT_ERROR_MESSAGE);
}

std::unique_ptr<BinIterator> ScatterSpace::getBinIter(int /*numSubsets*/,
                                                      int /*idxSubset*/) const
{
	throw std::runtime_error(FORBID_INPUT_ERROR_MESSAGE);
}

void ScatterSpace::initStepSizes()
{
	m_TOFBinStep_ps = getMaxTOF_ps() / static_cast<float>(m_numTOFBins);
	m_angleStep = TWOPI_FLT / static_cast<float>(m_numAngles);
	m_planeStep = getAxialFOV() / static_cast<float>(m_numPlanes);
}

}  // namespace yrt
