/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/ProjectionData.hpp"
#include "geometry/Constants.hpp"

#include "geometry/Matrix.hpp"
#include "utils/Globals.hpp"

#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void py_setup_projectiondata(py::module& m)
{
	auto c = py::class_<ProjectionData, Variable>(m, "ProjectionData");
	c.def("getScanner", &ProjectionData::getScanner);
	c.def("count", &ProjectionData::count);
	c.def("getProjectionValue", &ProjectionData::getProjectionValue,
	      py::arg("id"));
	c.def("setProjectionValue", &ProjectionData::setProjectionValue,
	      py::arg("id"), py::arg("value"));
	c.def("getTimestamp", &ProjectionData::getTimestamp, py::arg("id"));
	c.def("getFrame", &ProjectionData::getFrame, py::arg("id"));
	c.def("getDetector1", &ProjectionData::getDetector1, py::arg("id"));
	c.def("getDetector2", &ProjectionData::getDetector2, py::arg("id"));
	c.def("getDetectorPair",
	      [](const ProjectionData& self, bin_t ev)
	      {
		      auto [d1, d2] = self.getDetectorPair(ev);
		      return py::make_tuple(d1, d2);
	      });
	c.def("getHistogramBin", &ProjectionData::getHistogramBin, py::arg("id"));
	c.def("getBinIter", &ProjectionData::getBinIter, py::arg("numSubsets"),
	      py::arg("idxSubset"));
	c.def("isUniform", &ProjectionData::isUniform);
	c.def("hasMotion", &ProjectionData::hasMotion);
	c.def("getNumFrames", &ProjectionData::getNumFrames);
	c.def("getTransformOfFrame", &ProjectionData::getTransformOfFrame,
	      py::arg("frame"));
	c.def("getDurationOfFrame", &ProjectionData::getDurationOfFrame,
	      py::arg("frame"));
	c.def("getScanDuration", &ProjectionData::getScanDuration);
	c.def("hasTOF", &ProjectionData::hasTOF);
	c.def("getTOFValue", &ProjectionData::getTOFValue, py::arg("id"));
	c.def("getRandomsEstimate", &ProjectionData::getRandomsEstimate,
	      py::arg("id"));
	c.def("clearProjections", &ProjectionData::clearProjections,
	      py::arg("value"));
	c.def("hasArbitraryLORs", &ProjectionData::hasArbitraryLORs);
	c.def("getArbitraryLOR",
	      [](const ProjectionData& self, bin_t bin)
	      {
		      Line3D l = self.getArbitraryLOR(bin);
		      // Return the raw data
		      return py::make_tuple(l.point1.x, l.point1.y, l.point1.z,
		                            l.point2.x, l.point2.y, l.point2.z);
	      });
	c.def("divideMeasurements", &ProjectionData::divideMeasurements,
	      py::arg("measurements"), py::arg("binIterator"));
}

#endif  // if BUILD_PYBIND11

ProjectionData::ProjectionData(const Scanner& pr_scanner)
    : mr_scanner(pr_scanner)
{
}

void ProjectionData::operationOnEachBin(const std::function<float(bin_t)>& func)
{
	for (bin_t i = 0; i < count(); i++)
	{
		setProjectionValue(i, func(i));
	}
}

void ProjectionData::operationOnEachBinParallel(
    const std::function<float(bin_t)>& func)
{
	bin_t i;
#pragma omp parallel for default(none) private(i), firstprivate(func)
	for (i = 0u; i < count(); i++)
	{
		setProjectionValue(i, func(i));
	}
}

bool ProjectionData::isUniform() const
{
	return false;
}

bool ProjectionData::hasMotion() const
{
	return false;
}

size_t ProjectionData::getNumFrames() const
{
	// By default, only one frame
	return 1ull;
}

transform_t ProjectionData::getTransformOfFrame(frame_t frame) const
{
	(void)frame;
	// Return identity rotation and null translation
	return {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
}

float ProjectionData::getDurationOfFrame(frame_t frame) const
{
	(void)frame;
	throw std::logic_error("getDurationOfFrame unimplemented");
}

timestamp_t ProjectionData::getScanDuration() const
{
	throw std::logic_error("getScanDuration unimplemented");
}

float ProjectionData::getTOFValue(bin_t id) const
{
	(void)id;
	throw std::logic_error("getTOFValue unimplemented");
}

float ProjectionData::getRandomsEstimate(bin_t id) const
{
	(void)id;
	return 0.0f;
}

bool ProjectionData::hasTOF() const
{
	return false;
}

bool ProjectionData::hasArbitraryLORs() const
{
	return false;
}

Line3D ProjectionData::getArbitraryLOR(bin_t id) const
{
	(void)id;
	throw std::logic_error("getArbitraryLOR Unimplemented");
}

ProjectionProperties ProjectionData::getProjectionProperties(bin_t bin) const
{
	auto [d1, d2] = getDetectorPair(bin);

	const Line3D lor = getLOR(bin);

	float tofValue = 0.0f;
	if (hasTOF())
	{
		tofValue = getTOFValue(bin);
	}

	const Vector3D det1Orient = mr_scanner.getDetectorOrient(d1);
	const Vector3D det2Orient = mr_scanner.getDetectorOrient(d2);
	return ProjectionProperties{lor, tofValue, det1Orient, det2Orient};
}

Line3D ProjectionData::getLOR(bin_t bin) const
{
	Line3D lor;

	if (hasArbitraryLORs())
	{
		lor = getArbitraryLOR(bin);
	}
	else
	{
		auto [d1, d2] = getDetectorPair(bin);
		const Vector3D p1 = mr_scanner.getDetectorPos(d1);
		const Vector3D p2 = mr_scanner.getDetectorPos(d2);
		lor = Line3D{p1, p2};
	}

	if (hasMotion())
	{
		const frame_t frame = getFrame(bin);
		const transform_t transfo = getTransformOfFrame(frame);

		const Matrix MRot{transfo.r00, transfo.r01, transfo.r02,
		                  transfo.r10, transfo.r11, transfo.r12,
		                  transfo.r20, transfo.r21, transfo.r22};
		const Vector3D VTrans{transfo.tx, transfo.ty, transfo.tz};

		Vector3D point1Prim = MRot * lor.point1;
		point1Prim = point1Prim + VTrans;

		Vector3D point2Prim = MRot * lor.point2;
		point2Prim = point2Prim + VTrans;

		lor.update(point1Prim, point2Prim);
	}
	return lor;
}

timestamp_t ProjectionData::getTimestamp(bin_t id) const
{
	(void)id;
	return 0u;
}

frame_t ProjectionData::getFrame(bin_t id) const
{
	(void)id;
	return 0u;
}

const Scanner& ProjectionData::getScanner() const
{
	return mr_scanner;
}

// Note: If a bin is disabled, this should return {0, 0}
det_pair_t ProjectionData::getDetectorPair(bin_t id) const
{
	return {getDetector1(id), getDetector2(id)};
}

histo_bin_t ProjectionData::getHistogramBin(bin_t bin) const
{
	return getDetectorPair(bin);
}

void ProjectionData::clearProjections(float value)
{
	(void)value;
	throw std::logic_error("clearProjections undefined on this object");
}

void ProjectionData::divideMeasurements(const ProjectionData* measurements,
                                        const BinIterator* binIter)
{
	const bin_t numBins = binIter->size();
#pragma omp parallel for default(none) \
    firstprivate(numBins, binIter, measurements)
	for (bin_t binIdx = 0; binIdx < numBins; binIdx++)
	{
		const bin_t bin = binIter->get(binIdx);
		const float projValue = getProjectionValue(bin);
		// to prevent numerical instability
		if (projValue > EPS_FLT)
		{
			setProjectionValue(bin, measurements->getProjectionValue(bin) /
			                            projValue);
		}
	}
}
