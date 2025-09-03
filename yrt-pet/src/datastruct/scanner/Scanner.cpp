/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/scanner/Scanner.hpp"

#include "yrt-pet/datastruct/scanner/DetRegular.hpp"
#include "yrt-pet/datastruct/scanner/DetectorSetup.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/JSONUtils.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_scanner(pybind11::module& m)
{
	auto c = py::class_<Scanner>(m, "Scanner");
	c.def_property_readonly_static("SCANNER_FILE_VERSION", [](py::object)
	                               { return SCANNER_FILE_VERSION; });

	c.def(py::init<std::string, float, float, float, float, float, size_t,
	               size_t, size_t, size_t, size_t, size_t>(),
	      "scannerName"_a, "axialFOV"_a, "crystalSize_z"_a,
	      "crystalSize_trans"_a, "crystalDepth"_a, "scannerRadius"_a,
	      "detsPerRing"_a, "numRings"_a, "numDOI"_a, "maxRingDiff"_a,
	      "minAngDiff"_a, "detsPerBlock"_a);
	c.def(py::init<const std::string&>());
	c.def("getNumDets", &Scanner::getNumDets);
	c.def("getTheoreticalNumDets", &Scanner::getTheoreticalNumDets);
	c.def("getDetectorPos", &Scanner::getDetectorPos, "det_id"_a);
	c.def("getDetectorOrient", &Scanner::getDetectorOrient, "det_id"_a);
	c.def("isDetectorAllowed", &Scanner::isDetectorAllowed, "det_id"_a);
	c.def_readwrite("scannerName", &Scanner::scannerName);
	c.def_readwrite("axialFOV", &Scanner::axialFOV);
	c.def_readwrite("crystalSize_z", &Scanner::crystalSize_z);
	c.def_readwrite("crystalSize_trans", &Scanner::crystalSize_trans);
	c.def_readwrite("crystalDepth", &Scanner::crystalDepth);
	c.def_readwrite("scannerRadius", &Scanner::scannerRadius);
	c.def_readwrite("collimatorRadius", &Scanner::collimatorRadius);
	c.def_readwrite("fwhm", &Scanner::fwhm);
	c.def_readwrite("energyLLD", &Scanner::energyLLD);
	c.def_readwrite("detsPerRing", &Scanner::detsPerRing);
	c.def_readwrite("fwhm", &Scanner::fwhm);
	c.def_readwrite("numRings", &Scanner::numRings);
	c.def_readwrite("numDOI", &Scanner::numDOI);
	c.def_readwrite("maxRingDiff", &Scanner::maxRingDiff);
	c.def_readwrite("minAngDiff", &Scanner::minAngDiff);
	c.def_readwrite("detsPerBlock", &Scanner::detsPerBlock);
	c.def("createLUT",
	      [](const Scanner& s)
	      {
		      auto lut = std::make_unique<Array2D<float>>();
		      s.createLUT(*lut.get());
		      size_t shape[2]{s.getNumDets(), 6};
		      auto lut_array = py::array_t<float>(
		          shape, std::move(lut.get())->getRawPointer());
		      return lut_array;
	      });
	c.def("readFromFile", &Scanner::readFromFile, "fname"_a);
	c.def("readFromString", &Scanner::readFromString, "json_string"_a);
	c.def(pybind11::pickle([](const Scanner& s) { return s.getScannerPath(); },
	                       [](const std::string& fname)
	                       {
		                       std::unique_ptr<Scanner> s =
		                           std::make_unique<Scanner>(fname);
		                       return s;
	                       }));
	c.def("setDetectorSetup",
	      [](Scanner& s, const std::shared_ptr<DetectorSetup>& detCoords)
	      { s.setDetectorSetup(detCoords); });
	c.def("getDetectorSetup", &Scanner::getDetectorSetup);
}
}  // namespace yrt
#endif

namespace yrt
{
Scanner::Scanner(std::string pr_scannerName, float p_axialFOV,
                 float p_crystalSize_z, float p_crystalSize_trans,
                 float p_crystalDepth, float p_scannerRadius,
                 size_t p_detsPerRing, size_t p_numRings, size_t p_numDOI,
                 size_t p_maxRingDiff, size_t p_minAngDiff,
                 size_t p_detsPerBlock)
    : scannerName(std::move(pr_scannerName)),
      axialFOV(p_axialFOV),
      crystalSize_z(p_crystalSize_z),
      crystalSize_trans(p_crystalSize_trans),
      crystalDepth(p_crystalDepth),
      scannerRadius(p_scannerRadius),
      collimatorRadius(p_scannerRadius - p_crystalDepth),
      fwhm(0.2),
      energyLLD(400),
      detsPerRing(p_detsPerRing),
      numRings(p_numRings),
      numDOI(p_numDOI),
      maxRingDiff(p_maxRingDiff),
      minAngDiff(p_minAngDiff),
      detsPerBlock(p_detsPerBlock)
{
}

Scanner::Scanner(const std::string& p_definitionFile)
{
	readFromFile(p_definitionFile);
}

size_t Scanner::getNumDets() const
{
	return mp_detectors->getNumDets();
}

size_t Scanner::getTheoreticalNumDets() const
{
	return numDOI * numRings * detsPerRing;
}

Vector3D Scanner::getDetectorPos(det_id_t id) const
{
	return mp_detectors->getPos(id);
}

Vector3D Scanner::getDetectorOrient(det_id_t id) const
{
	return mp_detectors->getOrient(id);
}

std::shared_ptr<DetectorSetup> Scanner::getDetectorSetup() const
{
	return mp_detectors;
}

bool Scanner::isValid() const
{
	return mp_detectors != nullptr;
}

bool Scanner::isDetectorAllowed(det_id_t det) const
{
	return mp_detectors->isDetectorAllowed(det);
}

bool Scanner::hasMask() const
{
	return mp_detectors->hasMask();
}

void Scanner::collectConstraints(
    std::vector<std::unique_ptr<Constraint>>& constraints) const
{
	if (hasMask())
	{
		constraints.emplace_back(
		    std::make_unique<ConstraintDetectorMask>(this));
	}
}

void Scanner::createLUT(Array2D<float>& lut) const
{
	lut.allocate(this->getNumDets(), 6);
	for (size_t i = 0; i < this->getNumDets(); i++)
	{
		const Vector3D pos = mp_detectors->getPos(i);
		const Vector3D orient = mp_detectors->getOrient(i);
		lut[i][0] = pos.x;
		lut[i][1] = pos.y;
		lut[i][2] = pos.z;
		lut[i][3] = orient.x;
		lut[i][4] = orient.y;
		lut[i][5] = orient.z;
	}
}

void Scanner::setDetectorSetup(
    const std::shared_ptr<DetectorSetup>& pp_detectors)
{
	if (pp_detectors->getNumDets() != getTheoreticalNumDets())
		throw std::runtime_error(
		    "The number of detectors given by the LUT does not match "
		    "the scanner's characteristics. Namely, num_doi * num_rings * "
		    "dets_per_ring does not equal the size of the LUT");
	mp_detectors = pp_detectors;
}

void Scanner::readFromString(const std::string& fileContents)
{
	json j;
	try
	{
		j = json::parse(fileContents);
	}
	catch (json::exception& e)
	{
		throw std::invalid_argument("Error in Scanner JSON file parsing");
	}

	float scannerFileVersion = 0.0;
	util::getParam<float>(&j, &scannerFileVersion, "VERSION", 0.0, true,
	                      "Missing VERSION in scanner definition file");

	util::getParam<std::string>(
	    &j, &scannerName, "scannerName", "", true,
	    "Missing scanner name in scanner definition file");

	std::string detCoord;
	const bool isDetCoordGiven =
	    util::getParam<std::string>(&j, &detCoord, "detCoord", "", false);

	std::string detMask;
	const bool isDetMaskGiven =
	    util::getParam<std::string>(&j, &detMask, "detMask", "", false);

	util::getParam<float>(
	    &j, &axialFOV, "axialFOV", 0.0, true,
	    "Missing Axial Field Of View value in scanner definition file");
	util::getParam<float>(
	    &j, &crystalSize_trans, "crystalSize_trans", 0.0, true,
	    "Missing transaxial crystal size in scanner definition file");
	util::getParam<float>(
	    &j, &crystalSize_z, "crystalSize_z", 0.0, true,
	    "Missing z-axis crystal size in scanner definition file");

	// Optional values for Scatter estimation only
	util::getParam<float>(&j, &collimatorRadius, "collimatorRadius", -1.0,
	                      false);
	util::getParam<float>(&j, &fwhm, "fwhm", -1.0, false);
	util::getParam<float>(&j, &energyLLD, "energyLLD", -1.0, false);

	util::getParam<float>(&j, &crystalDepth, "crystalDepth", 0.0, true,
	                      "Missing crystal depth in scanner definition file");
	util::getParam<float>(&j, &scannerRadius, "scannerRadius", 0.0, true,
	                      "Missing scanner radius in scanner definition file");
	util::getParam<size_t>(&j, &detsPerRing, {"dets_per_ring", "detsPerRing"},
	                       0, true);
	util::getParam<size_t>(&j, &numRings, {"num_rings", "numRings"}, 0, true);
	util::getParam<size_t>(&j, &numDOI, {"num_doi", "numDOI"}, 0, true);
	util::getParam<size_t>(&j, &maxRingDiff, {"max_ring_diff", "maxRingDiff"},
	                       0, true);
	util::getParam<size_t>(&j, &minAngDiff, {"min_ang_diff", "minAngDiff"}, 0,
	                       true);
	util::getParam<size_t>(&j, &detsPerBlock,
	                       {"dets_per_block", "detsPerBlock"}, 1, false);

	// Check for errors
	if (scannerFileVersion > SCANNER_FILE_VERSION + SMALL_FLT)
	{
		throw std::invalid_argument(
		    "Wrong file version for Scanner JSON file, the "
		    "current version is " +
		    std::to_string(SCANNER_FILE_VERSION));
	}

	// Join paths for DetCoord
	if (isDetCoordGiven)
	{
		const fs::path detCoord_path =
		    m_scannerPath.parent_path() / fs::path(detCoord);
		fs::path detMask_path = "";
		if (isDetMaskGiven)
		{
			detMask_path = m_scannerPath.parent_path() / fs::path(detMask);
		}
		mp_detectors = std::make_shared<DetCoordOwned>(detCoord_path.string(),
		                                               detMask_path.string());
		if (mp_detectors->getNumDets() != getTheoreticalNumDets())
			throw std::runtime_error(
			    "The number of detectors given by the LUT file does not match "
			    "the scanner's characteristics. Namely, (num_doi * num_rings * "
			    "dets_per_ring) does not equal the size of the LUT");
	}
	else
	{
		mp_detectors = std::make_shared<DetRegular>(this);
		reinterpret_cast<DetRegular*>(mp_detectors.get())->generateLUT();
	}

	ASSERT_MSG(
	    maxRingDiff < numRings,
	    "Maximum Ring difference has to be lower than the number of rings");

	ASSERT_MSG(minAngDiff > 0, "Minimum angle difference has to be non-zero");
}

std::string Scanner::getScannerPath() const
{
	return m_scannerPath.string();
}

void Scanner::readFromFile(const std::string& p_definitionFile)
{
	m_scannerPath = fs::path(p_definitionFile);
	if (!(fs::exists(m_scannerPath)))
	{
		throw std::invalid_argument(
		    "The scanner definition file given does not exist");
	}

	std::ifstream i(m_scannerPath.string());
	auto ss = std::ostringstream{};
	ss << i.rdbuf();
	std::string fileContents = ss.str();
	readFromString(fileContents);
}
}  // namespace yrt
