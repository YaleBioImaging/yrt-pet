/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/Sinogram3D.hpp"

#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/utils/Assert.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <variant>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_sinogram3d(pybind11::module& m)
{
	// SinogramParams
	py::class_<SinogramParams>(m, "SinogramParams")
	    .def(py::init<>())
	    .def_readwrite("numPhi", &SinogramParams::numPhi)
	    .def_readwrite("numR", &SinogramParams::numR)
	    .def_readwrite("numTOF", &SinogramParams::numTOF)
	    .def_readwrite("span", &SinogramParams::span)
	    .def_readwrite("maxRingDiff", &SinogramParams::maxRingDiff)
	    .def_readonly("numZ", &SinogramParams::numZ)
	    .def_readonly("numSegSide", &SinogramParams::numSegSide)
	    .def_readonly("phiStep", &SinogramParams::phiStep)
	    .def_readonly("rStep", &SinogramParams::rStep)
	    .def_readonly("rMin", &SinogramParams::rMin)
	    .def("validate", &SinogramParams::validate)
	    .def_static("fromScanner", &SinogramParams::fromScanner,
	                py::arg("scanner"), py::arg("numPhi"), py::arg("numR"),
	                py::arg("span"), py::arg("maxRingDiff") = 0,
	                py::arg("numTOF") = 1)
	    .def_static("fromFile", &SinogramParams::fromFile, py::arg("filename"),
	                py::arg("scanner"));

	// Sinogram3D abstract base
	auto c = py::class_<Sinogram3D, Histogram>(m, "Sinogram3D",
	                                           py::buffer_protocol());
	c.def_readonly("numTOF", &Sinogram3D::numTOF);
	c.def_readonly("numZ", &Sinogram3D::numZ);
	c.def_readonly("numPhi", &Sinogram3D::numPhi);
	c.def_readonly("numR", &Sinogram3D::numR);
	c.def_readonly("sinoSize", &Sinogram3D::sinoSize);
	c.def_buffer(
	    [](Sinogram3D& self) -> py::buffer_info
	    {
		    Array4DBase<float>& d = self.getData();
		    return py::buffer_info(d.getRawPointer(), sizeof(float),
		                           py::format_descriptor<float>::format(), 4,
		                           d.getDims(), d.getStrides());
	    });
	c.def("getShape",
	      [](const Sinogram3D& self) {
		      return py::make_tuple(self.numTOF, self.numZ, self.numPhi,
		                            self.numR);
	      });
	c.def("getData",
	      static_cast<const Array4DBase<float>& (Sinogram3D::*)() const>(
	          &Sinogram3D::getData),
	      py::return_value_policy::reference_internal);
	c.def("getParams", &Sinogram3D::getParams,
	      py::return_value_policy::reference_internal);
	c.def("getBinId", &Sinogram3D::getBinId, py::arg("tof"), py::arg("z"),
	      py::arg("phi"), py::arg("r"));
	c.def(
	    "getCoords",
	    [](const Sinogram3D& self, bin_t id)
	    {
		    size_t tof, z, phi, r;
		    self.getCoords(id, tof, z, phi, r);
		    return py::make_tuple(tof, z, phi, r);
	    },
	    py::arg("bin_id"));
	c.def("getSegment", &Sinogram3D::getSegment, py::arg("iz"));
	c.def("getPlaneInSegment", &Sinogram3D::getPlaneInSegment, py::arg("iz"));
	c.def("getZFromSegmentPlane", &Sinogram3D::getZFromSegmentPlane,
	      py::arg("segment"), py::arg("plane"));
	c.def(
	    "getRingPair",
	    [](const Sinogram3D& self, size_t iZ)
	    {
		    coord_t z1, z2;
		    self.getRingPair(iZ, z1, z2);
		    return py::make_tuple(z1, z2);
	    },
	    py::arg("iz"));
	c.def("getPhi", &Sinogram3D::getPhi, py::arg("iphi"));
	c.def("getR", &Sinogram3D::getR, py::arg("ir"));
	c.def("getZMid", &Sinogram3D::getZMid, py::arg("iz"));
	c.def("getBinIdFromDetPair", &Sinogram3D::getBinIdFromDetPair,
	      py::arg("d1"), py::arg("d2"));
	c.def("writeToFile", &Sinogram3D::writeToFile, py::arg("fname"));

	// Sinogram3DOwned
	py::class_<Sinogram3DOwned, Sinogram3D>(m, "Sinogram3DOwned")
	    .def(py::init<const Scanner&, const SinogramParams&>())
	    .def("allocate", &Sinogram3DOwned::allocate)
	    .def("readFromFile", &Sinogram3DOwned::readFromFile, py::arg("fname"));

	// Sinogram3DAlias
	py::class_<Sinogram3DAlias, Sinogram3D>(m, "Sinogram3DAlias")
	    .def(py::init<const Scanner&, const SinogramParams&>())
	    .def(
	        "bind",
	        [](Sinogram3DAlias& self, py::buffer buf)
	        {
		        // Bind from a numpy array (must be float32, shape
		        // [numTOF,numZ,numPhi,numR])
		        py::buffer_info info = buf.request();
		        if (info.format != py::format_descriptor<float>::format())
			        throw std::invalid_argument("Expected float32 buffer");
		        if (info.ndim != 4)
			        throw std::invalid_argument(
			            "Expected 4-D buffer [numTOF, numZ, numPhi, numR]");
		        // Build a temporary Array4DAlias for the bind call.
		        // bind() copies only the raw pointer and shape, so the
		        // alias does not need to outlive this call.
		        Array4DAlias<float> alias;
		        alias.bind(static_cast<float*>(info.ptr), info.shape[0],
		                   info.shape[1], info.shape[2], info.shape[3]);
		        self.bind(alias);
	        },
	        py::arg("array"));
}
}  // namespace yrt
#endif  // BUILD_PYBIND11

namespace yrt
{

// ---------------------------------------------------------------------------
// SinogramParams
// ---------------------------------------------------------------------------

void SinogramParams::validate(const Scanner& scanner)
{
	ASSERT_MSG(numPhi > 0, "SinogramParams: numPhi must be > 0");
	ASSERT_MSG(numR > 0, "SinogramParams: numR must be > 0");
	ASSERT_MSG(numTOF > 0, "SinogramParams: numTOF must be > 0");
	ASSERT_MSG(span > 0, "SinogramParams: span must be > 0");
	ASSERT_MSG(maxRingDiff > 0,
	           "SinogramParams: maxRingDiff must be > 0 (set to "
	           "scanner.maxRingDiff or a smaller value)");
	ASSERT_MSG(maxRingDiff <= scanner.maxRingDiff,
	           "SinogramParams: maxRingDiff exceeds scanner.maxRingDiff");
	ASSERT_MSG(maxRingDiff < scanner.numRings,
	           "SinogramParams: maxRingDiff must be < numRings");

	numSegSide = maxRingDiff / span + 1;

	// numPlanes(s) = numRings - s*span  for s = 0..numSegSide-1
	numZ = scanner.numRings;
	for (size_t s = 1; s < numSegSide; s++)
	{
		const size_t planes = scanner.numRings - s * span;
		numZ += 2 * planes;  // positive and negative segment
	}

	phiStep = PI_FLT / static_cast<float>(numPhi);
	rMin = -scanner.scannerRadius;
	rStep = 2.0f * scanner.scannerRadius / static_cast<float>(numR);
}

SinogramParams SinogramParams::fromScanner(const Scanner& scanner,
                                           size_t numPhi, size_t numR,
                                           size_t span, size_t maxRingDiff,
                                           size_t numTOF)
{
	SinogramParams p;
	p.numPhi = numPhi;
	p.numR = numR;
	p.numTOF = numTOF;
	p.span = span;
	p.maxRingDiff = (maxRingDiff == 0) ? scanner.maxRingDiff : maxRingDiff;
	p.validate(scanner);
	return p;
}

SinogramParams SinogramParams::fromFile(const std::string& filename,
                                        const Scanner& scanner)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())
	{
		throw std::filesystem::filesystem_error(
		    "Cannot open sinogram file: " + filename,
		    std::make_error_code(std::errc::no_such_file_or_directory));
	}

	int magic = 0;
	int ndims = 0;
	file.read(reinterpret_cast<char*>(&magic), sizeof(int));
	if (magic != Sinogram3D::SINOGRAM_MAGIC)
	{
		throw std::runtime_error("File \"" + filename +
		                         "\" is not a Sinogram3D file "
		                         "(wrong magic number)");
	}
	file.read(reinterpret_cast<char*>(&ndims), sizeof(int));
	if (ndims != 4)
	{
		throw std::runtime_error("File \"" + filename +
		                         "\" has wrong number of dimensions");
	}

	size_t dims[4];
	file.read(reinterpret_cast<char*>(dims), 4 * sizeof(size_t));

	SinogramParams p;
	p.numTOF = dims[0];
	// dims[1] = numZ (derived; we reconstruct it via validate)
	p.numPhi = dims[2];
	p.numR = dims[3];
	file.read(reinterpret_cast<char*>(&p.span), sizeof(size_t));
	file.read(reinterpret_cast<char*>(&p.maxRingDiff), sizeof(size_t));

	p.validate(scanner);

	if (p.numZ != dims[1])
	{
		throw std::runtime_error(
		    "File \"" + filename +
		    "\": stored numZ does not match value computed from "
		    "span/maxRingDiff and scanner.numRings");
	}
	return p;
}

// ---------------------------------------------------------------------------
// Sinogram3D
// ---------------------------------------------------------------------------

Sinogram3D::Sinogram3D(const Scanner& scanner, const SinogramParams& params)
    : Histogram{scanner}, m_params(params), mp_data(nullptr)
{
	numTOF = m_params.numTOF;
	numZ = m_params.numZ;
	numPhi = m_params.numPhi;
	numR = m_params.numR;
	sinoSize = numTOF * numZ * numPhi * numR;
	setupSegments();
}

Sinogram3D::~Sinogram3D() {}

void Sinogram3D::setupSegments()
{
	m_segments.clear();
	m_segments.reserve(2 * m_params.numSegSide - 1);

	size_t cumZ = 0;
	for (size_t s = 0; s < m_params.numSegSide; s++)
	{
		const size_t np = mr_scanner.numRings - s * m_params.span;  // numPlanes

		m_segments.push_back({static_cast<int>(s), cumZ, np});
		cumZ += np;

		if (s > 0)
		{
			m_segments.push_back({-static_cast<int>(s), cumZ, np});
			cumZ += np;
		}
	}
	ASSERT(cumZ == numZ);
}

bin_t Sinogram3D::getBinId(size_t tof, size_t z, size_t phi, size_t r) const
{
	return tof * (numZ * numPhi * numR) + z * (numPhi * numR) + phi * numR + r;
}

void Sinogram3D::getCoords(bin_t id, size_t& tof, size_t& z, size_t& phi,
                           size_t& r) const
{
	const size_t sliceSize = numZ * numPhi * numR;
	tof = id / sliceSize;
	const size_t rem = id % sliceSize;
	z = rem / (numPhi * numR);
	phi = (rem % (numPhi * numR)) / numR;
	r = rem % numR;
}

int Sinogram3D::getSegment(size_t iZ) const
{
	for (const auto& seg : m_segments)
	{
		if (iZ < seg.offset + seg.nPlanes)
			return seg.segment;
	}
	throw std::range_error("Sinogram3D::getSegment: iZ out of range");
}

size_t Sinogram3D::getPlaneInSegment(size_t iZ) const
{
	for (const auto& seg : m_segments)
	{
		if (iZ >= seg.offset && iZ < seg.offset + seg.nPlanes)
			return iZ - seg.offset;
	}
	throw std::range_error("Sinogram3D::getPlaneInSegment: iZ out of range");
}

size_t Sinogram3D::getZFromSegmentPlane(int segment, size_t plane) const
{
	for (const auto& seg : m_segments)
	{
		if (seg.segment == segment)
		{
			ASSERT_MSG(plane < seg.nPlanes,
			           "Sinogram3D::getZFromSegmentPlane: plane out of range");
			return seg.offset + plane;
		}
	}
	throw std::range_error(
	    "Sinogram3D::getZFromSegmentPlane: segment not found");
}

void Sinogram3D::getRingPair(size_t iZ, coord_t& z1, coord_t& z2) const
{
	const int s = getSegment(iZ);
	const size_t p = getPlaneInSegment(iZ);
	const coord_t absDelta = static_cast<coord_t>(std::abs(s)) * m_params.span;

	if (s >= 0)
	{
		z1 = static_cast<coord_t>(p);
		z2 = z1 + absDelta;
	}
	else
	{
		// convention: plane = min(z1,z2), and z1 > z2 for negative segments
		z2 = static_cast<coord_t>(p);
		z1 = z2 + absDelta;
	}
}

float Sinogram3D::getPhi(size_t iPhi) const
{
	return (static_cast<float>(iPhi) + 0.5f) * m_params.phiStep;
}

float Sinogram3D::getR(size_t iR) const
{
	return m_params.rMin + (static_cast<float>(iR) + 0.5f) * m_params.rStep;
}

float Sinogram3D::getZMid(size_t iZ) const
{
	coord_t z1, z2;
	getRingPair(iZ, z1, z2);
	const float ringStep =
	    mr_scanner.axialFOV / static_cast<float>(mr_scanner.numRings);
	const float z1_mm = -mr_scanner.axialFOV * 0.5f +
	                    (static_cast<float>(z1) + 0.5f) * ringStep;
	const float z2_mm = -mr_scanner.axialFOV * 0.5f +
	                    (static_cast<float>(z2) + 0.5f) * ringStep;
	return 0.5f * (z1_mm + z2_mm);
}

// Approximate forward map: (phi_idx, r_idx) + (z1, z2) → detector IDs.
// Uses nominal scanner radius, DOI=0.  Sufficient for scatter/corrections use.
void Sinogram3D::getDetPairApprox(size_t iPhi, size_t iR, coord_t z1,
                                  coord_t z2, det_id_t& d1, det_id_t& d2) const
{
	const float phi = getPhi(iPhi);
	const float r = getR(iR);
	const float R = mr_scanner.scannerRadius;

	// Clamp to avoid NaN from asin
	const float rOverR = std::clamp(r / R, -0.9999f, 0.9999f);
	const float alpha = std::asin(rOverR);

	const float N = static_cast<float>(mr_scanner.detsPerRing);

	// angle1 and angle2 are the transaxial detector angles (in [0, 2π))
	auto wrapAngle = [&](float a) -> float
	{
		a = std::fmod(a, TWOPI_FLT);
		if (a < 0.0f)
			a += TWOPI_FLT;
		return a;
	};

	const float angle1 = wrapAngle(phi + alpha);
	const float angle2 = wrapAngle(phi + PI_FLT - alpha);

	const auto toTrans = [&](float angle) -> det_id_t
	{
		auto t = static_cast<det_id_t>(
		    static_cast<size_t>(std::lroundf(angle * N / TWOPI_FLT)) %
		    static_cast<size_t>(N));
		return t;
	};

	d1 = toTrans(angle1) + z1 * mr_scanner.detsPerRing;
	d2 = toTrans(angle2) + z2 * mr_scanner.detsPerRing;
}

det_id_t Sinogram3D::getDetector1(bin_t id) const
{
	size_t tof, z, phi, r;
	getCoords(id, tof, z, phi, r);
	coord_t z1, z2;
	getRingPair(z, z1, z2);
	det_id_t d1, d2;
	getDetPairApprox(phi, r, z1, z2, d1, d2);
	return d1;
}

det_id_t Sinogram3D::getDetector2(bin_t id) const
{
	size_t tof, z, phi, r;
	getCoords(id, tof, z, phi, r);
	coord_t z1, z2;
	getRingPair(z, z1, z2);
	det_id_t d1, d2;
	getDetPairApprox(phi, r, z1, z2, d1, d2);
	return d2;
}

det_pair_t Sinogram3D::getDetectorPair(bin_t id) const
{
	size_t tof, z, phi, r;
	getCoords(id, tof, z, phi, r);
	coord_t z1, z2;
	getRingPair(z, z1, z2);
	det_id_t d1, d2;
	getDetPairApprox(phi, r, z1, z2, d1, d2);
	return {d1, d2};
}

// Exact reverse map: detector pair → sinogram bin.
// Uses scanner LUT positions and the cross-product formula from
// MakeDownsampledSinogram.cpp.
bin_t Sinogram3D::getBinIdFromDetPair(det_id_t d1, det_id_t d2) const
{
	// Transaxial positions via scanner LUT
	const Vector3D pos1 = mr_scanner.getDetectorPos(d1);
	const Vector3D pos2 = mr_scanner.getDetectorPos(d2);

	float a1 = std::atan2(pos1.y, pos1.x);
	float a2 = std::atan2(pos2.y, pos2.x);
	if (a1 < 0.0f)
		a1 += TWOPI_FLT;
	if (a2 < 0.0f)
		a2 += TWOPI_FLT;

	// Canonical ordering: first angle in [0, π)
	const bool flip = (a1 >= PI_FLT);
	const Vector3D& P1 = flip ? pos2 : pos1;
	const Vector3D& P2 = flip ? pos1 : pos2;
	const float phi = std::fmod(flip ? a2 : a1, PI_FLT);

	// Signed radial offset via 2D cross product
	const float dx = P2.x - P1.x;
	const float dy = P2.y - P1.y;
	const float len = std::sqrt(dx * dx + dy * dy);
	const float rVal = (len > 0.0f) ? (P1.x * P2.y - P1.y * P2.x) / len : 0.0f;

	// Bin indices — clamp to valid range
	const size_t iPhi =
	    static_cast<size_t>(std::clamp(static_cast<int>(phi / m_params.phiStep),
	                                   0, static_cast<int>(numPhi) - 1));
	const size_t iR = static_cast<size_t>(
	    std::clamp(static_cast<int>((rVal - m_params.rMin) / m_params.rStep), 0,
	               static_cast<int>(numR) - 1));

	// Ring indices from detector IDs
	const coord_t ring1 = static_cast<coord_t>((d1 / mr_scanner.detsPerRing) %
	                                           mr_scanner.numRings);
	const coord_t ring2 = static_cast<coord_t>((d2 / mr_scanner.detsPerRing) %
	                                           mr_scanner.numRings);

	const coord_t deltaRing = static_cast<coord_t>(
	    std::abs(static_cast<int>(ring1) - static_cast<int>(ring2)));

	if (deltaRing > m_params.maxRingDiff)
		return sinoSize;  // sentinel: outside sinogram coverage

	const int segIdx = static_cast<int>(deltaRing / m_params.span);
	const bool isNeg = (ring1 > ring2);
	const coord_t plane = std::min(ring1, ring2);

	const size_t iZ = getZFromSegmentPlane(isNeg ? -segIdx : segIdx, plane);
	return getBinId(0, iZ, iPhi, iR);  // TOF bin 0
}

float Sinogram3D::getProjectionValue(bin_t id) const
{
	return mp_data->getFlat(id);
}

void Sinogram3D::setProjectionValue(bin_t id, float v)
{
	mp_data->setFlat(id, v);
}

void Sinogram3D::incrementProjection(bin_t id, float incVal)
{
	mp_data->incrementFlat(id, incVal);
}

size_t Sinogram3D::count() const
{
	return sinoSize;
}

void Sinogram3D::clearProjections(float value)
{
	mp_data->fill(value);
}

std::unique_ptr<BinIterator> Sinogram3D::getBinIter(int numSubsets,
                                                    int idxSubset) const
{
	// Stride through phi bins across all TOF and Z slices, treating
	// [numTOF*numZ] as a single combined outer dimension.
	return std::make_unique<BinIteratorRangeHistogram3D>(
	    numTOF * numZ, numPhi, numR, numSubsets, idxSubset);
}

histo_bin_t Sinogram3D::getHistogramBin(bin_t bin) const
{
	// Return the detector pair so cross-format histograms can look up by pair.
	return getDetectorPair(bin);
}

float Sinogram3D::getProjectionValueFromHistogramBin(histo_bin_t histoBin) const
{
	if (std::holds_alternative<bin_t>(histoBin))
	{
		// Direct bin index — valid when measurements and scatter are both
		// Sinogram3D with matching params.
		return getProjectionValue(std::get<bin_t>(histoBin));
	}

	det_id_t d1, d2;
	if (std::holds_alternative<det_pair_tof_t>(histoBin))
	{
		const auto& p = std::get<det_pair_tof_t>(histoBin);
		d1 = p.d1;
		d2 = p.d2;
	}
	else
	{
		const auto& p = std::get<det_pair_t>(histoBin);
		d1 = p.d1;
		d2 = p.d2;
	}

	if (d1 == 0 && d2 == 0)
		return 0.0f;

	const bin_t binId = getBinIdFromDetPair(d1, d2);
	if (binId >= sinoSize)
		return 0.0f;  // LOR outside sinogram coverage
	return getProjectionValue(binId);
}

bool Sinogram3D::isMemoryValid() const
{
	return mp_data != nullptr && mp_data->isMemoryValid();
}

void Sinogram3D::writeToFile(const std::string& filename) const
{
	ASSERT_MSG(isMemoryValid(), "Sinogram3D: cannot write unallocated data");

	std::ofstream file(filename, std::ios::binary | std::ios::out);
	if (!file.is_open())
	{
		throw std::filesystem::filesystem_error(
		    "Cannot open file for writing: " + filename,
		    std::make_error_code(std::errc::io_error));
	}

	const int magic = SINOGRAM_MAGIC;
	const int ndims = 4;
	const size_t dims[4] = {numTOF, numZ, numPhi, numR};

	file.write(reinterpret_cast<const char*>(&magic), sizeof(int));
	file.write(reinterpret_cast<const char*>(&ndims), sizeof(int));
	file.write(reinterpret_cast<const char*>(dims), 4 * sizeof(size_t));
	file.write(reinterpret_cast<const char*>(&m_params.span), sizeof(size_t));
	file.write(reinterpret_cast<const char*>(&m_params.maxRingDiff),
	           sizeof(size_t));
	file.write(reinterpret_cast<const char*>(mp_data->getRawPointer()),
	           sinoSize * sizeof(float));
}

// ---------------------------------------------------------------------------
// Sinogram3DOwned
// ---------------------------------------------------------------------------

Sinogram3DOwned::Sinogram3DOwned(const Scanner& scanner,
                                 const SinogramParams& params)
    : Sinogram3D(scanner, params)
{
	mp_data = std::make_unique<Array4DOwned<float>>();
}

Sinogram3DOwned::Sinogram3DOwned(const Scanner& scanner,
                                 const SinogramParams& params,
                                 const std::string& filename)
    : Sinogram3DOwned(scanner, params)
{
	readFromFile(filename);
}

void Sinogram3DOwned::allocate()
{
	static_cast<Array4DOwned<float>*>(mp_data.get())
	    ->allocate(numTOF, numZ, numPhi, numR);
}

void Sinogram3DOwned::readFromFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())
	{
		throw std::filesystem::filesystem_error(
		    "Cannot open sinogram file: " + filename,
		    std::make_error_code(std::errc::no_such_file_or_directory));
	}

	// Read and verify header
	int magic = 0;
	int ndims = 0;
	file.read(reinterpret_cast<char*>(&magic), sizeof(int));
	if (magic != SINOGRAM_MAGIC)
	{
		throw std::runtime_error("File \"" + filename +
		                         "\" is not a Sinogram3D file");
	}
	file.read(reinterpret_cast<char*>(&ndims), sizeof(int));
	if (ndims != 4)
	{
		throw std::runtime_error("File \"" + filename +
		                         "\": wrong number of dimensions");
	}

	size_t dims[4];
	file.read(reinterpret_cast<char*>(dims), 4 * sizeof(size_t));

	size_t fileSpan = 0;
	size_t fileMRD = 0;
	file.read(reinterpret_cast<char*>(&fileSpan), sizeof(size_t));
	file.read(reinterpret_cast<char*>(&fileMRD), sizeof(size_t));

	if (dims[0] != numTOF || dims[1] != numZ || dims[2] != numPhi ||
	    dims[3] != numR || fileSpan != m_params.span ||
	    fileMRD != m_params.maxRingDiff)
	{
		throw std::runtime_error(
		    "File \"" + filename +
		    "\": stored dimensions or sinogram params do not match "
		    "the SinogramParams used to open this file");
	}

	// Verify file size
	file.seekg(0, std::ios::end);
	const size_t fileSize = file.tellg();
	constexpr size_t headerSize =
	    sizeof(int) + sizeof(int) + 4 * sizeof(size_t) + 2 * sizeof(size_t);
	const size_t expectedSize = headerSize + sinoSize * sizeof(float);
	if (fileSize != expectedSize)
	{
		throw std::runtime_error("File \"" + filename +
		                         "\": wrong file size (expected " +
		                         std::to_string(expectedSize) + ", got " +
		                         std::to_string(fileSize) + ")");
	}
	file.seekg(static_cast<std::streamoff>(headerSize), std::ios::beg);

	allocate();
	file.read(reinterpret_cast<char*>(mp_data->getRawPointer()),
	          static_cast<std::streamsize>(sinoSize * sizeof(float)));
}

std::unique_ptr<ProjectionData>
    Sinogram3DOwned::create(const Scanner& scanner, const std::string& filename,
                            const io::OptionsResult& /*options*/)
{
	SinogramParams params = SinogramParams::fromFile(filename, scanner);
	return std::make_unique<Sinogram3DOwned>(scanner, params, filename);
}

plugin::OptionsListPerPlugin Sinogram3DOwned::getOptions()
{
	return {};
}

REGISTER_PROJDATA_PLUGIN("SINO", Sinogram3DOwned, Sinogram3DOwned::create,
                         Sinogram3DOwned::getOptions)

// ---------------------------------------------------------------------------
// Sinogram3DAlias
// ---------------------------------------------------------------------------

Sinogram3DAlias::Sinogram3DAlias(const Scanner& scanner,
                                 const SinogramParams& params)
    : Sinogram3D(scanner, params)
{
	mp_data = std::make_unique<Array4DAlias<float>>();
}

void Sinogram3DAlias::bind(Array4DBase<float>& data)
{
	static_cast<Array4DAlias<float>*>(mp_data.get())->bind(data);
}

}  // namespace yrt
