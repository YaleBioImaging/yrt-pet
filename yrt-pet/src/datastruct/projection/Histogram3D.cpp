/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/utils/Assert.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_histogram3d(pybind11::module& m)
{
	auto c = py::class_<Histogram3D, Histogram>(m, "Histogram3D",
	                                            py::buffer_protocol());
	c.def_readonly("numZBin", &Histogram3D::numZBin);
	c.def_readonly("numPhi", &Histogram3D::numPhi);
	c.def_readonly("numR", &Histogram3D::numR);
	c.def_readonly("histoSize", &Histogram3D::histoSize);
	c.def_buffer(
	    [](Histogram3D& self) -> py::buffer_info
	    {
		    Array3DBase<float>& d = self.getData();
		    return py::buffer_info(d.getRawPointer(), sizeof(float),
		                           py::format_descriptor<float>::format(), 3,
		                           d.getDims(), d.getStrides());
	    });
	c.def("writeToFile", &Histogram3D::writeToFile, py::arg("fname"));
	c.def("getShape", [](const Histogram3D& self)
	      { return py::make_tuple(self.numZBin, self.numPhi, self.numR); });
	c.def("getData",
	      static_cast<const Array3DBase<float>& (Histogram3D::*)() const>(
	          &Histogram3D::getData),
	      py::return_value_policy::reference_internal);
	c.def("getBinIdFromCoords", &Histogram3D::getBinIdFromCoords, py::arg("r"),
	      py::arg("phi"), py::arg("z_bin"));
	c.def(
	    "getCoordsFromBinId",
	    [](const Histogram3D& self, size_t binId)
	    {
		    if (binId >= self.histoSize)
			    throw std::invalid_argument(
			        "The binId provided exceeds the size "
			        "permitted by the histogram");
		    coord_t r, phi, z_bin;
		    self.getCoordsFromBinId(binId, r, phi, z_bin);
		    return py::make_tuple(r, phi, z_bin);
	    },
	    py::arg("bin_id"));
	c.def(
	    "getDetPairFromCoords",
	    [](const Histogram3D& self, coord_t r, coord_t phi, coord_t z_bin)
	    {
		    if (r >= self.numR)
		    {
			    throw std::invalid_argument("The r coordinate given does not "
			                                "respect the histogram shape");
		    }
		    if (phi >= self.numPhi)
		    {
			    throw std::invalid_argument("The phi coordinate given does not "
			                                "respect the histogram shape");
		    }
		    if (z_bin >= self.numZBin)
		    {
			    throw std::invalid_argument("The z_bin coordinate given does "
			                                "not respect the histogram shape");
		    }
		    det_id_t d1, d2;
		    self.getDetPairFromCoords(r, phi, z_bin, d1, d2);
		    return py::make_tuple(d1, d2);
	    },
	    py::arg("r"), py::arg("phi"), py::arg("z_bin"));
	c.def(
	    "getDetPairFromBinId",
	    [](const Histogram3D& self, bin_t binId)
	    {
		    if (binId >= self.histoSize)
			    throw std::invalid_argument(
			        "The binId provided exceeds the size "
			        "permitted by the histogram");
		    auto [d1, d2] = self.getDetPairFromBinId(binId);
		    return py::make_tuple(d1, d2);
	    },
	    py::arg("bin_id"));
	c.def(
	    "getCoordsFromDetPair",
	    [](const Histogram3D& self, det_id_t d1, det_id_t d2)
	    {
		    coord_t r, phi, z_bin;
		    self.getCoordsFromDetPair(d1, d2, r, phi, z_bin);
		    return py::make_tuple(r, phi, z_bin);
	    },
	    py::arg("d1"), py::arg("d2"));
	c.def("getBinIdFromDetPair", &Histogram3D::getBinIdFromDetPair,
	      py::arg("d1"), py::arg("d2"));
	c.def("incrementProjection", &Histogram3D::incrementProjection,
	      py::arg("bin_id"), py::arg("inc_val"));
	c.def(
	    "getZ1Z2",
	    [](const Histogram3D& self, coord_t z_bin)
	    {
		    if (z_bin >= self.numZBin)
			    throw std::invalid_argument(
			        "The binId provided exceeds the size "
			        "permitted by the histogram");
		    coord_t z1, z2;
		    self.getZ1Z2(z_bin, z1, z2);
		    return py::make_tuple(z1, z2);
	    },
	    py::arg("z_bin"));

	auto c_owned =
	    py::class_<Histogram3DOwned, Histogram3D>(m, "Histogram3DOwned");
	c_owned.def(py::init<const Scanner&>(), py::arg("scanner"));
	c_owned.def(py::init<const Scanner&, std::string>(), py::arg("scanner"),
	            py::arg("fname"));
	c_owned.def("readFromFile", &Histogram3DOwned::readFromFile,
	            py::arg("fname"));
	c_owned.def("allocate", &Histogram3DOwned::allocate);

	auto c_alias =
	    py::class_<Histogram3DAlias, Histogram3D>(m, "Histogram3DAlias");
	c_alias.def(py::init<const Scanner&>(), py::arg("scanner"));
	c_alias.def("bind", &Histogram3DAlias::bind, py::arg("array3dfloat"));
	c_alias.def(
	    "bind",
	    [](Histogram3DAlias& self, py::buffer& np_data)
	    {
		    py::buffer_info buffer = np_data.request();
		    if (buffer.ndim != 3)
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have 3 dimensions");
		    }
		    if (buffer.format != py::format_descriptor<float>::format())
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have a float32 format");
		    }
		    std::vector<size_t> dims = {self.numZBin, self.numPhi, self.numR};
		    for (int i = 0; i < 3; i++)
		    {
			    if (buffer.shape[i] != static_cast<int>(dims[i]))
			    {
				    throw std::invalid_argument(
				        "The buffer shape does not match with the image "
				        "parameters");
			    }
		    }
		    static_cast<Array3DAlias<float>&>(self.getData())
		        .bind(reinterpret_cast<float*>(buffer.ptr), dims[0], dims[1],
		              dims[2]);
	    });
}
}  // namespace yrt
#endif

namespace yrt
{
Histogram3D::Histogram3D(const Scanner& pr_scanner)
    : Histogram{pr_scanner}, mp_data(nullptr)
{
	ASSERT_MSG(mr_scanner.detsPerRing % 2 == 0,
	           "Limitation: The number of detectors per ring has to be even "
	           "for the histogram to be properly defined");

	// Double checks (These checks are also done on Scanner.cpp)
	ASSERT_MSG(
	    mr_scanner.maxRingDiff < mr_scanner.numRings,
	    "Maximum Ring difference has to be lower than the number of rings");
	ASSERT_MSG(mr_scanner.minAngDiff > 0,
	           "Minimum angle difference cannot be zero");

	m_rCut = mr_scanner.minAngDiff / 2;
	if (mr_scanner.minAngDiff % 2 != 0 && mr_scanner.detsPerRing % 4 == 0)
	{
		m_rCut++;
	}
	m_numDOIPoss = mr_scanner.numDOI * mr_scanner.numDOI;

	numR =
	    m_numDOIPoss * (mr_scanner.detsPerRing / 2 + 1 - mr_scanner.minAngDiff);

	numPhi = mr_scanner.detsPerRing;

	size_t dz_max = mr_scanner.maxRingDiff;
	numZBin = (dz_max + 1) * mr_scanner.numRings - (dz_max * (dz_max + 1)) / 2;
	// Number of z_bins that have z1 < z2
	m_numZBinDiff = numZBin - mr_scanner.numRings;
	// Other side for if z1 > z2
	numZBin = numZBin + m_numZBinDiff;
	setupHistogram();
	histoSize = numZBin * numPhi * numR;
}

Histogram3D::~Histogram3D() {}

Histogram3DOwned::Histogram3DOwned(const Scanner& pr_scanner)
    : Histogram3D(pr_scanner)
{
	mp_data = std::make_unique<Array3D<float>>();
}

Histogram3DOwned::Histogram3DOwned(const Scanner& pr_scanner,
                                   const std::string& filename)
    : Histogram3DOwned(pr_scanner)
{
	readFromFile(filename);
}

Histogram3DAlias::Histogram3DAlias(const Scanner& pr_scanner)
    : Histogram3D(pr_scanner)
{
	mp_data = std::make_unique<Array3DAlias<float>>();
}

void Histogram3DOwned::readFromFile(const std::string& filename)
{
	std::array<size_t, 3> dims{numZBin, numPhi, numR};
	try
	{
		mp_data->readFromFile(filename, dims);
	}
	catch (const std::exception& e)
	{
		throw std::runtime_error(
		    "Error during Histogram initialization either the scanner\'s "
		    "attributes do not match the histogram given, the file given is "
		    "inexistant or the file given is not a valid histogram file");
	}
}

void Histogram3DAlias::bind(Array3DBase<float>& pr_data)
{
	static_cast<Array3DAlias<float>*>(mp_data.get())->bind(pr_data);
	if (mp_data->getRawPointer() != pr_data.getRawPointer())
	{
		throw std::runtime_error(
		    "Error occured in the binding of the given array");
	}
}

void Histogram3DOwned::allocate()
{
	static_cast<Array3D<float>*>(mp_data.get())
	    ->allocate(numZBin, numPhi, numR);
}

void Histogram3D::writeToFile(const std::string& filename) const
{
	mp_data->writeToFile(filename);
}

bin_t Histogram3D::getBinIdFromCoords(coord_t r, coord_t phi,
                                      coord_t z_bin) const
{
	return z_bin * numPhi * numR + phi * numR + r;
}

void Histogram3D::getCoordsFromBinId(bin_t binId, coord_t& r, coord_t& phi,
                                     coord_t& z_bin) const
{
	z_bin = binId / (numPhi * numR);
	phi = (binId % (numPhi * numR)) / numR;
	r = (binId % (numPhi * numR)) % numR;
}

size_t Histogram3D::count() const
{
	return histoSize;
}

void Histogram3D::setupHistogram()
{
	for (coord_t phi = 0; phi < numPhi; phi++)
	{
		for (coord_t r_ring = 0; r_ring < (numR / m_numDOIPoss); r_ring++)
		{
			det_id_t d1_ring, d2_ring;
			getDetPairInSameRing(r_ring, phi, d1_ring, d2_ring);
			m_ringMap[{d1_ring, d2_ring}] = {r_ring, phi};
		}
	}
}

void Histogram3D::getDetPairFromCoords(coord_t r, coord_t phi, coord_t z_bin,
                                       det_id_t& d1, det_id_t& d2) const
{
	// Bound checks for the input r, phi, z_bin
	if (r >= numR || phi >= numPhi || z_bin >= numZBin)
	{
		throw std::range_error(
		    "The requested histogram coordinates [" + std::to_string(z_bin) +
		    ", " + std::to_string(phi) + ", " + std::to_string(r) +
		    "] are outside the histogram array (size:[" +
		    std::to_string(numZBin) + ", " + std::to_string(numPhi) + ", " +
		    std::to_string(numR) + "])");
	}

	coord_t r_ring = r / m_numDOIPoss;
	coord_t d1_ring, d2_ring;
	getDetPairInSameRing(r_ring, phi, d1_ring, d2_ring);

	// Specific case when minAngDiff == 1 on some bins
	if (d1_ring == d2_ring)
	{
		ASSERT(mr_scanner.minAngDiff == 1);  // Sanity check
		d1 = 0;
		d2 = 0;
		return;
	}

	coord_t doi_case = r % m_numDOIPoss;
	coord_t doi_d1 = doi_case % mr_scanner.numDOI;
	coord_t doi_d2 = doi_case / mr_scanner.numDOI;

	// Determine delta Z and Z1
	coord_t z1, z2;
	if (z_bin < mr_scanner.numRings)
	{
		z1 = z2 = z_bin;
	}
	else
	{
		// Regardless of if z1<z2 or z1>z2
		int current_z_bin =
		    (((int)z_bin) - mr_scanner.numRings) % m_numZBinDiff +
		    mr_scanner.numRings;
		int current_n_planes = mr_scanner.numRings;
		size_t delta_z = 0;
		while (current_z_bin - current_n_planes >= 0)
		{
			current_z_bin -= current_n_planes;
			current_n_planes -= 1;
			delta_z++;
		}
		z1 = current_z_bin;
		z2 = z1 + delta_z;
		// Check if i had to switch (z1>z2)
		if (z_bin - mr_scanner.numRings >= m_numZBinDiff)
		{
			std::swap(z1, z2);
		}
	}

	d1 = d1_ring + z1 * mr_scanner.detsPerRing +
	     doi_d1 * (mr_scanner.detsPerRing * mr_scanner.numRings);
	d2 = d2_ring + z2 * mr_scanner.detsPerRing +
	     doi_d2 * (mr_scanner.detsPerRing * mr_scanner.numRings);
}

// Transpose
void Histogram3D::getCoordsFromDetPair(det_id_t d1, det_id_t d2, coord_t& r,
                                       coord_t& phi, coord_t& z_bin) const
{
	coord_t r_ring;
	if (d1 > d2)
		std::swap(d1, d2);
	det_id_t d1_ring = d1 % (mr_scanner.detsPerRing);
	det_id_t d2_ring = d2 % (mr_scanner.detsPerRing);
	if (d1_ring > d2_ring)
	{
		std::swap(d1, d2);
		std::swap(d1_ring, d2_ring);
	}

	getCoordsInSameRing(d1_ring, d2_ring, r_ring, phi);
	det_id_t doi_d1 = d1 / (mr_scanner.numRings * mr_scanner.detsPerRing);
	det_id_t doi_d2 = d2 / (mr_scanner.numRings * mr_scanner.detsPerRing);
	if (d1_ring > d2_ring)
		std::swap(doi_d1, doi_d2);
	r = r_ring * m_numDOIPoss + (doi_d1 + doi_d2 * mr_scanner.numDOI);

	const int z1 = (d1 / (mr_scanner.detsPerRing)) % (mr_scanner.numRings);
	const int z2 = (d2 / (mr_scanner.detsPerRing)) % (mr_scanner.numRings);

	coord_t delta_z = static_cast<coord_t>(std::abs(z2 - z1));
	if (delta_z > mr_scanner.maxRingDiff)
	{
		throw std::range_error("The detector pair given does not respect the "
		                       "maximum ring difference rule");
	}
	coord_t num_removed_z_bins = delta_z * (delta_z - 1) / 2;
	z_bin =
	    delta_z * mr_scanner.numRings + std::min(z1, z2) - num_removed_z_bins;
	if (delta_z > 0 && z1 > z2)
	{
		z_bin += m_numZBinDiff;  // switch
	}
}

det_pair_t Histogram3D::getDetPairFromBinId(bin_t binId) const
{
	coord_t r, phi, z_bin;
	det_id_t d1, d2;
	getCoordsFromBinId(binId, r, phi, z_bin);
	getDetPairFromCoords(r, phi, z_bin, d1, d2);
	return {d1, d2};
}

bin_t Histogram3D::getBinIdFromDetPair(det_id_t d1, det_id_t d2) const
{
	coord_t r, phi, z_bin;
	getCoordsFromDetPair(d1, d2, r, phi, z_bin);
	return getBinIdFromCoords(r, phi, z_bin);
}

histo_bin_t Histogram3D::getHistogramBin(bin_t bin) const
{
	return bin;
}

void Histogram3D::getDetPairInSameRing(coord_t r_ring, coord_t phi,
                                       det_id_t& d1_ring,
                                       det_id_t& d2_ring) const
{
	int n_tot_ring = mr_scanner.detsPerRing;
	int r = r_ring;  // for cleanness
	int d01 = 0;
	int d02 = n_tot_ring / 2;
	if (phi % 2 != 0)
		d02 = n_tot_ring / 2 + 1;
	int dr1 = d01 + (r - n_tot_ring / 4 + m_rCut);
	int dr2 = d02 - (r - n_tot_ring / 4 + m_rCut);
	if (dr1 < 0)
		dr1 += n_tot_ring;
	if (dr2 < 0)
		dr2 += n_tot_ring;
	int d1_ring_i = (dr1 + phi / 2) % n_tot_ring;
	int d2_ring_i = (dr2 + phi / 2) % n_tot_ring;
	d1_ring = static_cast<det_id_t>(d1_ring_i);
	d2_ring = static_cast<det_id_t>(d2_ring_i);
	if (d1_ring > d2_ring)
		std::swap(d1_ring, d2_ring);
}

bool Histogram3D::getCoordsInSameRing_safe(det_id_t d1_ring, det_id_t d2_ring,
                                           coord_t& r_ring, coord_t& phi) const
{
	det_id_t d1_ring_c = d1_ring;
	det_id_t d2_ring_c = d2_ring;
	if (d1_ring_c > d2_ring_c)
		std::swap(d1_ring_c, d2_ring_c);

	auto got = m_ringMap.find({d1_ring_c, d2_ring_c});
	if (got == m_ringMap.end())
	{
		std::cerr << "Detector ring Exception found at " << d1_ring_c << ", "
		          << d2_ring_c << std::endl;
		return false;
	}
	DetRingCoordinates coords = got->second;
	r_ring = coords[0];
	phi = coords[1];
	return true;
}

void Histogram3D::getCoordsInSameRing(det_id_t d1_ring, det_id_t d2_ring,
                                      coord_t& r_ring, coord_t& phi) const
{
	// Note: ".at(...)" is the "const" version of "[...]"
	det_id_t d1_ring_c = d1_ring;
	det_id_t d2_ring_c = d2_ring;
	if (d1_ring_c > d2_ring_c)
		std::swap(d1_ring_c, d2_ring_c);
	try
	{
		const DetRingCoordinates coords = m_ringMap.at({d1_ring_c, d2_ring_c});
		r_ring = coords[0];
		phi = coords[1];
	}
	catch (const std::out_of_range& e)
	{
		std::cerr << "Caught: " << e.what() << std::endl;
		throw std::range_error(
		    "A Line-Of-Response might not respect the Minimum-Angle-Difference "
		    "restriction of the scanner.\nCheck the scanner properties or the "
		    "input files.");
	}
}

float Histogram3D::getProjectionValue(bin_t binId) const
{
	return mp_data->getFlat(binId);
}

void Histogram3D::setProjectionValue(bin_t binId, float val)
{
	mp_data->setFlat(binId, val);
}

void Histogram3D::incrementProjection(bin_t binId, float val)
{
	mp_data->incrementFlat(binId, val);
}

det_id_t Histogram3D::getDetector1(bin_t evId) const
{
	auto [d1, _] = getDetPairFromBinId(evId);
	return d1;
}

det_id_t Histogram3D::getDetector2(bin_t evId) const
{
	auto [_, d2] = getDetPairFromBinId(evId);
	return d2;
}

det_pair_t Histogram3D::getDetectorPair(bin_t id) const
{
	return getDetPairFromBinId(id);
}

void Histogram3D::getZ1Z2(coord_t z_bin, coord_t& z1, coord_t& z2) const
{
	if (z_bin < mr_scanner.numRings)
	{
		z1 = z2 = z_bin;
	}
	else
	{
		// Regardless of if z1<z2 or z1>z2
		int current_z_bin =
		    (((int)z_bin) - mr_scanner.numRings) % m_numZBinDiff +
		    mr_scanner.numRings;
		int current_n_planes = mr_scanner.numRings;
		size_t delta_z = 0;
		while (current_z_bin - current_n_planes >= 0)
		{
			current_z_bin -= current_n_planes;
			current_n_planes -= 1;
			delta_z++;
		}
		z1 = current_z_bin;
		z2 = z1 + delta_z;
		// Check if i had to switch (z1>z2)
		if (z_bin - mr_scanner.numRings >= m_numZBinDiff)
		{
			std::swap(z1, z2);
		}
	}
}

bool Histogram3D::isMemoryValid() const
{
	return mp_data != nullptr && mp_data->getRawPointer() != nullptr;
}

void Histogram3D::clearProjections()
{
	clearProjections(0.0f);
}

void Histogram3D::clearProjections(float value)
{
	mp_data->fill(value);
}

std::unique_ptr<BinIterator> Histogram3D::getBinIter(int numSubsets,
                                                     int idxSubset) const
{
	if (idxSubset < 0 || numSubsets <= 0)
		throw std::invalid_argument(
		    "The subset index cannot be negative, the number of subsets cannot "
		    "be less or equal than zero");
	if (idxSubset >= numSubsets)
		throw std::invalid_argument(
		    "The subset index has to be smaller than the number of subsets");
	return std::make_unique<BinIteratorRangeHistogram3D>(numZBin, numPhi, numR,
	                                                     numSubsets, idxSubset);
}

float Histogram3D::getProjectionValueFromHistogramBin(
    histo_bin_t histoBinId) const
{
	if (std::holds_alternative<bin_t>(histoBinId))
	{
		// Use bin itself
		return getProjectionValue(std::get<bin_t>(histoBinId));
	}

	// use the detector pair
	const auto [d1, d2] = std::get<det_pair_t>(histoBinId);
	const bin_t binId = getBinIdFromDetPair(d1, d2);
	return getProjectionValue(binId);
}

std::unique_ptr<ProjectionData>
    Histogram3DOwned::create(const Scanner& scanner,
                             const std::string& filename,
                             const io::OptionsResult& options)
{
	(void)options;  // No use for extra options
	return std::make_unique<Histogram3DOwned>(scanner, filename);
}

plugin::OptionsListPerPlugin Histogram3DOwned::getOptions()
{
	// No extra options
	return {};
}

REGISTER_PROJDATA_PLUGIN("H", Histogram3DOwned, Histogram3DOwned::create,
                         Histogram3DOwned::getOptions)

}  // namespace yrt
