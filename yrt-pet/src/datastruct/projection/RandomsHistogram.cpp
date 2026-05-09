/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/RandomsHistogram.hpp"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/utils/Assert.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

using namespace pybind11::literals;
namespace py = pybind11;

namespace yrt
{
void py_setup_randomsHistogram(py::module& m)
{
	auto c = py::class_<RandomsHistogram, Histogram>(m, "RandomsHistogram");
	c.def(py::init<const Scanner&, float>(), "scanner"_a, "tau"_a = 0.f);
	c.def(py::init<const Scanner&, const std::string&, float>(), "scanner"_a,
	      "filename"_a, "tau"_a = 0.f);
	c.def("populateFromListMode", &RandomsHistogram::populateFromListMode,
	      "list_mode"_a);
	c.def("readFromFile", &RandomsHistogram::readFromFile, "filename"_a);
	c.def("writeToFile", &RandomsHistogram::writeToFile, "filename"_a);
}
}  // namespace yrt

#endif

namespace yrt
{

RandomsHistogram::RandomsHistogram(const Scanner& pr_scanner, float p_tau)
    : Histogram(pr_scanner),
      mp_singles(std::make_unique<Array1DOwned<float>>()),
      m_timeWindow(p_tau)
{
}

RandomsHistogram::RandomsHistogram(const Scanner& pr_scanner,
                                   const std::string& filename, float p_tau)
    : RandomsHistogram(pr_scanner, p_tau)
{
	readFromFile(filename);
}

void RandomsHistogram::populateFromListMode(const ListMode& listMode)
{
	const size_t numDetectors = mr_scanner.getNumDets();
	mp_singles->allocate(numDetectors);
	for (det_id_t det = 0; det < numDetectors; det++)
	{
		(*mp_singles)[det] = listMode.getSinglesRate(det);
	}
}

float RandomsHistogram::getProjectionValueFromHistogramBin(
    histo_bin_t histoBinId) const
{
	det_id_t d1, d2;
	if (std::holds_alternative<det_pair_t>(histoBinId))
	{
		const auto& detPair = std::get<det_pair_t>(histoBinId);
		d1 = detPair.d1;
		d2 = detPair.d2;
	}
	else if (std::holds_alternative<det_pair_tof_t>(histoBinId))
	{
		const auto& detPair = std::get<det_pair_tof_t>(histoBinId);
		d1 = detPair.d1;
		d2 = detPair.d2;
	}
	else
	{
		ASSERT_MSG(false,
		           "RandomsHistogram::getProjectionValueFromHistogramBin "
		           "only supports det_pair_t and det_pair_tof_t variants");
		return 0.0f;
	}

	ASSERT_MSG(mp_singles != nullptr && mp_singles->isMemoryValid(),
	           "RandomsHistogram singles array is not allocated");
	return (*mp_singles)[d1] * (*mp_singles)[d2] * (2.0f * m_timeWindow);
}

size_t RandomsHistogram::count() const
{
	return 0;
}

float RandomsHistogram::getProjectionValue(bin_t /*id*/) const
{
	ASSERT_MSG(false, "RandomsHistogram does not support getProjectionValue; "
	                  "use getProjectionValueFromHistogramBin instead");
	return 0.0f;
}

void RandomsHistogram::setProjectionValue(bin_t /*id*/, float /*val*/)
{
	ASSERT_MSG(false, "RandomsHistogram does not support setProjectionValue");
}

det_id_t RandomsHistogram::getDetector1(bin_t /*id*/) const
{
	ASSERT_MSG(false, "RandomsHistogram does not support getDetector1");
	return 0;
}

det_id_t RandomsHistogram::getDetector2(bin_t /*id*/) const
{
	ASSERT_MSG(false, "RandomsHistogram does not support getDetector2");
	return 0;
}

std::unique_ptr<BinIterator>
    RandomsHistogram::getBinIter(int /*numSubsets*/, int /*idxSubset*/) const
{
	ASSERT_MSG(false, "RandomsHistogram does not support bin iteration");
	return nullptr;
}

void RandomsHistogram::writeToFile(const std::string& filename) const
{
	ASSERT_MSG(mp_singles != nullptr && mp_singles->isMemoryValid(),
	           "RandomsHistogram singles array is not allocated");
	mp_singles->writeToFile(filename);
}

void RandomsHistogram::readFromFile(const std::string& filename)
{
	mp_singles->readFromFile(filename);
}

std::unique_ptr<ProjectionData>
    RandomsHistogram::create(const Scanner& scanner,
                             const std::string& filename,
                             const io::OptionsResult& /*options*/)
{
	return std::make_unique<RandomsHistogram>(scanner, filename);
}

plugin::OptionsListPerPlugin RandomsHistogram::getOptions()
{
	return {};
}

REGISTER_PROJDATA_PLUGIN("RH", RandomsHistogram, RandomsHistogram::create,
                         RandomsHistogram::getOptions)
}  // namespace yrt
