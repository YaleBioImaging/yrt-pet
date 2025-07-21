/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/Crystal.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Utilities.hpp"

#include <stdexcept>
#include <string>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_crystal(py::module& m)
{
	py::enum_<scatter::CrystalMaterial>(m, "CrystalMaterial")
	    .value("LSO", scatter::CrystalMaterial::LSO)
	    .value("LYSO", scatter::CrystalMaterial::LYSO)
	    .export_values();
	m.def("getMuDet", &scatter::getMuDet);
	m.def("getCrystalMaterialFromName", &scatter::getCrystalMaterialFromName);
	m.def("getCrystal", &scatter::getCrystalMaterialFromName);  // alias
}
}  // namespace yrt

#endif

namespace yrt
{
namespace scatter
{

double getMuDet(double energy, CrystalMaterial crystalMat)
{
	const int e = static_cast<int>(energy) - 1;
	ASSERT(e >= 0 && e < 1000);
	if (crystalMat == CrystalMaterial::LSO)
	{
		return MuLSO[e];
	}
	return MuLYSO[e];
}

CrystalMaterial
    getCrystalMaterialFromName(const std::string& crystalMaterial_name)
{
	const std::string crystalMaterial_uppercaseName =
	    util::toUpper(crystalMaterial_name);

	CrystalMaterial crystalMaterial;
	if (crystalMaterial_uppercaseName == "LYSO")
	{
		crystalMaterial = CrystalMaterial::LYSO;
	}
	else if (crystalMaterial_uppercaseName == "LSO")
	{
		crystalMaterial = CrystalMaterial::LSO;
	}
	else
	{
		throw std::invalid_argument("Error: energy out of range");
	}
	return crystalMaterial;
}

}  // namespace scatter
}  // namespace yrt
