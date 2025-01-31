/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "scatter/Crystal.hpp"
#include "utils/Assert.hpp"
#include "utils/Utilities.hpp"

#include <stdexcept>
#include <string>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_crystal(py::module& m)
{
	py::enum_<Scatter::CrystalMaterial>(m, "CrystalMaterial")
	    .value("LSO", Scatter::CrystalMaterial::LSO)
	    .value("LYSO", Scatter::CrystalMaterial::LYSO)
	    .export_values();
	m.def("getMuDet", &Scatter::getMuDet);
	m.def("getCrystalMaterialFromName", &Scatter::getCrystalMaterialFromName);
	m.def("getCrystal", &Scatter::getCrystalMaterialFromName);  // alias
}
#endif

namespace Scatter
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
		    Util::toUpper(crystalMaterial_name);

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

}  // namespace Scatter