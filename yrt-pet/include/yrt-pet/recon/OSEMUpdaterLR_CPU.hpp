/*
* This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/recon/OSEMUpdater_CPU.hpp"

namespace yrt
{
class OSEMUpdaterLR_CPU : public OSEMUpdater_CPU
{
	explicit OSEMUpdaterLR_CPU(OSEM_CPU* pp_osem);

protected:
	// LR H Numerator in H update case
	std::unique_ptr<Array2D<float>> mp_HNumerator;
	// LR sensitivity matrix factor correction
	std::vector<float> m_cWUpdate;
	std::vector<float> m_cHUpdate;
};
}