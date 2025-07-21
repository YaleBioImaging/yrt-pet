/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/GPUUtils.cuh"

#include "yrt-pet/geometry/Constants.hpp"

#include <cmath>

namespace yrt
{
class TimeOfFlightHelper
{
public:
	explicit TimeOfFlightHelper(float tof_width_ps, int tof_n_std = -1);

	HOST_DEVICE_CALLABLE inline void getAlphaRange(float& alpha_min,
	                                               float& alpha_max,
	                                               float lorNorm,
	                                               float tofValue_ps) const
	{
		using namespace std;
		if (m_truncWidth_mm <= 0.f)
		{
			alpha_min = 0.0;
			alpha_max = 1.0;
		}
		else
		{
			const float tof_value_mm = tofValue_ps * SPEED_OF_LIGHT_MM_PS * 0.5;
			alpha_min =
			    max(0.0, 0.5 + (tof_value_mm - m_truncWidth_mm) / lorNorm);
			alpha_max =
			    min(1.0, 0.5 + (tof_value_mm + m_truncWidth_mm) / lorNorm);
		}
	}

	HOST_DEVICE_CALLABLE inline float getWeight(float lorNorm,
	                                            float tofValue_ps,
	                                            float offLo_mm,
	                                            float offHi_mm) const
	{
		const float tof_value_mm = tofValue_ps * SPEED_OF_LIGHT_MM_PS * 0.5;
		const float pc = 0.5 * lorNorm + tof_value_mm;

		const float x_cent_norm = (0.5f * (offLo_mm + offHi_mm) - pc) / m_sigma;
		return exp(-0.5f * x_cent_norm * x_cent_norm) * m_norm;
	}

	float getSigma() const;
	float getTruncWidth() const;
	float getNorm() const;

private:
	// FWHM
	float m_sigma;
	float m_truncWidth_mm;
	float m_norm;
};
}  // namespace yrt
