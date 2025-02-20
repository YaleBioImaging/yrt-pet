/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "utils/GPUUtils.cuh"

namespace Util
{
	HOST_DEVICE_CALLABLE inline void get_alpha(float r0, float r1, float p1,
	                                           float p2, float inv_p12,
	                                           float& amin, float& amax)
	{
		amin = 0.0;
		amax = 1.0;
		if (p1 != p2)
		{
			const float a0 = (r0 - p1) * inv_p12;
			const float a1 = (r1 - p1) * inv_p12;
			if (a0 < a1)
			{
				amin = a0;
				amax = a1;
			}
			else
			{
				amin = a1;
				amax = a0;
			}
		}
		else if (p1 < r0 || p1 > r1)
		{
			amax = 0.0;
			amin = 1.0;
		}
	}

	HOST_DEVICE_CALLABLE inline float calculateIntegral(const float* kernel,
	                                                    int kernelSize,
	                                                    float kernelStep,
	                                                    float x0, float x1)
	{
		const int halfSize = (kernelSize - 1) / 2;
		if (x0 > (halfSize + 1) * kernelStep ||
		    x1 < -(halfSize + 1) * kernelStep || x0 >= x1)
		{
			return 0.f;
		}

		// x0
		const float x0_s = x0 / kernelStep;
		const int x0_i = static_cast<int>(std::floor(x0_s + halfSize));
		const float tau_0 = x0_s + -(-halfSize + x0_i);
		const float v0_lo = (x0_i < 0 || x0_i >= kernelSize) ? 0 : kernel[x0_i];
		const float v0_hi =
		    (x0_i < -1 || x0_i >= kernelSize - 1) ? 0.f : kernel[x0_i + 1];
		// Interpolated value at x0
		const float v0 = v0_lo * (1 - tau_0) + v0_hi * tau_0;

		// x1
		const float x1_s = x1 / kernelStep;
		const int x1_i = static_cast<int>(std::floor(x1_s + halfSize));
		const float tau_1 = x1_s - (-halfSize + x1_i);
		const float v1_lo = (x1_i < 0 || x1_i >= kernelSize) ? 0 : kernel[x1_i];
		const float v1_hi =
		    (x1_i < -1 || x1_i >= kernelSize - 1) ? 0.f : kernel[x1_i + 1];
		// Interpolated value at x1
		const float v1 = v1_lo * (1 - tau_1) + v1_hi * tau_1;

		// Integral calculation
		float out = 0;
		if (x0_i == x1_i)
		{
			// Special case when x0 and x1 are in the same bin
			out = 0.5f * (v0 + v1) * (x1 - x0);
		}
		else
		{
			// Integration over partial bin for x0
			out += 0.5f * (v0 + v0_hi) * kernelStep * (1 - tau_0);
			// Integration over partial bin for x1
			out += 0.5f * (v1_lo + v1) * kernelStep * tau_1;
			// Add full bins between x0 and x1
			for (int xi = x0_i + 1; xi < x1_i; xi++)
			{
				const float v_lo =
				    (xi < 0 || xi >= kernelSize) ? 0.f : kernel[xi];
				const float v_hi =
				    (xi + 1 < 0 || xi + 1 >= kernelSize) ? 0.f : kernel[xi + 1];
				out += kernelStep * 0.5f * (v_lo + v_hi);
			}
		}
		return out;
	}

}  // namespace Util
