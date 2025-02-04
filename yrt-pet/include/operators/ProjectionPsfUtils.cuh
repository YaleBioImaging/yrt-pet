/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/ProjectionPsfManagerDevice.cuh"
#include "geometry/ProjectorUtils.hpp"

namespace Util
{
	__device__ inline const float*
		getKernel(const float* pd_kernels,
				  const ProjectionPsfProperties& projectionPsfProperties,
				  float p1x, float p1y, float p2x, float p2y)
	{
		float n_plane_x = p2y - p1y;
		float n_plane_y = p1x - p2x;
		const float n_plane_norm =
			sqrt(n_plane_x * n_plane_x + n_plane_y * n_plane_y);
		int s_idx;

		if (n_plane_norm == 0)
		{
			s_idx = 0;
		}
		else
		{
			n_plane_x /= n_plane_norm;
			n_plane_y /= n_plane_norm;
			const float s = fabsf(p1x * n_plane_x + p1y * n_plane_y);

			const int kernelIdx =
				static_cast<int>(floor(s / projectionPsfProperties.sStep));
			const int lastKernel = projectionPsfProperties.numKernels - 1;

			s_idx = kernelIdx < lastKernel ? kernelIdx : lastKernel;
		}

		return pd_kernels + s_idx * projectionPsfProperties.kernelSize;
	}

	__device__ inline float
		getWeight(const float* pd_kernel,
			  const ProjectionPsfProperties& projectionPsfProperties,
			  float x0, float x1)
	{
		const int halfWidth = (projectionPsfProperties.kernelSize + 1) / 2;
		if (x0 > halfWidth * projectionPsfProperties.kSpacing ||
			x1 < -halfWidth * projectionPsfProperties.kSpacing || x0 >= x1)
		{
			return 0.f;
		}
		return Util::calculateIntegral(
			pd_kernel, projectionPsfProperties.kernelSize,
			projectionPsfProperties.kSpacing, x0, x1);
	}
}
