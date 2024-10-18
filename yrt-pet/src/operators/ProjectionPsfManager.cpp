/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/ProjectionPsfManager.hpp"

#include "geometry/ProjectorUtils.hpp"
#include "utils/Tools.hpp"

ProjectionPsfManager::ProjectionPsfManager() : m_sStep(0.f), m_kSpacing(0.f) {}

ProjectionPsfManager::ProjectionPsfManager(const std::string& psfFilename)
    : ProjectionPsfManager{}
{
	readFromFileInternal(psfFilename);
}

void ProjectionPsfManager::readFromFile(const std::string& psfFilename)
{
	readFromFileInternal(psfFilename);
}

void ProjectionPsfManager::readFromFileInternal(const std::string& psfFilename)
{
	Util::readCSV<float>(psfFilename, m_kernelDataRaw);
	m_sStep = m_kernelDataRaw[0][0];
	m_kSpacing = m_kernelDataRaw[0][1];
	const int kernel_size = m_kernelDataRaw[0][2];
	const int num_s = (m_kernelDataRaw.getSize(1) - 3) / kernel_size;

	if ((kernel_size % 2) == 0)
	{
		throw std::runtime_error("Kernels must be of odd length");
	}

	m_kernels.bind(m_kernelDataRaw[0] + 3, num_s, kernel_size);
	m_kernelsFlipped.allocate(m_kernels.getSize(0), m_kernels.getSize(1));
	for (size_t i = 0; i < m_kernels.getSize(0); i++)
	{
		for (size_t j = 0; j < m_kernels.getSize(1); j++)
		{
			m_kernelsFlipped[i][m_kernels.getSize(1) - 1 - j] = m_kernels[i][j];
		}
	}
}


float ProjectionPsfManager::getHalfWidth_mm() const
{
	// Zero-padding by one element
	return (m_kernels.getSize(1) + 1) / 2 * m_kSpacing;
}

int ProjectionPsfManager::getKernelSize() const
{
	return m_kernels.getSize(1);
}

const float* ProjectionPsfManager::getKernel(const Line3D& lor,
                                             bool flagFlipped) const
{
	const Vector3D p1 = lor.point1;
	const Vector3D p2 = lor.point2;
	float n_plane_x = p2.y - p1.y;
	float n_plane_y = p1.x - p2.x;
	const float n_plane_norm =
	    std::sqrt(n_plane_x * n_plane_x + n_plane_y * n_plane_y);
	size_t s_idx;
	if (n_plane_norm == 0)
	{
		s_idx = 0;
	}
	else
	{
		n_plane_x /= n_plane_norm;
		n_plane_y /= n_plane_norm;
		float s = std::abs(p1.x * n_plane_x + p1.y * n_plane_y);
		s_idx = std::min(static_cast<size_t>(std::floor(s / m_sStep)),
		                 m_kernels.getSize(0) - 1);
	}

	if (!flagFlipped)
	{
		return m_kernels[s_idx];
	}
	return m_kernelsFlipped[s_idx];
}

float ProjectionPsfManager::getWeight(const float* kernel, const float x0,
                                      const float x1) const
{
	const int halfWidth = (m_kernels.getSize(1) + 1) / 2;
	if (x0 > halfWidth * m_kSpacing || x1 < -halfWidth * m_kSpacing || x0 >= x1)
	{
		return 0.f;
	}
	return Util::calculateIntegral(kernel, m_kernels.getSize(1), m_kSpacing, x0,
	                               x1);
}
