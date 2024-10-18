/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/Line3D.hpp"
#include "utils/Array.hpp"

class ProjectionPsfManager
{
public:
	explicit ProjectionPsfManager(const std::string& psfFilename);
	virtual ~ProjectionPsfManager() = default;
	virtual void readFromFile(const std::string& psfFilename);
	float getHalfWidth_mm() const;
	int getKernelSize() const;

	float getWeight(const float* kernel, float x0, float x1) const;
	const float* getKernel(const Line3D& lor, bool flagFlipped = false) const;

protected:
	ProjectionPsfManager();
	Array2D<float> m_kernelDataRaw;
	Array2DAlias<float> m_kernels;
	Array2D<float> m_kernelsFlipped;
	float m_sStep;
	float m_kSpacing;

private:
	void readFromFileInternal(const std::string& psfFilename);
};
