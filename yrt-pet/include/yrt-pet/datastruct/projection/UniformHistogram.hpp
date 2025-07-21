/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/Histogram3D.hpp"

namespace yrt
{
class UniformHistogram : public Histogram3D
{
public:
	explicit UniformHistogram(const Scanner& p_scanner, float p_value = 1.0f);

	void writeToFile(const std::string& filename) const override;
	float getProjectionValue(bin_t binId) const override;
	float getProjectionValueFromHistogramBin(
	    histo_bin_t histoBinId) const override;
	void setProjectionValue(bin_t binId, float val) override;
	void incrementProjection(bin_t binId, float val) override;
	void clearProjections(float p_value) override;
	void setValue(float p_value = 0.0f);
	bool isUniform() const override;

private:
	float m_value;
};
}  // namespace yrt
