/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */
#pragma once

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"

namespace yrt
{

class Histogram : public ProjectionData
{
public:
	static constexpr bool IsListMode() { return false; }

	virtual float
	    getProjectionValueFromHistogramBin(histo_bin_t histoBinId) const = 0;

protected:
	explicit Histogram(const Scanner& pr_scanner);
};

}  // namespace yrt
