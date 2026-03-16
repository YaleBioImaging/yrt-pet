/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/operators/ProjectorParams.hpp"
#include "yrt-pet/utils/Array.hpp"

namespace yrt
{
class LREM
{
public:
	// YN: Should this class be "friend" with OSEM ?

	LREM() = default;
	virtual ~LREM() = default;

	bool isLowRank();
	bool isDualUpdate();
	void setUpdateH(bool updateH);
	void setHBasis(const Array2DBase<float>& HBasis);
	void saveHBasisBinary(const std::string& base_path, int iter,
	                      int numDigitsInFilename);

	virtual ProjectorParams& getProjectorParams() = 0;

protected:
	void resetSensScaling();
	void applyHUpdate();
	void generateWUpdateSensScaling();
	void generateHUpdateSensScaling(const Image& Wimage,
	                                const Image& sensImage);
	void allocateHBasisTmpBuffer();

	std::vector<float> m_cWUpdate;
	// LR sensitivity matrix factor correction
	std::vector<float> m_cHUpdate;
	// LR H Numerator in H update case
	std::unique_ptr<Array2DOwned<float>> mp_HNumerator;
};
}  // namespace yrt
