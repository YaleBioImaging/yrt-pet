/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "datastruct/scanner/Scanner.hpp"

class Image;
class ProjectionList;

namespace TestUtils
{
	std::unique_ptr<Scanner> makeScanner();
	double getRMSE(const Image& imgRef, const Image& img);
	double getRMSE(const ProjectionList& projListRef,
	               const ProjectionList& projList);

	template <bool EQUAL_NAN = false>
	bool allclose(const ProjectionList& projValuesRef,
	              const ProjectionList& projValues, float rtol = 1e-5,
	              float atol = 1e-8);

	template <typename TFloat, bool EQUAL_NAN = false>
	bool allclose(const TFloat* valuesRef, const TFloat* values,
	              size_t numValues, TFloat rtol = 1e-5, TFloat atol = 1e-8);

	std::unique_ptr<ImageOwned>
	    makeImageWithRandomPrism(const ImageParams& params);
}  // namespace TestUtils
