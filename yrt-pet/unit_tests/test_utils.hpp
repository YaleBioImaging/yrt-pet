/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"

#include <random>

namespace yrt
{
class Image;
class ProjectionList;

namespace util::test
{

std::unique_ptr<Scanner> makeScanner();
double getRMSE(const Image& imgRef, const Image& img);
double getRMSE(const ProjectionList& projListRef,
               const ProjectionList& projList);

template <bool EQUAL_NAN = false>
bool allclose(const ProjectionList& projValuesRef,
              const ProjectionList& projValues, float rtol = 1e-5,
              float atol = 1e-8);
template <bool EQUAL_NAN = false>
bool allclose(const Image& imageRef, const Image& image, float rtol = 1e-5,
              float atol = 1e-8);

template <typename TFloat, bool EQUAL_NAN = false>
bool allclose(const TFloat* valuesRef, const TFloat* values, size_t numValues,
              TFloat rtol = 1e-5, TFloat atol = 1e-8);

std::unique_ptr<ImageOwned>
    makeImageWithRandomPrism(const ImageParams& params,
                             std::default_random_engine* p_engine = nullptr);

}  // namespace util::test
}  // namespace yrt
