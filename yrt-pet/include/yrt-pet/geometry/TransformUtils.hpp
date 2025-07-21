/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/Vector3D.hpp"
#include "yrt-pet/utils/Types.hpp"

namespace yrt
{
namespace util
{
transform_t invertTransform(const transform_t& trnsfrm);
transform_t fromRotationAndTranslationVectors(const Vector3D& rotation,
                                              const Vector3D& translation);
}  // namespace util
}  // namespace yrt
