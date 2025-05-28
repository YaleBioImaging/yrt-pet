/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "utils/Types.hpp"
#include "geometry/Vector3D.hpp"

namespace Util
{
	transform_t invertTransform(const transform_t& trnsfrm);
	transform_t fromRotationAndTranslationVectors(const Vector3D& rotation,
		const Vector3D& translation);
}
