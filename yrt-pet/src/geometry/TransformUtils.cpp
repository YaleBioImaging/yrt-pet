/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/geometry/TransformUtils.hpp"

#include <cmath>

namespace yrt::util
{

transform_t fromRotationAndTranslationVectors(const Vector3D& rotation,
                                              const Vector3D& translation)
{
	const float alpha = rotation.z;
	const float beta = rotation.y;
	const float gamma = rotation.x;

	// Alpha = yaw (Z), Beta = pitch (Y), Gamma = roll (X)
	const float ca = std::cos(alpha), sa = std::sin(alpha);  // Yaw
	const float cb = std::cos(beta), sb = std::sin(beta);    // Pitch
	const float cc = std::cos(gamma), sc = std::sin(gamma);  // Roll

	// Row-major 3x3 matrix for Rz * Ry * Rx (ZYX)
	const float r00 = ca * cb;
	const float r01 = ca * sb * sc - sa * cc;
	const float r02 = ca * sb * cc + sa * sc;

	const float r10 = sa * cb;
	const float r11 = sa * sb * sc + ca * cc;
	const float r12 = sa * sb * cc - ca * sc;

	const float r20 = -sb;
	const float r21 = cb * sc;
	const float r22 = cb * cc;

	return transform_t{r00, r01, r02, translation.x,
	                   r10, r11, r12, translation.y,
	                   r20, r21, r22, translation.z};
}

}  // namespace yrt::util
