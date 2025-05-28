/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/TransformUtils.hpp"

#include <cmath>

namespace Util
{
	transform_t invertTransform(const transform_t& trnsfrm)
	{
		// Extract R (rotation 3x3) and t (translation)
		const float R[9] = {trnsfrm.r00, trnsfrm.r01, trnsfrm.r02,   // Row 0
		                    trnsfrm.r10, trnsfrm.r11, trnsfrm.r12,   // Row 1
		                    trnsfrm.r20, trnsfrm.r21, trnsfrm.r22};  // Row 2
		const float t[3] = {trnsfrm.tx, trnsfrm.ty, trnsfrm.tz};

		// Compute R^T (transpose of rotation)
		const float RT[9] = {R[0], R[3], R[6],   // Row 0
		                     R[1], R[4], R[7],   // Row 1
		                     R[2], R[5], R[8]};  // Row 2

		// Compute -R^T * t
		const float t_new[3] = {-(RT[0] * t[0] + RT[1] * t[1] + RT[2] * t[2]),
		                        -(RT[3] * t[0] + RT[4] * t[1] + RT[5] * t[2]),
		                        -(RT[6] * t[0] + RT[7] * t[1] + RT[8] * t[2])};

		return transform_t{RT[0], RT[1], R[2], t_new[0],   // Row 0
		                   RT[3], RT[4], R[5], t_new[1],   // Row 1
		                   RT[6], RT[7], R[8], t_new[2]};  // Row 2
	}

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
}  // namespace Util
