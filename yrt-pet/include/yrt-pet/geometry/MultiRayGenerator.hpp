/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/Line3D.hpp"

namespace yrt
{
class Scanner;

class MultiRayGenerator
{
public:
	static constexpr bool USE_PARALLEL_LINES = false;
	static constexpr bool USE_DEPTH = false;
	MultiRayGenerator(float thickness_z_i, float thickness_trans_i,
	                  float depth_i = 0.f,
	                  bool isParallel_i = USE_PARALLEL_LINES,
	                  bool hasDepth_i = USE_DEPTH);
	Line3D getRandomLine(unsigned int& seed) const;
	void setupGenerator(const Line3D& lor, const Vector3D& n1,
	                    const Vector3D& n2);

protected:
	float thickness_z, thickness_trans, depth;
	bool isSingleRay;
	bool isParallel;
	bool hasDepth;

private:
	Vector3D vect_parrallel_to_z;
	Vector3D vect_parrallel_to_trans1;
	Vector3D vect_parrallel_to_trans2;
	Vector3D vect_n1;
	Vector3D vect_n2;
	const Line3D* currentLor;
};
}  // namespace yrt
