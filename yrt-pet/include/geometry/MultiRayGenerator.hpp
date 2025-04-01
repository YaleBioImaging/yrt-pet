/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/Line3D.hpp"

class Scanner;

class MultiRayGenerator
{
public:
	static constexpr bool USE_PARALLEL_LINES = false;
	MultiRayGenerator(float thickness_z_i, float thickness_trans_i,
	                  bool isParallel_i = USE_PARALLEL_LINES);
	Line3D getRandomLine(unsigned int& seed) const;
	void setupGenerator(const Line3D& lor, const Vector3D& n1,
	                    const Vector3D& n2);

protected:
	float thickness_z, thickness_trans;
	bool isSingleRay;
	bool isParallel;

private:
	Vector3D vect_parrallel_to_z;
	Vector3D vect_parrallel_to_trans1;
	Vector3D vect_parrallel_to_trans2;
	const Line3D* currentLor;
};
