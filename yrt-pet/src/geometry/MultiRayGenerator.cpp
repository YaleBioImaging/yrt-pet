/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/geometry/MultiRayGenerator.hpp"

#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Constants.hpp"

#include <cmath>
#include <utility>
#include <vector>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace yrt
{
void py_setup_multiraygenerator(pybind11::module& m)
{
	auto c = pybind11::class_<MultiRayGenerator>(m, "MultiRayGenerator");
	c.def(pybind11::init<float, float, bool>(), pybind11::arg("thickness_z_i"),
	      pybind11::arg("thickness_trans_i"),
	      pybind11::arg("is_parallel") = MultiRayGenerator::USE_PARALLEL_LINES);
	c.def("setupGenerator", &MultiRayGenerator::setupGenerator);
	c.def("getRandomLine", &MultiRayGenerator::getRandomLine);
}
}  // namespace yrt

#endif

namespace yrt
{

MultiRayGenerator::MultiRayGenerator(float thickness_z_i,
                                     float thickness_trans_i, bool isParallel_i)
    : thickness_z(thickness_z_i),
      thickness_trans(thickness_trans_i),
      isParallel(isParallel_i),
      vect_parrallel_to_z{0, 0, 1},
      vect_parrallel_to_trans1{},
      vect_parrallel_to_trans2{},
      currentLor(nullptr)
{
	isSingleRay = (thickness_trans <= 0 && thickness_z <= 0);
}

void MultiRayGenerator::setupGenerator(const Line3D& lor, const Vector3D& n1,
                                       const Vector3D& n2)
{
	currentLor = &lor;
	if (!isSingleRay)
	{
		vect_parrallel_to_trans1 =
		    n1.crossProduct(vect_parrallel_to_z).normalize();
		vect_parrallel_to_trans2 =
		    n2.crossProduct(vect_parrallel_to_z).normalize();
	}
}

Line3D MultiRayGenerator::getRandomLine(unsigned int& seed) const
{
	if (isSingleRay)
	{
		return *currentLor;
	}
	const float rand_i_1 =
	    static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX) - 0.5f;
	const float rand_j_1 =
	    static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX) - 0.5f;

	const float rand_i_2 =
	    (isParallel) ?
	        rand_i_1 :
	        (static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX) -
	         0.5f);
	const float rand_j_2 =
	    (isParallel) ?
	        rand_j_1 :
	        (static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX) -
	         0.5f);

	const Vector3D pt1 =
	    currentLor->point1 + vect_parrallel_to_z * (rand_i_1 * thickness_z) +
	    vect_parrallel_to_trans1 * (rand_j_1 * thickness_trans);
	const Vector3D pt2 =
	    currentLor->point2 + vect_parrallel_to_z * (rand_i_2 * thickness_z) -
	    vect_parrallel_to_trans2 * (rand_j_2 * thickness_trans);

	return Line3D{pt1, pt2};
}
}  // namespace yrt
