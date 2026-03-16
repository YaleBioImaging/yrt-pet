/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "../test_utils.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/scatter/ScatterSpace.hpp"

#include <algorithm>
#include <cmath>
#include <random>


// TODO NOW:
//  - Write helper functions that:
//     - Forward project into a Histogram3D and image and add Poisson noise
//     - Convert the Histogram3D into a ListmodeLUT
//  - Test here (using a small scanner and a random image):
//     - LREM_CPU and LREM_GPU give the same image (in 4D)
//     - LREM with R == T is the same as OSEM
