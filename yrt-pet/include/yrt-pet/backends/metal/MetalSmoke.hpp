/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

namespace yrt::backend::metal
{

bool isAvailable();
bool runSmokeKernel();
bool runProjectionVectorGoldenTests();
bool runImageScalarOpsGoldenTests();
bool runPsfConvolutionGoldenTests();

}  // namespace yrt::backend::metal
