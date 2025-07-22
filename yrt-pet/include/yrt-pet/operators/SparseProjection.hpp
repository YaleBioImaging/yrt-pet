/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

namespace yrt
{

class Image;
class OperatorProjector;
class SparseHistogram;

namespace util
{
void forwProjectToSparseHistogram(const Image& sourceImage,
                                  const OperatorProjector& projector,
                                  SparseHistogram& sparseHistogram);
}
}  // namespace yrt