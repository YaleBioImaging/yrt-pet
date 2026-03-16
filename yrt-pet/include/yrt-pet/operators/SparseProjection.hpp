/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

namespace yrt
{

class Image;
class Projector;
class SparseHistogram;

namespace util
{
template <bool PrintProgress = true>
void forwProjectToSparseHistogram(const Image& sourceImage,
                                  const Projector& projector,
                                  SparseHistogram& sparseHistogram);
}
}  // namespace yrt