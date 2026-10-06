/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/projection/Sinogram3D.hpp"

#include <memory>

namespace yrt::sino
{

// Fill a Sinogram3D from list-mode data.
// Events are binned using the scanner LUT (exact getBinIdFromDetPair).
// Out-of-range events (ring diff exceeds maxRingDiff, or radial out of bounds)
// are silently dropped.
// numThreads <= 0 uses globals::getNumThreads().
void fillFromListMode(Sinogram3D& sino, const ListMode& lm,
                      int numThreads = -1);

// Convert Histogram3D to Sinogram3D using the provided params.
// Each H3D bin is mapped to its sinogram bin via getBinIdFromDetPair.
// Out-of-range LORs are silently dropped.
std::unique_ptr<Sinogram3DOwned> fromHistogram3D(const Histogram3D& h3d,
                                                 const SinogramParams& params);

// Convert Sinogram3D to Histogram3D (SSRB-like rebinning).
// Each sinogram bin is mapped to an H3D bin via the approximate
// getDetector1/2 formula.  The result has the same scanner as the sinogram.
std::unique_ptr<Histogram3DOwned> toHistogram3D(const Sinogram3D& sino);

}  // namespace yrt::sino
