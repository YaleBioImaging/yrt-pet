/*
* This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/Histogram.hpp"
#include "datastruct/projection/ProjectionList.hpp"
#include "datastruct/image/Image.hpp"

/*
 * This class provides the additive correction factors for each LOR given
 * measurements and individual components
 */
class Corrector {

public:
// std::unique_ptr<ProjectionList> getAdditiveCorrectionFactors(ProjectionData* measurements);

private:

Histogram* randoms; // if nullptr, use getRandomsEstimate()
Histogram* scatter; // Eventually will be a sinogram

Histogram* acf; // In case ACFs were already calculated
Image* attenuationImage;

Histogram* sensitivity; // LOR sensitivity, can be nullptr, in which case all LORs are equally sensitive

};
