/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

namespace yrt
{
class Image;
}

namespace yrt::backend::metal
{

class Context;

bool fill(const Context& context, Image& image, float value);
bool multiplyByScalar(const Context& context, Image& image, float scalar);
bool add3DTo3D(const Context& context, const Image& input3D, Image& output3D);
bool add3DTo4D(const Context& context, const Image& input3D, Image& output4D);
bool applyThreshold(const Context& context, Image& image3D, const Image& mask3D,
                    float threshold, float valLeScale, float valLeOffset,
                    float valGtScale, float valGtOffset);
bool applyThresholdBroadcast(const Context& context, Image& image4D,
                             const Image& mask3D, float threshold,
                             float valLeScale, float valLeOffset,
                             float valGtScale, float valGtOffset);
bool updateEMStatic(const Context& context, Image& image3D, const Image& update3D,
                    const Image& sensitivity3D, float threshold);
bool updateEMDynamic(const Context& context, Image& image4D, const Image& update4D,
                     const Image& sensitivity3D, float threshold);

}  // namespace yrt::backend::metal
