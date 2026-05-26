/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <vector>

namespace yrt::backend::metal
{

class Context;

bool clear(const Context& context, std::vector<float>& values, float value);
bool add(const Context& context, const std::vector<float>& input,
         std::vector<float>& output);
bool multiplyByScalar(const Context& context, std::vector<float>& values,
                      float scalar);
bool multiplyElementwise(const Context& context,
                         const std::vector<float>& input,
                         std::vector<float>& output);
bool divideMeasurements(const Context& context,
                        const std::vector<float>& measurements,
                        std::vector<float>& output);
bool invert(const Context& context, const std::vector<float>& input,
            std::vector<float>& output);
bool convertToACF(const Context& context, const std::vector<float>& input,
                  std::vector<float>& output, float unitFactor);

}  // namespace yrt::backend::metal
