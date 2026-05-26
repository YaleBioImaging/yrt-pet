/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"

#include <string>

namespace yrt::backend::metal
{

class ImageMetal
{
public:
	explicit ImageMetal(const ImageParams& params);
	explicit ImageMetal(const Image& image);
	ImageMetal(const Context& context, const ImageParams& params);
	ImageMetal(const Context& context, const Image& image);

	bool isValid() const;
	const std::string& errorMessage() const;

	ImageOwned& image();
	const ImageOwned& image() const;
	const ImageParams& getParams() const;

	bool fill(float value);
	bool multiplyByScalar(float scalar);
	bool add3DTo3D(const Image& input3D);
	bool add3DTo3D(const ImageMetal& input3D);
	bool add3DTo4D(const Image& input3D);
	bool add3DTo4D(const ImageMetal& input3D);
	bool applyThreshold(const Image& mask3D, float threshold,
	                    float valLeScale, float valLeOffset,
	                    float valGtScale, float valGtOffset);
	bool applyThreshold(const ImageMetal& mask3D, float threshold,
	                    float valLeScale, float valLeOffset,
	                    float valGtScale, float valGtOffset);
	bool applyThresholdBroadcast(const Image& mask3D, float threshold,
	                             float valLeScale, float valLeOffset,
	                             float valGtScale, float valGtOffset);
	bool applyThresholdBroadcast(const ImageMetal& mask3D, float threshold,
	                             float valLeScale, float valLeOffset,
	                             float valGtScale, float valGtOffset);
	bool updateEMStatic(const Image& update3D, const Image& sensitivity3D,
	                    float threshold);
	bool updateEMStatic(const ImageMetal& update3D,
	                    const ImageMetal& sensitivity3D, float threshold);
	bool updateEMDynamic(const Image& update4D, const Image& sensitivity3D,
	                     float threshold);
	bool updateEMDynamic(const ImageMetal& update4D,
	                     const ImageMetal& sensitivity3D, float threshold);

private:
	Context m_context;
	ImageOwned m_image;
};

}  // namespace yrt::backend::metal
