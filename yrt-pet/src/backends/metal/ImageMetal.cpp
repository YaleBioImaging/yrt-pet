/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ImageMetal.hpp"

#include "yrt-pet/backends/metal/ImageOps.hpp"

namespace yrt::backend::metal
{

ImageMetal::ImageMetal(const ImageParams& params)
    : ImageMetal{Context{}, params}
{
}

ImageMetal::ImageMetal(const Image& image) : ImageMetal{Context{}, image}
{
}

ImageMetal::ImageMetal(const Context& context, const ImageParams& params)
    : m_context{context}, m_image{params}
{
	m_image.allocate();
}

ImageMetal::ImageMetal(const Context& context, const Image& image)
    : ImageMetal{context, image.getParams()}
{
	m_image.copyFromImage(&image);
}

bool ImageMetal::isValid() const
{
	return m_context.isValid();
}

const std::string& ImageMetal::errorMessage() const
{
	return m_context.errorMessage();
}

ImageOwned& ImageMetal::image()
{
	return m_image;
}

const ImageOwned& ImageMetal::image() const
{
	return m_image;
}

const ImageParams& ImageMetal::getParams() const
{
	return m_image.getParams();
}

bool ImageMetal::fill(float value)
{
	return yrt::backend::metal::fill(m_context, m_image, value);
}

bool ImageMetal::multiplyByScalar(float scalar)
{
	return yrt::backend::metal::multiplyByScalar(m_context, m_image, scalar);
}

bool ImageMetal::add3DTo3D(const Image& input3D)
{
	return yrt::backend::metal::add3DTo3D(m_context, input3D, m_image);
}

bool ImageMetal::add3DTo3D(const ImageMetal& input3D)
{
	return add3DTo3D(input3D.image());
}

bool ImageMetal::add3DTo4D(const Image& input3D)
{
	return yrt::backend::metal::add3DTo4D(m_context, input3D, m_image);
}

bool ImageMetal::add3DTo4D(const ImageMetal& input3D)
{
	return add3DTo4D(input3D.image());
}

bool ImageMetal::applyThreshold(const Image& mask3D, float threshold,
                                float valLeScale, float valLeOffset,
                                float valGtScale, float valGtOffset)
{
	return yrt::backend::metal::applyThreshold(m_context, m_image, mask3D,
	                                           threshold, valLeScale,
	                                           valLeOffset, valGtScale,
	                                           valGtOffset);
}

bool ImageMetal::applyThreshold(const ImageMetal& mask3D, float threshold,
                                float valLeScale, float valLeOffset,
                                float valGtScale, float valGtOffset)
{
	return applyThreshold(mask3D.image(), threshold, valLeScale, valLeOffset,
	                      valGtScale, valGtOffset);
}

bool ImageMetal::applyThresholdBroadcast(const Image& mask3D, float threshold,
                                         float valLeScale, float valLeOffset,
                                         float valGtScale, float valGtOffset)
{
	return yrt::backend::metal::applyThresholdBroadcast(
	    m_context, m_image, mask3D, threshold, valLeScale, valLeOffset,
	    valGtScale, valGtOffset);
}

bool ImageMetal::applyThresholdBroadcast(const ImageMetal& mask3D,
                                         float threshold, float valLeScale,
                                         float valLeOffset, float valGtScale,
                                         float valGtOffset)
{
	return applyThresholdBroadcast(mask3D.image(), threshold, valLeScale,
	                               valLeOffset, valGtScale, valGtOffset);
}

bool ImageMetal::updateEMStatic(const Image& update3D,
                                const Image& sensitivity3D, float threshold)
{
	return yrt::backend::metal::updateEMStatic(m_context, m_image, update3D,
	                                           sensitivity3D, threshold);
}

bool ImageMetal::updateEMStatic(const ImageMetal& update3D,
                                const ImageMetal& sensitivity3D,
                                float threshold)
{
	return updateEMStatic(update3D.image(), sensitivity3D.image(), threshold);
}

bool ImageMetal::updateEMDynamic(const Image& update4D,
                                 const Image& sensitivity3D, float threshold)
{
	return yrt::backend::metal::updateEMDynamic(m_context, m_image, update4D,
	                                            sensitivity3D, threshold);
}

bool ImageMetal::updateEMDynamic(const ImageMetal& update4D,
                                 const ImageMetal& sensitivity3D,
                                 float threshold)
{
	return updateEMDynamic(update4D.image(), sensitivity3D.image(), threshold);
}

}  // namespace yrt::backend::metal
