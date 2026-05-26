/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ExperimentalBackend.hpp"

#include "yrt-pet/backends/metal/MetalBackend.hpp"

#include <utility>

namespace yrt::backend::metal
{

ExperimentalBackend::ExperimentalBackend() = default;

ExperimentalBackend::ExperimentalBackend(const std::string& metallibPath)
    : m_context{metallibPath}
{
}

bool ExperimentalBackend::isAvailable() const
{
	return isDeviceAvailable();
}

bool ExperimentalBackend::isValid() const
{
	return m_context.isValid();
}

const std::string& ExperimentalBackend::errorMessage() const
{
	return m_context.errorMessage();
}

const Context& ExperimentalBackend::context() const
{
	return m_context;
}

ProjectionVectorMetal ExperimentalBackend::makeProjectionVector(
    std::vector<float> values) const
{
	return ProjectionVectorMetal{m_context, std::move(values)};
}

ProjectionVectorMetal ExperimentalBackend::makeProjectionVector(
    std::size_t size, float initialValue) const
{
	return ProjectionVectorMetal{m_context, size, initialValue};
}

ImageMetal ExperimentalBackend::makeImage(const ImageParams& params) const
{
	return ImageMetal{m_context, params};
}

ImageMetal ExperimentalBackend::makeImage(const Image& image) const
{
	return ImageMetal{m_context, image};
}

OperatorPsfMetal ExperimentalBackend::makeOperatorPsf(
    const std::string& imagePsfFilename) const
{
	return OperatorPsfMetal{m_context, imagePsfFilename};
}

OperatorPsfMetal ExperimentalBackend::makeOperatorPsf(
    const std::vector<float>& kernelX, const std::vector<float>& kernelY,
    const std::vector<float>& kernelZ) const
{
	return OperatorPsfMetal{m_context, kernelX, kernelY, kernelZ};
}

bool ExperimentalBackend::applyOperatorPsfForward(
    const Image& input, Image& output,
    const std::string& imagePsfFilename) const
{
	OperatorPsfMetal psf = makeOperatorPsf(imagePsfFilename);
	return psf.isValid() && psf.applyA(input, output);
}

bool ExperimentalBackend::applyOperatorPsfForward(
    const Image& input, Image& output, const std::vector<float>& kernelX,
    const std::vector<float>& kernelY, const std::vector<float>& kernelZ) const
{
	OperatorPsfMetal psf = makeOperatorPsf(kernelX, kernelY, kernelZ);
	return psf.isValid() && psf.applyA(input, output);
}

bool ExperimentalBackend::applyOperatorPsfAdjoint(
    const Image& input, Image& output,
    const std::string& imagePsfFilename) const
{
	OperatorPsfMetal psf = makeOperatorPsf(imagePsfFilename);
	return psf.isValid() && psf.applyAH(input, output);
}

bool ExperimentalBackend::applyOperatorPsfAdjoint(
    const Image& input, Image& output, const std::vector<float>& kernelX,
    const std::vector<float>& kernelY, const std::vector<float>& kernelZ) const
{
	OperatorPsfMetal psf = makeOperatorPsf(kernelX, kernelY, kernelZ);
	return psf.isValid() && psf.applyAH(input, output);
}

}  // namespace yrt::backend::metal
