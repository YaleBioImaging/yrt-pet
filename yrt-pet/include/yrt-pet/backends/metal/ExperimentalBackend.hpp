/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/ImageMetal.hpp"
#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/OperatorPsfMetal.hpp"
#include "yrt-pet/backends/metal/ProjectionVectorMetal.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace yrt::backend::metal
{

class ExperimentalBackend
{
public:
	ExperimentalBackend();
	explicit ExperimentalBackend(const std::string& metallibPath);

	bool isAvailable() const;
	bool isValid() const;
	const std::string& errorMessage() const;

	const Context& context() const;

	ProjectionVectorMetal makeProjectionVector(
	    std::vector<float> values) const;
	ProjectionVectorMetal makeProjectionVector(std::size_t size,
	                                           float initialValue) const;
	ImageMetal makeImage(const ImageParams& params) const;
	ImageMetal makeImage(const Image& image) const;
	OperatorPsfMetal makeOperatorPsf(
	    const std::string& imagePsfFilename) const;
	OperatorPsfMetal makeOperatorPsf(const std::vector<float>& kernelX,
	                                 const std::vector<float>& kernelY,
	                                 const std::vector<float>& kernelZ) const;
	bool applyOperatorPsfForward(
	    const Image& input, Image& output,
	    const std::string& imagePsfFilename) const;
	bool applyOperatorPsfForward(const Image& input, Image& output,
	                             const std::vector<float>& kernelX,
	                             const std::vector<float>& kernelY,
	                             const std::vector<float>& kernelZ) const;
	bool applyOperatorPsfAdjoint(
	    const Image& input, Image& output,
	    const std::string& imagePsfFilename) const;
	bool applyOperatorPsfAdjoint(const Image& input, Image& output,
	                             const std::vector<float>& kernelX,
	                             const std::vector<float>& kernelY,
	                             const std::vector<float>& kernelZ) const;

private:
	Context m_context;
};

}  // namespace yrt::backend::metal
