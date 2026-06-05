/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/ImageSpaceKernels.hpp"
#include "yrt-pet/backends/metal/MetalContext.hpp"

#include <string>
#include <vector>

namespace yrt
{
class Image;
}

namespace yrt::backend::metal
{

class OperatorPsfMetal
{
public:
	explicit OperatorPsfMetal(const std::string& imagePsfFilename);
	OperatorPsfMetal(const std::vector<float>& kernelX,
	                 const std::vector<float>& kernelY,
	                 const std::vector<float>& kernelZ);
	OperatorPsfMetal(const Context& context,
	                 const std::string& imagePsfFilename);
	OperatorPsfMetal(const Context& context, const std::vector<float>& kernelX,
	                 const std::vector<float>& kernelY,
	                 const std::vector<float>& kernelZ);

	bool isValid() const;
	const std::string& errorMessage() const;

	bool applyA(const Image& input, Image& output) const;
	bool applyAH(const Image& input, Image& output) const;
	bool applyA(const Buffer& input, Buffer& output,
	            const ImageShape& shape) const;
	bool applyAH(const Buffer& input, Buffer& output,
	             const ImageShape& shape) const;

	const std::vector<float>& getKernelX() const;
	const std::vector<float>& getKernelY() const;
	const std::vector<float>& getKernelZ() const;

private:
	bool ensureKernelBuffers() const;
	bool ensureScratchBuffer(const ImageShape& shape) const;
	bool ensureOutputBuffer(Buffer& output, const ImageShape& shape) const;
	bool apply(const Buffer& input, Buffer& output, const Buffer& kernelX,
	           std::uint32_t kernelXSize, const Buffer& kernelY,
	           std::uint32_t kernelYSize, const Buffer& kernelZ,
	           std::uint32_t kernelZSize, const ImageShape& shape) const;

	Context m_context;
	std::vector<float> m_kernelX;
	std::vector<float> m_kernelY;
	std::vector<float> m_kernelZ;
	std::vector<float> m_kernelXFlipped;
	std::vector<float> m_kernelYFlipped;
	std::vector<float> m_kernelZFlipped;
	mutable Buffer m_kernelXBuffer;
	mutable Buffer m_kernelYBuffer;
	mutable Buffer m_kernelZBuffer;
	mutable Buffer m_kernelXFlippedBuffer;
	mutable Buffer m_kernelYFlippedBuffer;
	mutable Buffer m_kernelZFlippedBuffer;
	mutable Buffer m_scratchBuffer;
};

}  // namespace yrt::backend::metal
