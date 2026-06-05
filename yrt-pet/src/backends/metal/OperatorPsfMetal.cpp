/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/OperatorPsfMetal.hpp"

#include "yrt-pet/backends/metal/PsfOps.hpp"
#include "yrt-pet/utils/Array.hpp"
#include "yrt-pet/utils/Tools.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace yrt::backend::metal
{
namespace
{

struct UniformPsfKernels
{
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;
};

std::vector<float> flipped(const std::vector<float>& kernel)
{
	return std::vector<float>(kernel.rbegin(), kernel.rend());
}

std::vector<float> readKernelRow(const Array2DOwned<float>& data,
                                 std::size_t row, int size)
{
	std::vector<float> kernel;
	kernel.reserve(static_cast<std::size_t>(size));
	for (int i = 0; i < size; ++i)
	{
		kernel.push_back(data[row][static_cast<std::size_t>(i)]);
	}
	return kernel;
}

UniformPsfKernels readUniformPsfCsv(const std::string& psfFilename)
{
	Array2DOwned<float> data;
	util::readCSV<float>(psfFilename, data);
	const auto dims = data.getDims();
	if (dims[0] < 4 || dims[1] < 3)
	{
		throw std::runtime_error(
		    "PSF CSV must contain at least four rows and three columns");
	}

	const std::array<int, 3> kernelSizes = {
	    static_cast<int>(data[3][0]),
	    static_cast<int>(data[3][1]),
	    static_cast<int>(data[3][2])};
	for (const int size : kernelSizes)
	{
		if (size <= 0 || size % 2 == 0 ||
		    static_cast<std::size_t>(size) > dims[1])
		{
			throw std::runtime_error(
			    "PSF CSV kernel sizes must be positive odd values");
		}
	}

	return {readKernelRow(data, 0, kernelSizes[0]),
	        readKernelRow(data, 1, kernelSizes[1]),
	        readKernelRow(data, 2, kernelSizes[2])};
}

std::size_t imageByteCount(const ImageShape& shape)
{
	return sizeof(float) * shape.voxelCount();
}

bool copyKernelToBuffer(const Context& context, const std::vector<float>& kernel,
                        Buffer& buffer)
{
	const std::size_t byteCount = sizeof(float) * kernel.size();
	if (!context.isValid() || kernel.empty())
	{
		return false;
	}
	if (buffer.isValid() && buffer.byteCount() >= byteCount)
	{
		return true;
	}
	buffer = Buffer::copyFromHost(context.device(), kernel.data(), byteCount);
	return buffer.isValid();
}

}  // namespace

OperatorPsfMetal::OperatorPsfMetal(const std::string& imagePsfFilename)
    : OperatorPsfMetal{Context{}, imagePsfFilename}
{
}

OperatorPsfMetal::OperatorPsfMetal(const std::vector<float>& kernelX,
                                   const std::vector<float>& kernelY,
                                   const std::vector<float>& kernelZ)
    : OperatorPsfMetal{Context{}, kernelX, kernelY, kernelZ}
{
}

OperatorPsfMetal::OperatorPsfMetal(const Context& context,
                                   const std::string& imagePsfFilename)
    : m_context{context}
{
	const UniformPsfKernels kernels = readUniformPsfCsv(imagePsfFilename);
	m_kernelX = kernels.x;
	m_kernelY = kernels.y;
	m_kernelZ = kernels.z;
	m_kernelXFlipped = flipped(m_kernelX);
	m_kernelYFlipped = flipped(m_kernelY);
	m_kernelZFlipped = flipped(m_kernelZ);
}

OperatorPsfMetal::OperatorPsfMetal(const Context& context,
                                   const std::vector<float>& kernelX,
                                   const std::vector<float>& kernelY,
                                   const std::vector<float>& kernelZ)
    : m_context{context},
      m_kernelX{kernelX},
      m_kernelY{kernelY},
      m_kernelZ{kernelZ},
      m_kernelXFlipped{flipped(kernelX)},
      m_kernelYFlipped{flipped(kernelY)},
      m_kernelZFlipped{flipped(kernelZ)}
{
}

bool OperatorPsfMetal::isValid() const
{
	return m_context.isValid();
}

const std::string& OperatorPsfMetal::errorMessage() const
{
	return m_context.errorMessage();
}

bool OperatorPsfMetal::applyA(const Image& input, Image& output) const
{
	return convolve3DSeparableHost(m_context, input, output, m_kernelX,
	                               m_kernelY, m_kernelZ);
}

bool OperatorPsfMetal::applyAH(const Image& input, Image& output) const
{
	return convolve3DSeparableHost(m_context, input, output, m_kernelXFlipped,
	                               m_kernelYFlipped, m_kernelZFlipped);
}

bool OperatorPsfMetal::applyA(const Buffer& input, Buffer& output,
                              const ImageShape& shape) const
{
	return ensureKernelBuffers() && apply(input, output, m_kernelXBuffer,
	                                  static_cast<std::uint32_t>(
	                                      m_kernelX.size()),
	                                  m_kernelYBuffer,
	                                  static_cast<std::uint32_t>(
	                                      m_kernelY.size()),
	                                  m_kernelZBuffer,
	                                  static_cast<std::uint32_t>(
	                                      m_kernelZ.size()),
	                                  shape);
}

bool OperatorPsfMetal::applyAH(const Buffer& input, Buffer& output,
                               const ImageShape& shape) const
{
	return ensureKernelBuffers() && apply(input, output, m_kernelXFlippedBuffer,
	                                  static_cast<std::uint32_t>(
	                                      m_kernelXFlipped.size()),
	                                  m_kernelYFlippedBuffer,
	                                  static_cast<std::uint32_t>(
	                                      m_kernelYFlipped.size()),
	                                  m_kernelZFlippedBuffer,
	                                  static_cast<std::uint32_t>(
	                                      m_kernelZFlipped.size()),
	                                  shape);
}

const std::vector<float>& OperatorPsfMetal::getKernelX() const
{
	return m_kernelX;
}

const std::vector<float>& OperatorPsfMetal::getKernelY() const
{
	return m_kernelY;
}

const std::vector<float>& OperatorPsfMetal::getKernelZ() const
{
	return m_kernelZ;
}

bool OperatorPsfMetal::ensureKernelBuffers() const
{
	return copyKernelToBuffer(m_context, m_kernelX, m_kernelXBuffer) &&
	       copyKernelToBuffer(m_context, m_kernelY, m_kernelYBuffer) &&
	       copyKernelToBuffer(m_context, m_kernelZ, m_kernelZBuffer) &&
	       copyKernelToBuffer(
	           m_context, m_kernelXFlipped, m_kernelXFlippedBuffer) &&
	       copyKernelToBuffer(
	           m_context, m_kernelYFlipped, m_kernelYFlippedBuffer) &&
	       copyKernelToBuffer(
	           m_context, m_kernelZFlipped, m_kernelZFlippedBuffer);
}

bool OperatorPsfMetal::ensureScratchBuffer(const ImageShape& shape) const
{
	const std::size_t byteCount = imageByteCount(shape);
	if (!m_context.isValid() || byteCount == 0)
	{
		return false;
	}
	if (!m_scratchBuffer.isValid() ||
	    m_scratchBuffer.byteCount() < byteCount)
	{
		m_scratchBuffer = Buffer::allocate(m_context.device(), byteCount);
	}
	return m_scratchBuffer.isValid();
}

bool OperatorPsfMetal::ensureOutputBuffer(Buffer& output,
                                          const ImageShape& shape) const
{
	const std::size_t byteCount = imageByteCount(shape);
	if (!m_context.isValid() || byteCount == 0)
	{
		return false;
	}
	if (!output.isValid() || output.byteCount() < byteCount)
	{
		output = Buffer::allocate(m_context.device(), byteCount);
	}
	return output.isValid();
}

bool OperatorPsfMetal::apply(const Buffer& input, Buffer& output,
                             const Buffer& kernelX,
                             std::uint32_t kernelXSize,
                             const Buffer& kernelY,
                             std::uint32_t kernelYSize,
                             const Buffer& kernelZ,
                             std::uint32_t kernelZSize,
                             const ImageShape& shape) const
{
	return ensureOutputBuffer(output, shape) && ensureScratchBuffer(shape) &&
	       convolve3DSeparableBuffer(m_context, input, output, m_scratchBuffer,
	           kernelX, kernelXSize, kernelY, kernelYSize, kernelZ,
	           kernelZSize, shape);
}

}  // namespace yrt::backend::metal
