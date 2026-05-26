/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/MetalContext.hpp"

#ifndef YRTPET_METAL_SMOKE_METALLIB
#define YRTPET_METAL_SMOKE_METALLIB ""
#endif

namespace yrt::backend::metal
{

Context::Context() : Context{YRTPET_METAL_SMOKE_METALLIB} {}

Context::Context(const std::string& metallibPath)
    : m_device{Device::createSystemDefault()}
{
	if (!m_device.isValid())
	{
		m_errorMessage = "Metal device unavailable";
		return;
	}

	m_library = Library::loadFromFile(m_device, metallibPath);
	if (!m_library.isValid())
	{
		m_errorMessage = "Metal library unavailable: " + metallibPath;
		return;
	}

	m_commandQueue = CommandQueue::create(m_device);
	if (!m_commandQueue.isValid())
	{
		m_errorMessage = "Metal command queue unavailable";
	}
}

bool Context::isValid() const
{
	return m_device.isValid() && m_library.isValid() &&
	       m_commandQueue.isValid();
}

const std::string& Context::errorMessage() const
{
	return m_errorMessage;
}

const Device& Context::device() const
{
	return m_device;
}

const Library& Context::library() const
{
	return m_library;
}

const CommandQueue& Context::commandQueue() const
{
	return m_commandQueue;
}

}  // namespace yrt::backend::metal
