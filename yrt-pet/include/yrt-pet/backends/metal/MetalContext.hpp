/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalBackend.hpp"

#include <string>

namespace yrt::backend::metal
{

class Context
{
public:
	Context();
	explicit Context(const std::string& metallibPath);

	bool isValid() const;
	const std::string& errorMessage() const;

	const Device& device() const;
	const Library& library() const;
	const CommandQueue& commandQueue() const;

private:
	Device m_device;
	Library m_library;
	CommandQueue m_commandQueue;
	std::string m_errorMessage;
};

}  // namespace yrt::backend::metal
