/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/PsfFileOps.hpp"

#include "yrt-pet/backends/metal/ExperimentalBackend.hpp"

namespace yrt::backend::metal
{

bool applyPsfForward(const Image& input, Image& output,
                     const std::string& imagePsfFilename)
{
	const ExperimentalBackend backend;
	return backend.isAvailable() && backend.isValid() &&
	       backend.applyOperatorPsfForward(input, output, imagePsfFilename);
}

bool applyPsfAdjoint(const Image& input, Image& output,
                     const std::string& imagePsfFilename)
{
	const ExperimentalBackend backend;
	return backend.isAvailable() && backend.isValid() &&
	       backend.applyOperatorPsfAdjoint(input, output, imagePsfFilename);
}

}  // namespace yrt::backend::metal
