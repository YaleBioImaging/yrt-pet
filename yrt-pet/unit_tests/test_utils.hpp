/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/scanner/Scanner.hpp"

class Image;
class ProjectionList;

namespace TestUtils
{
	std::unique_ptr<Scanner> makeScanner();
	double getRMSE(const Image& imgRef, const Image& img);
	double getRMSE(const ProjectionList& projListRef,
	               const ProjectionList& projList);
}  // namespace TestUtils
