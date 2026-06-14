/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Types.hpp"

namespace yrt
{

class Scanner;
class ImageParams;
class Image;
class ImageDevice;

struct RawScannerParams
{
	float crystalSize_trans;
	float crystalSize_z;
	float crystalDepth;
	size_t numDets;
};

struct RawImageParams
{
	ssize_t nx, ny, nz;
	float vx, vy, vz;
	float length_x, length_y, length_z;
	float off_x, off_y, off_z;
	float fovRadius;
};

struct RawImage
{
	RawImageParams rawParams;
	float* rawPointer = nullptr;
};

RawScannerParams getRawScannerParams(const Scanner& scanner);
RawImageParams getRawImageParams(const ImageParams& imgParams);
RawImage getRawImage(Image& img);
RawImage getRawImage(ImageDevice& img);

}  // namespace yrt
