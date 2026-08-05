/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/DynamicFraming.hpp"

#include <memory>

namespace yrt
{

class DetectorMask;
class Histogram3D;
class ListMode;
class ListModeLUTOwned;
class LORMotion;
class ProjectionData;

namespace util
{

template <bool PrintProgress = true>
std::unique_ptr<ImageOwned> timeAverageMoveImage(const LORMotion& lorMotion,
												 const Image* unmovedImage);
template <bool PrintProgress = true>
void timeAverageMoveImage(const LORMotion& lorMotion, const Image* unmovedImage,
						  Image* outImage, frame_t outDynamicFrame = 0);

template <bool PrintProgress = true>
std::unique_ptr<ImageOwned>
	timeAverageMoveImage(const LORMotion& lorMotion, const Image* unmovedImage,
						 timestamp_t timeStart, timestamp_t timeStop);
template <bool PrintProgress = true>
void timeAverageMoveImage(const LORMotion& lorMotion, const Image* unmovedImage,
						  Image* outImage, timestamp_t timeStart,
						  timestamp_t timeStop, frame_t outDynamicFrame = 0);

template <bool PrintProgress = true>
std::unique_ptr<ImageOwned>
	timeAverageMoveImageDynamic(const LORMotion& lorMotion,
								const Image* unmovedImage,
								const DynamicFraming& dynamicFraming);
template <bool PrintProgress = true>
void timeAverageMoveImageDynamic(const LORMotion& lorMotion,
								 const Image* unmovedImage, Image* outImage,
								 const DynamicFraming& dynamicFraming);


void fillCircle(Image& image, float value, float centerX, float centerY,
				float radius, ssize_t zSlice = 0, frame_t dynamicFrame = 0);

void fillEllipse(Image& image, float value, float centerX, float centerY,
				 float semiAxisX, float semiAxisY, float angle = 0.0f,
				 ssize_t zSlice = 0, frame_t dynamicFrame = 0);

std::unique_ptr<ImageOwned> getCircleImage(const ImageParams& imgParams,
										   float value, float centerX,
										   float centerY, float radius);

std::unique_ptr<ImageOwned> getEllipseImage(const ImageParams& imgParams,
											float value, float centerX,
											float centerY, float semiAxisX,
											float semiAxisY,
											float angle = 0.0f);

void fillSphere(Image& image, float value, float centerX, float centerY,
				float centerZ, float radius, frame_t dynamicFrame = 0);

void fillEllipsoid(Image& image, float value, float centerX, float centerY,
				   float centerZ, float semiAxisX, float semiAxisY,
				   float semiAxisZ, frame_t dynamicFrame = 0);

std::unique_ptr<ImageOwned> getSphereImage(const ImageParams& imgParams,
										   float value, float centerX,
										   float centerY, float centerZ,
										   float radius);

std::unique_ptr<ImageOwned> getEllipsoidImage(const ImageParams& imgParams,
											  float value, float centerX,
											  float centerY, float centerZ,
											  float semiAxisX, float semiAxisY,
											  float semiAxisZ);


}
}
