/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/ImageUtils.hpp"

#include "yrt-pet/utils/ProgressDisplay.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_imageutils(pybind11::module& m)
{
	auto fillShapeDescription = [](const std::string& shapeName)
	{
		return ("Fill a given image with " + shapeName +
		        " given its parameters. This function will increment the "
		        "inside of the shape by `value`.")
		    .c_str();
	};
	m.def("fillCircle", &util::fillCircle, "image"_a, "value"_a, "center_x"_a,
	      "center_y"_a, "radius"_a, "z_slice"_a = 0, "dynamic_frame"_a = 0,
	      fillShapeDescription("a circle"));
	m.def("fillEllipse", &util::fillEllipse, "image"_a, "value"_a, "center_x"_a,
	      "center_y"_a, "semi_axis_x"_a, "semi_axis_y"_a, "angle"_a = 0.0f,
	      "z_slice"_a = 0, "dynamic_frame"_a = 0,
	      fillShapeDescription("an ellipse"));
	m.def("fillSphere", &util::fillSphere, "image"_a, "value"_a, "center_x"_a,
	      "center_y"_a, "center_z"_a, "radius"_a, "dynamic_frame"_a = 0,
	      fillShapeDescription("a sphere"));
	m.def("fillEllipsoid", &util::fillEllipsoid, "image"_a, "value"_a,
	      "center_x"_a, "center_y"_a, "center_z"_a, "semi_axis_x"_a,
	      "semi_axis_y"_a, "semi_axis_z"_a, "dynamic_frame"_a = 0,
	      fillShapeDescription("an ellipsoid"));

	auto getShapeDescription = [](const std::string& shapeName)
	{
		return ("Create an image, fill it with " + shapeName +
		        " given its parameters and return it.")
		    .c_str();
	};
	m.def("getCircleImage", &util::getCircleImage, "img_params"_a, "value"_a,
	      "center_x"_a, "center_y"_a, "radius"_a,
	      getShapeDescription("a circle"));
	m.def("getEllipseImage", &util::getEllipseImage, "img_params"_a, "value"_a,
	      "center_x"_a, "center_y"_a, "semi_axis_x"_a, "semi_axis_y"_a,
	      "angle"_a = 0.0f, getShapeDescription("an ellipse"));
	m.def("getSphereImage", &util::getSphereImage, "img_params"_a, "value"_a,
	      "center_x"_a, "center_y"_a, "center_z"_a, "radius"_a,
	      getShapeDescription("a sphere"));
	m.def("getEllipsoidImage", &util::getEllipsoidImage, "img_params"_a,
	      "value"_a, "center_x"_a, "center_y"_a, "center_z"_a, "semi_axis_x"_a,
	      "semi_axis_y"_a, "semi_axis_z"_a,
	      getShapeDescription("an ellipsoid"));

	m.def("timeAverageMoveImage",
	      static_cast<std::unique_ptr<ImageOwned> (*)(
	          const LORMotion&, const Image*)>(&util::timeAverageMoveImage),
	      "lor_motion"_a, "unmoved_image"_a,
	      "Blur a given image based on given motion information. Return the "
	      "resulting image.");

	m.def(
	    "timeAverageMoveImage",
	    static_cast<void (*)(const LORMotion&, const Image*, Image*, frame_t)>(
	        &util::timeAverageMoveImage),
	    "lor_motion"_a, "unmoved_image"_a, "out_image"_a,
	    "out_dynamic_frame"_a = 0,
	    "Blur a given image based on given motion information. Write directly "
	    "in \"out_image\" in the dynamic frame \"out_dynamic_frame\".");

	m.def("timeAverageMoveImage",
	      static_cast<std::unique_ptr<ImageOwned> (*)(
	          const LORMotion&, const Image*, timestamp_t, timestamp_t)>(
	          &util::timeAverageMoveImage),
	      "lor_motion"_a, "unmoved_image"_a, "time_start"_a, "time_stop"_a,
	      "Blur a given image based on given motion information. Return the "
	      "resulting image. Use \"time_start\" and \"time_stop\" to define how "
	      "motion frames are selected and weighted.");

	m.def("timeAverageMoveImage",
	      static_cast<void (*)(const LORMotion&, const Image*, Image*,
	                           timestamp_t, timestamp_t, frame_t)>(
	          &util::timeAverageMoveImage),
	      "lor_motion"_a, "unmoved_image"_a, "out_image"_a, "time_start"_a,
	      "time_stop"_a, "out_dynamic_frame"_a = 0,
	      "Blur a given image based on given motion information. Write "
	      "directly in \"out_image\" in the dynamic frame "
	      "\"out_dynamic_frame\". Use \"time_start\" and \"time_stop\" to "
	      "define how motion frames are selected and weighted.");

	m.def("timeAverageMoveImageDynamic",
	      static_cast<std::unique_ptr<ImageOwned> (*)(
	          const LORMotion& lorMotion, const Image* unmovedImage,
	          const DynamicFraming& dynamicFraming)>(
	          &util::timeAverageMoveImageDynamic),
	      "lor_motion"_a, "unmoved_image"_a, "dynamic_framing"_a,
	      "Blur a given image based on the given motion information, but "
	      "follow the dynamic framing provided. The dynamic framing provided "
	      "will be used to select the motion frames to use for each blurring. "
	      "This is used for generating a 4-dimensional sensitivity image "
	      "with both a dynamic framing and rigid motion correction. Return the "
	      "resulting image.");
	m.def("timeAverageMoveImageDynamic",
	      static_cast<void (*)(const LORMotion&, const Image*, Image*,
	                           const DynamicFraming&)>(
	          &util::timeAverageMoveImageDynamic),
	      "lor_motion"_a, "unmoved_image"_a, "out_image"_a, "dynamic_framing"_a,
	      "Blur a given image based on the given motion information, but "
	      "follow the dynamic framing provided. The dynamic framing provided "
	      "will be used to select the motion frames to use for each blurring. "
	      "This is used for generating a 4-dimensional sensitivity image "
	      "with both a dynamic framing and rigid motion correction. The "
	      "resulting image will be written in \"out_image\".");

}
}

#endif

namespace yrt::util
{

template <bool PrintProgress>
std::unique_ptr<ImageOwned> timeAverageMoveImage(const LORMotion& lorMotion,
                                                 const Image* unmovedImage)
{
	auto [timeStart, timeStop] = getFullTimeRange(lorMotion);
	return timeAverageMoveImage<PrintProgress>(lorMotion, unmovedImage,
	                                           timeStart, timeStop);
}
template std::unique_ptr<ImageOwned>
    timeAverageMoveImage<true>(const LORMotion&, const Image*);
template std::unique_ptr<ImageOwned>
    timeAverageMoveImage<false>(const LORMotion&, const Image*);

template <bool PrintProgress>
void timeAverageMoveImage(const LORMotion& lorMotion, const Image* unmovedImage,
                          Image* outImage, frame_t outDynamicFrame)
{
	auto [timeStart, timeStop] = getFullTimeRange(lorMotion);
	timeAverageMoveImage<PrintProgress>(lorMotion, unmovedImage, outImage,
	                                    timeStart, timeStop, outDynamicFrame);
}
template void timeAverageMoveImage<true>(const LORMotion&, const Image*, Image*,
                                         frame_t);
template void timeAverageMoveImage<false>(const LORMotion&, const Image*,
                                          Image*, frame_t);

template <bool PrintProgress>
std::unique_ptr<ImageOwned>
    timeAverageMoveImage(const LORMotion& lorMotion, const Image* unmovedImage,
                         timestamp_t timeStart, timestamp_t timeStop)
{
	ASSERT_MSG(unmovedImage != nullptr, "Null input image given");
	const ImageParams& params = unmovedImage->getParams();
	ASSERT_MSG(params.isValid(), "Image parameters incomplete");
	ASSERT_MSG(unmovedImage->isMemoryValid(),
	           "Sensitivity image given is not allocated");

	auto outImage = std::make_unique<ImageOwned>(params);
	outImage->allocate();

	timeAverageMoveImage<PrintProgress>(lorMotion, unmovedImage, outImage.get(),
	                                    timeStart, timeStop, 0);

	return outImage;
}
template std::unique_ptr<ImageOwned>
    timeAverageMoveImage<true>(const LORMotion&, const Image*, timestamp_t,
                               timestamp_t);
template std::unique_ptr<ImageOwned>
    timeAverageMoveImage<false>(const LORMotion&, const Image*, timestamp_t,
                                timestamp_t);

template <bool PrintProgress>
void timeAverageMoveImage(const LORMotion& lorMotion, const Image* unmovedImage,
                          Image* outImage, timestamp_t timeStart,
                          timestamp_t timeStop, frame_t outDynamicFrame)
{
	ASSERT_MSG(unmovedImage != nullptr, "Null input image given");
	ASSERT_MSG(outImage->isMemoryValid(), "Output image not allocated");

	const int64_t numFrames = lorMotion.getNumFrames();
	const auto scanDuration = static_cast<float>(timeStop - timeStart);

	ProgressDisplay progress{numFrames};

	// TODO: Consider edge case:
	//  timeStart precedes the first frame's start time, therefore, we must
	//  add an *unmoved* image that has a weight scaled by:
	//  <time between timeStart and lorMotion.getStartingTimestamp(0)>/
	//  scanDuration
	//  This would be done in order to take into account the cases
	//  when the camera has been started after the scan start.

	for (frame_t frame = 0; frame < numFrames; frame++)
	{
		if constexpr (PrintProgress)
		{
			progress.progress(frame);
		}

		const timestamp_t startingTimestamp =
		    lorMotion.getStartingTimestamp(frame);
		if (startingTimestamp >= timeStart)
		{
			if (startingTimestamp > timeStop)
			{
				break;
			}
			transform_t transform = lorMotion.getTransform(frame);
			const float weight = lorMotion.getDuration(frame) / scanDuration;
			unmovedImage->transformImage(transform, *outImage, weight,
			                             outDynamicFrame);
		}
	}
}
template void timeAverageMoveImage<true>(const LORMotion&, const Image*, Image*,
                                         timestamp_t, timestamp_t, frame_t);
template void timeAverageMoveImage<false>(const LORMotion&, const Image*,
                                          Image*, timestamp_t, timestamp_t,
                                          frame_t);


template <bool PrintProgress>
std::unique_ptr<ImageOwned>
    timeAverageMoveImageDynamic(const LORMotion& lorMotion,
                                const Image* unmovedImage,
                                const DynamicFraming& dynamicFraming)
{
	ASSERT_MSG(unmovedImage != nullptr, "Null input image given");

	ImageParams params = unmovedImage->getParams();
	params.nt = dynamicFraming.getNumFrames();

	auto outImage = std::make_unique<ImageOwned>(params);
	outImage->allocate();

	timeAverageMoveImageDynamic<PrintProgress>(lorMotion, unmovedImage,
	                                           outImage.get(), dynamicFraming);

	return outImage;
}
template std::unique_ptr<ImageOwned>
    timeAverageMoveImageDynamic<true>(const LORMotion&, const Image*,
                                      const DynamicFraming&);
template std::unique_ptr<ImageOwned>
    timeAverageMoveImageDynamic<false>(const LORMotion&, const Image*,
                                       const DynamicFraming&);

template <bool PrintProgress>
void timeAverageMoveImageDynamic(const LORMotion& lorMotion,
                                 const Image* unmovedImage, Image* outImage,
                                 const DynamicFraming& dynamicFraming)
{
	ASSERT_MSG(unmovedImage != nullptr, "Null input image given");
	ASSERT_MSG(outImage != nullptr, "Output image given is null");

	const ssize_t numDynamicFrames = dynamicFraming.getNumFrames();

	ASSERT_MSG(outImage->getNumFrames() == numDynamicFrames,
	           "Output image does not have the same number of frames as the "
	           "given dynamic framing.");

	const auto numDynamicFrames_signed = static_cast<frame_t>(numDynamicFrames);

	for (frame_t dynamicFrame = 0; dynamicFrame < numDynamicFrames_signed;
	     dynamicFrame++)
	{
		const timestamp_t dynamicFrameStart =
		    dynamicFraming.getStartingTimestamp(dynamicFrame);
		const timestamp_t dynamicFrameStop =
		    dynamicFraming.getStoppingTimestamp(dynamicFrame);
		timeAverageMoveImage<PrintProgress>(lorMotion, unmovedImage, outImage,
		                                    dynamicFrameStart, dynamicFrameStop,
		                                    dynamicFrame);
	}
}
template void timeAverageMoveImageDynamic<true>(const LORMotion&, const Image*,
                                                Image*, const DynamicFraming&);
template void timeAverageMoveImageDynamic<false>(const LORMotion&, const Image*,
                                                 Image*, const DynamicFraming&);


void fillEllipse(Image& image, float value, float centerX, float centerY,
                 float semiAxisX, float semiAxisY, float angle, ssize_t zSlice,
                 frame_t dynamicFrame)
{
	const ImageParams& imgParams = image.getParams();
	const float cosA = std::cos(angle);
	const float sinA = std::sin(angle);
	const float aSq = semiAxisX * semiAxisX;
	const float bSq = semiAxisY * semiAxisY;
	const ssize_t nx = imgParams.nx;
	const ssize_t ny = imgParams.ny;

	parallelForChunked(
	    static_cast<size_t>(ny) * static_cast<size_t>(nx),
	    globals::getNumThreads(),
	    [&image, &imgParams, value, nx, zSlice, dynamicFrame, centerX, centerY,
	     cosA, sinA, aSq, bSq](size_t index, size_t /*tid*/)
	    {
		    const ssize_t iy = static_cast<ssize_t>(index) / nx;
		    const ssize_t ix = static_cast<ssize_t>(index) % nx;
		    const Vector3D pos = imgParams.indexToPosition(ix, iy, 0);
		    const float dx = pos.x - centerX;
		    const float dy = pos.y - centerY;
		    const float dxRot = dx * cosA + dy * sinA;
		    const float dyRot = -dx * sinA + dy * cosA;
		    if ((dxRot * dxRot) / aSq + (dyRot * dyRot) / bSq <= 1.0f)
		    {
			    image.getData().get({static_cast<size_t>(dynamicFrame),
			                         static_cast<size_t>(zSlice),
			                         static_cast<size_t>(iy),
			                         static_cast<size_t>(ix)}) += value;
		    }
	    });
}

void fillCircle(Image& image, float value, float centerX, float centerY,
                float radius, ssize_t zSlice, frame_t dynamicFrame)
{
	fillEllipse(image, value, centerX, centerY, radius, radius, 0.0f, zSlice,
	            dynamicFrame);
}

std::unique_ptr<ImageOwned> getCircleImage(const ImageParams& imgParams,
                                           float value, float centerX,
                                           float centerY, float radius)
{
	auto image = std::make_unique<ImageOwned>(imgParams);
	image->allocate();
	image->fill(0.0f);
	for (ssize_t iz = 0; iz < imgParams.nz; iz++)
	{
		for (frame_t it = 0; it < imgParams.nt; it++)
		{
			fillCircle(*image, value, centerX, centerY, radius, iz, it);
		}
	}
	return image;
}

std::unique_ptr<ImageOwned> getEllipseImage(const ImageParams& imgParams,
                                            float value, float centerX,
                                            float centerY, float semiAxisX,
                                            float semiAxisY, float angle)
{
	auto image = std::make_unique<ImageOwned>(imgParams);
	image->allocate();
	image->fill(0.0f);
	for (ssize_t iz = 0; iz < imgParams.nz; iz++)
	{
		for (frame_t frameIdx = 0; frameIdx < imgParams.nt; frameIdx++)
		{
			fillEllipse(*image, value, centerX, centerY, semiAxisX, semiAxisY,
			            angle, iz, frameIdx);
		}
	}
	return image;
}

void fillSphere(Image& image, float value, float centerX, float centerY,
                float centerZ, float radius, frame_t dynamicFrame)
{
	const ImageParams& imgParams = image.getParams();
	const float radiusSq = radius * radius;
	const ssize_t nx = imgParams.nx;
	const ssize_t ny = imgParams.ny;
	const ssize_t nz = imgParams.nz;
	const size_t nxy = static_cast<size_t>(ny) * static_cast<size_t>(nx);

	parallelForChunked(
	    static_cast<size_t>(nz) * nxy, globals::getNumThreads(),
	    [&image, &imgParams, value, nx, nxy, dynamicFrame, centerX, centerY,
	     centerZ, radiusSq](size_t index, size_t /*tid*/)
	    {
		    const ssize_t iz = static_cast<ssize_t>(index) / nxy;
		    const ssize_t remaining = static_cast<ssize_t>(index % nxy);
		    const ssize_t iy = remaining / nx;
		    const ssize_t ix = remaining % nx;
		    const Vector3D pos = imgParams.indexToPosition(ix, iy, iz);
		    const float dx = pos.x - centerX;
		    const float dy = pos.y - centerY;
		    const float dz = pos.z - centerZ;
		    if (dx * dx + dy * dy + dz * dz <= radiusSq)
		    {
			    image.getData().get(
			        {static_cast<size_t>(dynamicFrame), static_cast<size_t>(iz),
			         static_cast<size_t>(iy), static_cast<size_t>(ix)}) +=
			        value;
		    }
	    });
}

void fillEllipsoid(Image& image, float value, float centerX, float centerY,
                   float centerZ, float semiAxisX, float semiAxisY,
                   float semiAxisZ, frame_t dynamicFrame)
{
	const ImageParams& imgParams = image.getParams();
	const float aSq = semiAxisX * semiAxisX;
	const float bSq = semiAxisY * semiAxisY;
	const float cSq = semiAxisZ * semiAxisZ;
	const ssize_t nx = imgParams.nx;
	const ssize_t ny = imgParams.ny;
	const ssize_t nz = imgParams.nz;
	const size_t nxy = static_cast<size_t>(ny) * static_cast<size_t>(nx);

	parallelForChunked(
	    static_cast<size_t>(nz) * nxy, globals::getNumThreads(),
	    [&image, &imgParams, value, nx, nxy, dynamicFrame, centerX, centerY,
	     centerZ, aSq, bSq, cSq](size_t index, size_t /*tid*/)
	    {
		    const ssize_t iz = static_cast<ssize_t>(index) / nxy;
		    const size_t rem = index % nxy;
		    const ssize_t iy = static_cast<ssize_t>(rem) / nx;
		    const ssize_t ix = static_cast<ssize_t>(rem) % nx;
		    const Vector3D pos = imgParams.indexToPosition(ix, iy, iz);
		    const float dx = pos.x - centerX;
		    const float dy = pos.y - centerY;
		    const float dz = pos.z - centerZ;
		    if ((dx * dx) / aSq + (dy * dy) / bSq + (dz * dz) / cSq <= 1.0f)
		    {
			    image.getData().get(
			        {static_cast<size_t>(dynamicFrame), static_cast<size_t>(iz),
			         static_cast<size_t>(iy), static_cast<size_t>(ix)}) +=
			        value;
		    }
	    });
}

std::unique_ptr<ImageOwned> getSphereImage(const ImageParams& imgParams,
                                           float value, float centerX,
                                           float centerY, float centerZ,
                                           float radius)
{
	auto image = std::make_unique<ImageOwned>(imgParams);
	image->allocate();
	image->fill(0.0f);
	for (frame_t it = 0; it < imgParams.nt; it++)
	{
		fillSphere(*image, value, centerX, centerY, centerZ, radius, it);
	}
	return image;
}

std::unique_ptr<ImageOwned> getEllipsoidImage(const ImageParams& imgParams,
                                              float value, float centerX,
                                              float centerY, float centerZ,
                                              float semiAxisX, float semiAxisY,
                                              float semiAxisZ)
{
	auto image = std::make_unique<ImageOwned>(imgParams);
	image->allocate();
	image->fill(0.0f);
	for (frame_t it = 0; it < imgParams.nt; it++)
	{
		fillEllipsoid(*image, value, centerX, centerY, centerZ, semiAxisX,
		              semiAxisY, semiAxisZ, it);
	}
	return image;
}

}

