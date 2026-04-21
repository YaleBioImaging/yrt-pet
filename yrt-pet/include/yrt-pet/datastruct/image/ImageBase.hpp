/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageParams.hpp"
#include "yrt-pet/operators/Variable.hpp"
#include "yrt-pet/utils/Types.hpp"

#include "nlohmann/json_fwd.hpp"
#include <string>


namespace yrt
{

class ImageBase : public Variable
{
public:
	ImageBase() = default;
	explicit ImageBase(const ImageParams& imgParams);
	~ImageBase() override = default;

	// Common functions
	float getRadius() const;
	const ImageParams& getParams() const;
	void setParams(const ImageParams& newParams);
	size_t unravel(int iz, int iy, int ix, frame_t it = 0) const;

	virtual void fill(float initValue) = 0;
	virtual void copyFromImage(const ImageBase* imSrc) = 0;
	virtual void addFirstImageToSecond(ImageBase* second) const = 0;
	virtual void applyThreshold(const ImageBase* mask_img, float threshold,
	                            float val_le_scale, float val_le_off,
	                            float val_gt_scale, float val_gt_off) = 0;
	virtual void applyThresholdBroadcast(const ImageBase* mask_img,
	                                     float threshold, float val_le_scale,
	                                     float val_le_off, float val_gt_scale,
	                                     float val_gt_off) = 0;
	virtual void writeToFile(const std::string& image_fname) const = 0;

	// EM update multiplication

	// Both the update image and the sensitivity image are 3D.
	//  If the sensitivity image is 4D, only the first frame will be read.
	//  The update image has to have the same shape as the output image.
	virtual void updateEMThresholdStatic(ImageBase* updateImg,
	                                     const ImageBase* sensImg,
	                                     float threshold) = 0;

	// The update image is 4D.
	//  If the sensitivity image is 3D, use its first frame for all the update
	//   image's frames.
	//  If the sensitivity image is 4D, use each frame of the update image and
	//   the sensitivity image to update each frame of the output image.
	//  If the update image is 3D, only its first frame will be used.
	//  The update image has to have the same shape as the output image.
	virtual void updateEMThresholdDynamic(ImageBase* updateImg,
	                                      const ImageBase* sensImg,
	                                      float threshold) = 0;

	// The update image is 4D, but the sensitivity image is 3D.
	//  The update will be scaled for each frame by the values in "sensScaling"
	//  The update image has to have the same shape as the output image.
	virtual void updateEMThresholdDynamic(ImageBase* updateImg,
	                                      const ImageBase* sensImg,
	                                      const std::vector<float>& sensScaling,
	                                      float threshold) = 0;

private:
	ImageParams m_params;
};

}  // namespace yrt
