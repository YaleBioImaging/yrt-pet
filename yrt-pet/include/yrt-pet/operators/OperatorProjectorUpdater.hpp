/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageBase.hpp"
#include "yrt-pet/utils/Array.hpp"

#include <vector>

namespace yrt
{

class OperatorProjectorUpdater
{
public:
	OperatorProjectorUpdater() = default;
	virtual ~OperatorProjectorUpdater() = default;

//	virtual float forwardUpdate(
//	    float weight, float* cur_img_ptr,
//	    int offset_x
//	    ) const = 0;
//	virtual void backUpdate(
//	    float value, float weight, float* cur_img_ptr,
//	    int offset_x
//	    ) = 0;

	virtual float forwardUpdate(
	    float weight, float* cur_img_ptr,
	    int offset_x, int event_timeframe = 0,
	    size_t numVoxelPerFrame = 0
	) const = 0;
	virtual void backUpdate(
	    float value, float weight, float* cur_img_ptr,
	    int offset_x, int event_timeframe = 0,
	    size_t numVoxelPerFrame = 0
	    ) = 0;
};


class OperatorProjectorUpdaterDefault3D : public OperatorProjectorUpdater
{
public:
	OperatorProjectorUpdaterDefault3D() = default;

	float forwardUpdate(
	    float weight, float* cur_img_ptr,
	    int offset_x, int event_timeframe = 0,
	    size_t numVoxelPerFrame = 0
	) const override;

	void backUpdate(
	    float value, float weight, float* cur_img_ptr,
	    int offset_x, int event_timeframe = 0,
	    size_t numVoxelPerFrame = 0
	    ) override;
};

class OperatorProjectorUpdaterDefault4D : public OperatorProjectorUpdater
{
public:
	OperatorProjectorUpdaterDefault4D() = default;

	float forwardUpdate(
	    float weight, float* cur_img_ptr,
	    int offset_x, int event_timeframe,
	    size_t numVoxelPerFrame
	) const override;

	void backUpdate(
	    float value, float weight, float* cur_img_ptr,
	    int offset_x, int event_timeframe,
	    size_t numVoxelPerFrame
	    ) override;
};

class OperatorProjectorUpdaterLR : public OperatorProjectorUpdater
{
public:
	OperatorProjectorUpdaterLR() = default;

	float forwardUpdate(
	    float weight, float* cur_img_ptr,
	    int offset_x, int event_timeframe,
	    size_t numVoxelPerFrame
	) const override;

	void backUpdate(
	    float value, float weight, float* cur_img_ptr,
	    int offset_x, int event_timeframe,
	    size_t numVoxelPerFrame
	) override;

	void setHBasis(const Array2D<float>& HBasis);
	void setHBasis(const Array2DAlias<float>& HBasis);
	const Array2DAlias<float>& getHBasis() const;
	void setUpdateH(bool updateH);
	bool getUpdateH() const;

protected:
	int            m_rank = 1;
	int            m_numTimeFrames = 1;
	Array2DAlias<float> m_HBasis;
	bool           m_updateH = false;
};

}  // namespace yrt
