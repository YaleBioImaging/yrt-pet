/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageBase.hpp"
#include "yrt-pet/operators/OperatorProjectorUpdaterBase.hpp"
#include "yrt-pet/utils/Array.hpp"

#include <vector>

namespace yrt
{

class OperatorProjectorUpdater : public OperatorProjectorUpdaterBase
{
public:
	OperatorProjectorUpdater() = default;
	virtual ~OperatorProjectorUpdater() = default;

	virtual float forwardUpdate(
		float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame = 0,
		size_t numVoxelPerFrame = 0
		) const = 0;
	virtual void backUpdate(float value, float weight, float* cur_img_ptr,
	                        size_t offset, frame_t dynamicFrame = 0,
	                        size_t numVoxelPerFrame = 0, int tid = 0) = 0;
};


class OperatorProjectorUpdaterDefault3D : public OperatorProjectorUpdater
{
public:
	OperatorProjectorUpdaterDefault3D() = default;

	float forwardUpdate(
		float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame = 0,
		size_t numVoxelPerFrame = 0
		) const override;

	void backUpdate(float value, float weight, float* cur_img_ptr,
	                size_t offset, frame_t dynamicFrame = 0,
	                size_t numVoxelPerFrame = 0, int tid = 0) override;
};

class OperatorProjectorUpdaterDefault4D : public OperatorProjectorUpdater
{
public:
	OperatorProjectorUpdaterDefault4D() = default;

	float forwardUpdate(
		float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame,
		size_t numVoxelPerFrame
		) const override;

	void backUpdate(float value, float weight, float* cur_img_ptr,
	                size_t offset, frame_t dynamicFrame,
	                size_t numVoxelPerFrame, int tid = 0) override;
};

class OperatorProjectorUpdaterLR : public OperatorProjectorUpdater
{
public:
	OperatorProjectorUpdaterLR(const Array2DBase<float>& pr_HBasis);

	float forwardUpdate(
		float weight, float* cur_img_ptr,
		size_t offset, frame_t dynamicFrame,
		size_t numVoxelPerFrame
		) const override;

	void backUpdate(float value, float weight, float* cur_img_ptr,
	                size_t offset, frame_t dynamicFrame,
	                size_t numVoxelPerFrame, int tid = 0) override;

//	void setHBasis(const Array2D<float>& HBasis);
	void setHBasis(const Array2DBase<float>& pr_HBasis);
//	void setHBasis(std::unique_ptr<Array2DAlias<float>> HBasis);
	const Array2DAlias<float>& getHBasis() const;
	std::unique_ptr<Array2D<float>> getHBasisCopy() const;
	void setUpdateH(bool updateH);
	bool getUpdateH() const;
	void setHBasisWrite(const Array2DBase<float>& pr_HWrite);
	const Array2DAlias<float>& getHBasisWrite();
	void setCurrentImgBuffer(ImageBase* img);
	const float* getCurrentImgBuffer() const;
	void initializeWriteThread();
	void accumulateH();

protected:
	float* m_currentImg;
	Array3D<float> m_HWriteThread;
	Array2DAlias<float> mp_HBasis;  // used by forward/back math (read-only)
	Array2DAlias<float> mp_HWrite;  // used only when m_updateH==true (accumulate)
	bool m_updateH = false;
	int m_rank = 1;
	int m_numDynamicFrames = 1;
};

class OperatorProjectorUpdaterLRDualUpdate : public OperatorProjectorUpdaterLR
{
public:
	OperatorProjectorUpdaterLRDualUpdate(const Array2DBase<float>& pr_HBasis);

	void backUpdate(float value, float weight, float* cur_img_ptr,
	                size_t offset, frame_t dynamicFrame,
	                size_t numVoxelPerFrame, int tid = 0) override;

};

}  // namespace yrt
