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

class ProjectorUpdater
{
public:
	ProjectorUpdater() = default;
	virtual ~ProjectorUpdater() = default;

	virtual float forwardUpdate(float weight, float* cur_img_ptr, size_t offset,
	                            frame_t dynamicFrame,
	                            size_t numVoxelPerFrame) const = 0;
	virtual void backUpdate(float value, float weight, float* cur_img_ptr,
	                        size_t offset, frame_t dynamicFrame,
	                        size_t numVoxelPerFrame, int tid) = 0;
};

class ProjectorUpdaterDefault4D : public ProjectorUpdater
{
public:
	ProjectorUpdaterDefault4D() = default;

	float forwardUpdate(float weight, float* cur_img_ptr, size_t offset,
	                    frame_t dynamicFrame,
	                    size_t numVoxelsPerFrame) const override;

	void backUpdate(float value, float weight, float* cur_img_ptr,
	                size_t offset, frame_t dynamicFrame,
	                size_t numVoxelsPerFrame, int tid) override;
};

class ProjectorUpdaterLR : public ProjectorUpdater
{
public:
	explicit ProjectorUpdaterLR(const Array2DBase<float>& pr_HBasis);

	float forwardUpdate(float weight, float* cur_img_ptr, size_t offset,
	                    frame_t dynamicFrame,
	                    size_t numVoxelsPerFrame) const override;

	void backUpdate(float value, float weight, float* cur_img_ptr,
	                size_t offset, frame_t dynamicFrame,
	                size_t numVoxelPerFrame, int tid) override;

	//	void setHBasis(const Array2D<float>& HBasis);
	void setHBasis(const Array2DBase<float>& pr_HBasis);
	//	void setHBasis(std::unique_ptr<Array2DAlias<float>> HBasis);
	Array2DAlias<float> getHBasis();
	std::unique_ptr<Array2DOwned<float>> getHBasisCopy() const;
	void setUpdateH(bool updateH);
	bool getUpdateH() const;
	void setHBasisWrite(const Array2DBase<float>& pr_HWrite);
	const Array2DAlias<float>& getHBasisWrite();
	void setCurrentImgBuffer(ImageBase* img);
	const float* getCurrentImgBuffer() const;
	void initializeWriteThread();
	void accumulateH();
	const Array3DOwned<double>& getHBasisWriteThread();

protected:
	float* m_currentImg;
	Array3DOwned<double> m_HWriteThread;
	Array2DAlias<float> mp_HBasis;  // used by forward/back math (read-only)
	Array2DAlias<float>
	    mp_HWrite;  // used only when m_updateH==true (accumulate)
	bool m_updateH = false;
	int m_rank = 1;
	int m_numDynamicFrames = 1;
};

class ProjectorUpdaterLRDualUpdate : public ProjectorUpdaterLR
{
public:
	explicit ProjectorUpdaterLRDualUpdate(const Array2DBase<float>& pr_HBasis);

	void backUpdate(float value, float weight, float* cur_img_ptr,
	                size_t offset, frame_t dynamicFrame,
	                size_t numVoxelsPerFrame, int tid) override;
};

}  // namespace yrt
