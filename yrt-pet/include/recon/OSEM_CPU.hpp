/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "recon/OSEM.hpp"

class OSEM_CPU : public OSEM
{
public:
	explicit OSEM_CPU(const Scanner& pr_scanner);
	~OSEM_CPU() override;

protected:
	// Sens Image generator driver
	void setupOperatorsForSensImgGen() override;
	void allocateForSensImgGen() override;
	std::unique_ptr<Image>
	    getLatestSensitivityImage(bool isLastSubset) override;
	void endSensImgGen() override;

	// Reconstruction driver
	void setupOperatorsForRecon() override;
	void allocateForRecon() override;
	void endRecon() override;
	void completeMLEMIteration() override;

	// Internal getters
	ImageBase* getSensImageBuffer() override;
	ProjectionData* getSensDataInputBuffer() override;
	ImageBase* getMLEMImageBuffer() override;
	ImageBase* getMLEMImageTmpBuffer() override;
	ProjectionData* getMLEMDataBuffer() override;
	ProjectionData* getMLEMDataTmpBuffer() override;

	// Common methods
	void loadBatch(int batchId, bool forRecon) override;
	void loadSubset(int subsetId, bool forRecon) override;

private:
	// For sensitivity image generation
	std::unique_ptr<Image> mp_tempSensImageBuffer;
	// For reconstruction
	std::unique_ptr<Image> mp_mlemImageTmp;
	std::unique_ptr<ProjectionData> mp_datTmp;

	int m_current_OSEM_subset;
};
