/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/ProjectionData.hpp"
#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "recon/Corrector_GPU.cuh"
#include "recon/OSEM.hpp"
#include "recon/OSEMUpdater_GPU.cuh"
#include "utils/GPUStream.cuh"


class OSEM_GPU : public OSEM
{
public:
	explicit OSEM_GPU(const Scanner& pr_scanner);
	~OSEM_GPU() override;

	// Getters for internal objects
	const Corrector& getCorrector() const override;
	Corrector& getCorrector() override;
	const Corrector_GPU& getCorrector_GPU() const;
	Corrector_GPU& getCorrector_GPU();
	OperatorProjectorDevice* getProjector() const;
	const cudaStream_t* getAuxStream() const;
	const cudaStream_t* getMainStream() const;

	// Sens Image generator driver
	void setupOperatorsForSensImgGen() override;
	void allocateForSensImgGen() override;
	std::unique_ptr<Image>
	    getLatestSensitivityImage(bool isLastSubset) override;
	void computeSensitivityImage(ImageBase& destImage) override;
	void endSensImgGen() override;

	// Reconstruction driver
	void setupOperatorsForRecon() override;
	void allocateForRecon() override;
	void endRecon() override;
	void completeMLEMIteration() override;
	void computeEMUpdateImage(const ImageBase& inputImage,
	                          ImageBase& destImage) override;

	// Internal getters
	ImageBase* getSensImageBuffer() override;
	ImageBase* getMLEMImageBuffer() override;
	ImageBase* getImageTmpBuffer(TemporaryImageSpaceBufferType type) override;
	const ProjectionData* getMLEMDataBuffer() override;
	ProjectionData* getMLEMDataTmpBuffer() override;
	int getNumBatches(int subsetId, bool forRecon) const;
	int getCurrentOSEMSubset() const;
	const ProjectionDataDeviceOwned* getSensitivityDataDeviceBuffer() const;
	ProjectionDataDeviceOwned* getSensitivityDataDeviceBuffer();
	const ProjectionDataDeviceOwned* getMLEMDataDeviceBuffer() const;
	ProjectionDataDeviceOwned* getMLEMDataDeviceBuffer();
	const ProjectionDataDeviceOwned* getMLEMDataTmpDeviceBuffer() const;
	ProjectionDataDeviceOwned* getMLEMDataTmpDeviceBuffer();

	// Common methods
	void loadSubset(int subsetId, bool forRecon) override;
	void addImagePSF(const std::string& p_imagePsf_fname,
	                 ImagePSFMode p_imagePSFMode) override;

private:
	std::unique_ptr<ImageDeviceOwned> mpd_sensImageBuffer;
	std::unique_ptr<ImageDeviceOwned> mpd_mlemImage;
	std::unique_ptr<ImageDeviceOwned> mpd_mlemImageTmpEMRatio;
	std::unique_ptr<ImageDeviceOwned> mpd_imageTmpPsf;

	// Buffer used for sensitivity image generation
	std::unique_ptr<ProjectionDataDeviceOwned> mpd_tempSensDataInput;
	// Buffers used for reconstruction
	std::unique_ptr<ProjectionDataDeviceOwned> mpd_dat;
	std::unique_ptr<ProjectionDataDeviceOwned> mpd_datTmp;

	std::unique_ptr<Corrector_GPU> mp_corrector;
	std::unique_ptr<OSEMUpdater_GPU> mp_updater;

	int m_current_OSEM_subset;

	GPUStream m_mainStream;
	GPUStream m_auxStream;
};
