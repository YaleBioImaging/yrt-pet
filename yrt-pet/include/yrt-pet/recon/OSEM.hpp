/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/operators/Operator.hpp"
#include "yrt-pet/operators/ProjectorParams.hpp"
#include "yrt-pet/recon/Corrector.hpp"
#include "yrt-pet/utils/RangeList.hpp"
#include <vector>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#endif


namespace yrt
{

enum class ImagePSFMode
{
	UNIFORM = 0,
	VARIANT
};

class OSEM
{
public:
	// Constants
	static constexpr int DEFAULT_NUM_ITERATIONS = 10;
	static constexpr float DEFAULT_HARD_THRESHOLD = 1.0f;
	static constexpr float INITIAL_VALUE_MLEM = 0.125f;
	// Constructors/Destructors
	explicit OSEM(const Scanner& pr_scanner);
	virtual ~OSEM() = default;
	OSEM(const OSEM&) = delete;
	OSEM& operator=(const OSEM&) = delete;

	// Sensitivity image generation
	void generateSensitivityImages(const std::string& out_fname);
	void generateSensitivityImages(
	    std::vector<std::unique_ptr<Image>>& sensImages,
	    const std::string& out_fname);
	int getExpectedSensImagesAmount() const;

	// In case the sensitivity images were already generated
	void setSensitivityImages(const std::vector<Image*>& sensImages);
	void setSensitivityImages(
	    const std::vector<std::unique_ptr<Image>>& sensImages);
#if BUILD_PYBIND11
	void setSensitivityImages(const pybind11::list& pySensImgList);
#endif
	void setSensitivityImage(Image* sensImage, int subset = 0);

	// OSEM Reconstruction
	std::unique_ptr<Image> reconstruct(const std::string& out_fname);

	// Prints a summary of the parameters
	void summary() const;

	// Configuration of the reconstruction
	void setSensitivityHistogram(const Histogram* pp_sensitivity);
	const Histogram* getSensitivityHistogram() const;
	void setGlobalScalingFactor(float globalScalingFactor);
	void setInvertSensitivity(bool invert = true);
	const ProjectionData* getDataInput() const;
	void setDataInput(const ProjectionData* pp_dataInput);
	void addTOF(float p_tofWidth_ps, int p_tofNumStd);
	void setNumRays(int p_numRays);
	void addProjPSF(const std::string& pr_projPsf_fname);
	virtual void addImagePSF(const std::string& p_imagePsf_fname,
	                         ImagePSFMode p_imagePSFMode) = 0;
	void addImagePSF(const std::string& p_imagePsf_fname);
	void setSaveIterRanges(const util::RangeList& p_saveIterList,
	                       const std::string& p_saveIterPath);
	void setListModeEnabled(bool enabled);
	bool isListModeEnabled() const;
	void setProjector(const std::string& projectorName);  // Helper
	void setProjector(ProjectorType projectorType);
	bool hasImagePSF() const;
	void enableNeedToMakeCopyOfSensImage();  // For Python
	void setImageParams(const ImageParams& params);
	ImageParams getImageParams() const;
	ImageParams getImageParamsForSensitivityImage() const;
	void setRandomsHistogram(const Histogram* pp_randoms);
	void setScatterHistogram(const Histogram* pp_scatter);
	void setAttenuationImage(const Image* pp_attenuationImage);
	void setACFHistogram(const Histogram* pp_acf);
	void setHardwareAttenuationImage(const Image* pp_hardwareAttenuationImage);
	void setHardwareACFHistogram(const Histogram* pp_hardwareAcf);
	void setInVivoAttenuationImage(const Image* pp_inVivoAttenuationImage);
	void setInVivoACFHistogram(const Histogram* pp_inVivoAcf);
	void setProjectorUpdaterType(  // Check with YD: Should this be accessible ?
	    UpdaterType projectorUpdaterType);
	void setInitialEstimate(const Image* p_initialEstimate);
	void setMaskImage(const Image* p_maskImage);

	// Public getters
	const ProjectorParams& getProjectorParams() const;
	UpdaterType getProjectorUpdaterType() const;

	// Public members
	int num_MLEM_iterations;
	int num_OSEM_subsets;
	float hardThreshold;

protected:
	// Sens Image generator driver
	virtual void setupProjectorForSensImgGen() = 0;
	virtual void allocateForSensImgGen() = 0;
	virtual std::unique_ptr<Image>
	    generateSensitivityImageForCurrentSubset() = 0;
	virtual void endSensImgGen() = 0;

	// Reconstruction driver
	virtual void iterate();
	virtual void saveForCurrentIteration();
	virtual void setupForDynamicRecon();
	virtual void setupProjectorForRecon() = 0;
	virtual void allocateForRecon() = 0;
	virtual void loadCurrentSubset(bool p_forRecon) = 0;
	virtual void resetEMUpdateImage() = 0;
	virtual void computeEMUpdateImage() = 0;
	virtual void applyImageUpdate() = 0;
	virtual void completeSubset() = 0;
	virtual void completeMLEMIteration() = 0;
	virtual void endRecon() = 0;

	// Abstract Getters
	virtual ImageBase* getSensImageBuffer() = 0;
	virtual ImageBase* getMLEMImageBuffer() = 0;
	virtual ImageBase* getEMUpdateImageBuffer() = 0;
	virtual Corrector& getCorrector() = 0;
	virtual const Corrector& getCorrector() const = 0;

	// Internal Getters
	const Image* getSensitivityImage(int subsetId) const;
	Image* getSensitivityImage(int subsetId);
	std::vector<Constraint*> getConstraintsAsVectorOfPointers() const;
	int getCurrentOSEMSubset() const;
	int getCurrentMLEMIteration() const;
	const BinIterator* getBinIterator(int subsetId) const;

	// Protected members
	ProjectorParams projectorParams;
	const Scanner& scanner;
	bool flagImagePSF;
	std::string imagePsf_fname;
	std::unique_ptr<Operator> imagePsf;
	util::RangeList saveIterRanges;
	std::string saveIterPath;
	bool usingListModeInput;  // true => ListMode, false => Histogram
	bool needToMakeCopyOfSensImage;
	ImagePSFMode m_imagePSFMode{ImagePSFMode::UNIFORM};
	ImageParams imageParams;
	const Image* maskImage;
	const Image* initialEstimate;
	std::unique_ptr<Image> outImage;  // Note: This is a host image
	std::vector<std::unique_ptr<BinIterator>> m_binIterators;
	std::vector<std::unique_ptr<Constraint>> m_constraints;

private:
	void loadSubsetInternal(int p_subsetId, bool p_forRecon);
	void initializeForSensImgGen();
	void initializeForRecon();
	void generateSensitivityImagesCore(
	    bool saveOnDisk, const std::string& out_fname, bool saveOnMemory,
	    std::vector<std::unique_ptr<Image>>& sensImages);
	void initializeBinIterators(const ProjectionData* targetProjData);
	void initializeOutImageBuffer();
	void initializeSensImageBuffer();
	void collectConstraints();

	int m_current_OSEM_subset;
	int m_current_MLEM_iteration;
	const ProjectionData* mp_dataInput;
	std::vector<Image*> m_sensitivityImages;
	// In the specific case of ListMode reconstructions launched from Python
	std::unique_ptr<ImageOwned> mp_copiedSensitivityImage;
};
}  // namespace yrt
