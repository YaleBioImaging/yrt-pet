/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/UniformHistogram.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"
#include "yrt-pet/operators/OperatorVarPsf.hpp"
#include "yrt-pet/recon/Corrector.hpp"
#include "yrt-pet/utils/RangeList.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#endif


namespace yrt
{

enum ImagePSFMode
{
	UNIFORM = 0,
	VARIANT
};

class OSEM
{
public:
	// ---------- Constants ----------
	static constexpr int DEFAULT_NUM_ITERATIONS = 10;
	static constexpr float DEFAULT_HARD_THRESHOLD = 1.0f;
	static constexpr float INITIAL_VALUE_MLEM = 0.125f;
	// ---------- Public methods ----------
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
	std::unique_ptr<ImageOwned> reconstruct(const std::string& out_fname);

	// Prints a summary of the parameters
	void summary() const;

	// ---------- Getters and setters ----------
	void setSensitivityHistogram(const Histogram* pp_sensitivity);
	void setGlobalScalingFactor(float globalScalingFactor);
	void setInvertSensitivity(bool invert = true);
	const Histogram* getSensitivityHistogram() const;
	const ProjectionData* getDataInput() const;
	void setDataInput(const ProjectionData* pp_dataInput);
	void addTOF(float p_tofWidth_ps, int p_tofNumStd);
	void addProjPSF(const std::string& pr_projPsf_fname);
	virtual void
	    addImagePSF(const std::string& p_imagePsf_fname,
	                ImagePSFMode p_imagePSFMode = ImagePSFMode::UNIFORM);
	void setSaveIterRanges(util::RangeList p_saveIterList,
	                       const std::string& p_saveIterPath);
	void setListModeEnabled(bool enabled);
	void setProjector(const std::string& projectorName);  // Helper
	bool isListModeEnabled() const;
	bool hasImagePSF() const;
	void enableNeedToMakeCopyOfSensImage();
	ImageParams getImageParams() const;
	void setImageParams(const ImageParams& params);
	void setRandomsHistogram(const Histogram* pp_randoms);
	void setScatterHistogram(const Histogram* pp_scatter);
	void setAttenuationImage(const Image* pp_attenuationImage);
	void setACFHistogram(const Histogram* pp_acf);
	void setHardwareAttenuationImage(const Image* pp_hardwareAttenuationImage);
	void setHardwareACFHistogram(const Histogram* pp_hardwareAcf);
	void setInVivoAttenuationImage(const Image* pp_inVivoAttenuationImage);
	void setInVivoACFHistogram(const Histogram* pp_inVivoAcf);
	virtual const Corrector& getCorrector() const = 0;
	const OperatorProjectorParams& getProjectorParams() const;
	void setProjectorParams(OperatorProjectorParams params);
	const Array2DAlias<float>& getHBasis() const;
	void setHBasis(const Array2DAlias<float>& HBasisAlias);
	OperatorProjectorParams::ProjectorUpdaterType getProjectorUpdaterType() const;
	void setProjectorUpdaterType(OperatorProjectorParams::ProjectorUpdaterType projectorUpdaterType);


	// ---------- Public members ----------
	int num_MLEM_iterations;
	int num_OSEM_subsets;
	float hardThreshold;
	int numRays;  // For Siddon only
//	OperatorProjectorBase::ProjectorUpdaterType projectorUpdaterType;
	OperatorProjector::ProjectorType projectorType;
	std::unique_ptr<OperatorProjectorParams> projectorParams;
	const Scanner& scanner;
	const Image* maskImage;
	const Image* initialEstimate;

protected:
	enum class TemporaryImageSpaceBufferType
	{
		EM_RATIO,
		PSF
	};

	// ---------- Internal Getters ----------
	auto& getBinIterators() { return m_binIterators; }
	const auto& getBinIterators() const { return m_binIterators; }

	const Image* getSensitivityImage(int subsetId) const;
	Image* getSensitivityImage(int subsetId);

	// ---------- Protected members ----------
	bool flagImagePSF;
	std::string imagePsf_fname;
	std::unique_ptr<Operator> imagePsf;
	bool flagProjPSF;
	std::string projPsf_fname;
	bool flagProjTOF;
	float tofWidth_ps;
	int tofNumStd;
	util::RangeList saveIterRanges;
	std::string saveIterPath;
	bool usingListModeInput;  // true => ListMode, false => Histogram
	std::unique_ptr<OperatorProjectorBase> mp_projector;
	bool needToMakeCopyOfSensImage;
	ImageParams imageParams;
	std::unique_ptr<ImageOwned> outImage;  // Note: This is a host image

	// ---------- Virtual pure functions ----------

	// Sens Image generator driver
	virtual void setupOperatorsForSensImgGen() = 0;
	virtual void allocateForSensImgGen() = 0;
	virtual std::unique_ptr<Image>
	    getLatestSensitivityImage(bool isLastSubset) = 0;
	virtual void computeSensitivityImage(ImageBase& destImage) = 0;
	virtual void endSensImgGen() = 0;
	virtual void setupProjectorUpdater() = 0;

	// Reconstruction driver
	virtual void setupOperatorsForRecon() = 0;
	virtual void allocateForRecon() = 0;
	virtual void computeEMUpdateImage(const ImageBase& inputImage,
	                                  ImageBase& destImage) = 0;
	virtual void endRecon() = 0;
	virtual void completeMLEMIteration() = 0;

	// Abstract Getters
	virtual ImageBase* getSensImageBuffer() = 0;
	virtual ImageBase* getMLEMImageBuffer() = 0;
	virtual ImageBase*
	    getImageTmpBuffer(TemporaryImageSpaceBufferType type) = 0;
	virtual const ProjectionData* getMLEMDataBuffer() = 0;
	virtual ProjectionData* getMLEMDataTmpBuffer() = 0;
	virtual Corrector& getCorrector() = 0;

	// Common methods
	virtual void loadSubset(int p_subsetId, bool p_forRecon) = 0;

private:
	void loadSubsetInternal(int p_subsetId, bool p_forRecon);
	void initializeForSensImgGen();
	void generateSensitivityImageForLoadedSubset();
	void generateSensitivityImagesCore(
	    bool saveOnDisk, const std::string& out_fname, bool saveOnMemory,
	    std::vector<std::unique_ptr<Image>>& sensImages);
	void initializeForRecon();

	std::vector<std::unique_ptr<BinIterator>> m_binIterators;

	const ProjectionData* mp_dataInput;

	// Histogram used to iterate on all bins for sensitivity image generation,
	//  in case it's needed
	std::unique_ptr<UniformHistogram> mp_uniformHistogram;

	std::vector<Image*> m_sensitivityImages;
	// In the specific case of ListMode reconstructions launched from Python
	std::unique_ptr<ImageOwned> mp_copiedSensitivityImage;
};
}  // namespace yrt
