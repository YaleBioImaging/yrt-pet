/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "operators/Operator.hpp"

#include <string>
#include <vector>


/* **************************************************************************************
 * Def : Class that define the basic work flow of a warper tool used for image
 *motion
 *       correction in tomographic reconstruction model.
 *
 * Hypothesis:
 *		- All transforms are relative to a reference motion frame.
 *		- m_motionFrameUsed is only used internaly. It is not used to check if
 *        an image should be warped or inverse warped.
 *
 * TODO:
 *		- Implement merging of motion frame? Currently, it could be done by
 *		  pre-processing if we consider only merging connexe frame and give the
 *		  reconstruction reconstructor only merged information of the motions
 *        frames.
 *		  However I am not sure it could be done for not connexe frame, right
 *        now.
 *		- Evaluate cumulative deformation and cumulative inverse deformation.
 *		- Set verbosity options.
 * *************************************************************************************/

class ImageParams;

class ImageWarperTemplate : public Variable
{
public:
	// List of public methods.
	ImageWarperTemplate();
	~ImageWarperTemplate() override;
	/*
	 * Def.: Instantiate the object with the basic information of the image.
	 * @imDim: Number of pixel of the image in each dimension.
	 * @imSize: Size of the image in each dimension.
	 */
	void setImageHyperParam(const std::vector<int>& imDim,
	                        const std::vector<float>& imSize);
	void setImageHyperParam(const ImageParams& img_params);
	void setImageParams(const ImageParams& img_params);
	/*
	 * Def.: Instantiate the object with the basic information of the motion.
	 * @numberOfFrame: Number of frame.
	 */
	void setMotionHyperParam(int numberOfFrame);
	/*
	 * Def.: Extract and set the frames parameters from a file.
	 * @paramFileName: Path to the file with the parameters.
	 */
	void setFramesParamFromFile(std::string paramFileName);
	void readFromFile(std::string paramFileName);
	/*
	 * Def.: Instantiate the object by using previously defined basic
	 * information of
	 *       the image and the motion.
	 */
	void initParamContainer();
	/*
	 * Def.: Set the pointer to the image at the reference frame.
	 * @image: The pointer to the image.
	 */
	void setRefImage(const Image* image);
	/*
	 * Def.: Warp the reference image to the specified frame Id. If the
	 *specified
	 *       Id is the refernce one, then copy the image of the reference frame.
	 * @image: Pointer to where we want to store the warped image. It is assumed
	 *		that the image is already initialized in the same image space as the
	 *		reference image.
	 * @frameId: Id of the frame to which the reference image is deformed to.
	 */
	void warpRefImage(Image* image, int frameId) const;
	/*
	 * Def.: Warp the provided image using the transform that link the specified
	 *       state to the reference state.
	 * @image: Pointer of the image to warp. It is assumed that image is already
	 *		initialized in the same image space as the reference image.
	 * @frameId: Id of the motion frame to which the specified image is warped
	 *from.
	 */
	void warpImageToRefFrame(Image* image, int frameId) const;

	/*
	 * Def.: Warp the provided image using the transformation of each frame
	 *toward
	 *       the reference frame, for all the frames, and merge the results.
	 * @image: Pointer of the image to which we want to apply the global warping
	 *		toward the reference frame. It is assumed that image is already
	 *		initialized in the same image space as the reference image.
	 */
	void computeGlobalWarpToRefFrame(Image* image, bool writeFileSteps);

	/*
	 * Def.: Exclude the specified frame Id from the list of those used.
	 * @frameId: Id of the motion frame to which the specified image is warped
	 * from.
	 */
	void deactivateFrame(int frameId);
	/*
	 * Def.: Set the parameters of the reference frame.
	 * @frameId: Id of the reference frame.
	 * @frameTimeBinStart: The start time of the reference frame.
	 * @frameWeight: The weight of the reference frame.
	 */
	void setReferenceFrameParam(int frameId, float frameTimeBinStart,
	                            float frameWeight);

	/*
	 * Def.: Set the parameters of a specified frame Id.
	 * @frameId: Id of the frame.
	 * @warpParam: The warp parameters, relative to the reference frame, of the
	 *		current frame.
	 * @frameTimeBinStart: The start time of the current frame.
	 * @frameWeight: The weight of the current frame.
	 * TODO:
	 *		- Merge setFrameParam and setReferenceFrameParam?
	 */
	void setFrameParam(int frameId, const std::vector<double>& warpParam,
	                   float frameTimeBinStart, float frameWeight);
	/*
	 * Def.: Reset all the variables of this class.
	 */
	virtual void reset();

	/*
	 * Def.: Indicate if the frame is used or not.
	 * @frameId: Motion frame Id.
	 */
	bool isFrameUsed(int frameId);
	/*
	 * Def.: Give a copy of the variable m_imNbVoxel.
	 */
	std::vector<int> getImNbVoxel();
	/*
	 * Def.: Give a copy of the variable m_referenceFrameId.
	 */
	unsigned int getReferenceFrameId();
	/*
	 * Def.: Give a copy of the variable m_numberOfFrame.
	 */
	int getNumberOfFrame();
	/*
	 * Def.: Give the start time of the specified frame.
	 * @frameId: The Id of the frame of interest.
	 */
	float getFrameStartTime(int frameId);


protected:
	// List of constants.
	static constexpr unsigned int m_imNbOfDimension = 3;

	// User modifiable parameters.
	// Number of voxel in the image space in each dimension.
	std::vector<int> m_imNbVoxel;
	// Size of the image in each dimension.
	std::vector<float> m_imSize;
	// The center of rotation.
	std::vector<float> m_rotCenter;
	// Id of the reference frame.
	int m_referenceFrameId;
	// Total number of motion frame.
	int m_numberOfFrame;
	// Indicate which motion frames are used.
	std::vector<bool> m_motionFrameUsed;
	// Indicate the weight of each motion frame such that
	// histogram^k ~ m_weightOfFrame[k] * systemMatrix warpMatrix^k m_refImage
	// Normaly, should sum at one but I am not sure if it should be enforced.
	// Also, only used for the sensitivity matrix computation since applying it
	// for
	// the forward and backward projection would negate it (when scatter and
	// random
	// are neglected).
	std::vector<float> m_weightOfFrame;
	// The starting time of the frames time bin.
	std::vector<float> m_frameTimeBinStart;
	// Pointer to the reference frame image.
	const Image* mp_refImage;
	// Name of the warp tool current used.
	std::string m_warpMode;
	// Verbosity level of this class.
	int m_verboseLevel;
	// Bool that indicate if m_weightOfFrame is applied on warp() and invWarp().
	// TODO: Impose a procedure on when the flag can be modified or not.
	bool m_applyWeightToWarp;
	// State variables.
	bool m_imHyperInit;
	bool m_motionHyperInit;
	bool m_warperParamContainerInit;
	bool m_refFrameDefined;
	std::vector<bool> m_frameWarpParamDefined;

	// List of protected functions.
	/*
	 * Def.: Check if the specified frame Id is a valid candidate to define.
	 * @frameId: Id of the frame.
	 * @isRef: Boolean that indicate if this method is called for the referance
	 *		frame case.
	 */
	void isFrameIdValidToDefine(int frameId, bool isRef = false);
	/*
	 * Def.: If greater than zero, the weight of the specified frame is setted.
	 * @weight: The weight of the frame.
	 * @frameId: Id of the frame.
	 */
	void setFrameWeight(float weight, int frameId);
	/*
	 * Def.: If valid, the starting time of the specified frame is setted.
	 * @timeBinStart: The starting time of the frame.
	 * @frameId: Id of the frame.
	 */
	void setFrameTimeBinStart(float timeBinStart, int frameId);
	/*
	 * Def.: Extract the frames parameters from the provided file. It only check
	 *       that the file provides the correct number of parameters.
	 * @paramFileName: Path to the file with the parameters.
	 * @frameParam: Structure where the extracted frames parameters will be
	 * stored.
	 */
	int extractFramesParamFromFile(
	    std::string paramFileName,
	    std::vector<std::vector<double>>& frameParam);
	/*
	 * Def.: Split a string following the delimiter given.
	 * @stringTpParse: String to split.
	 * @delimiter: Delimiter used to split the string.
	 * TODO: Merge the two with template?
	 */
	std::vector<std::string> splitStringIntoVector(std::string stringToParse,
	                                               std::string delimiter);
	std::vector<double> splitStringIntoVectorOfDouble(std::string stringToParse,
	                                                  std::string delimiter);
	/*
	 * Def.: Create the rotation matrix corresponding to the angle and rotation
	 * axis
	 *       given. The matrix is flatten row-wise and than column-wise.
	 * @_angle: The rotation angle in degree.
	 * @rotationAxis: The rotation axis.
	 */
	std::vector<float> createRotationMatrix(float angle,
	                                        std::vector<float> rotationAxis);
	/*
	 * Def.: Convert a quaternion, in format [qw, qx, qy, qz], into its
	 *       corresponding rotation matrix. The matrix is flatten row-wise and
	 * than
	 *       column-wise.
	 * @quaternion: The quaternion to convert into a matrix.
	 */
	std::vector<double>
	    convertQuaternionToRotationMatrix(std::vector<double> quaternion);

	// The methods to be defined in the child class.
	virtual void initWarpModeSpecificParameters() = 0;
	virtual void setFrameWarpParameters(int frameId,
	                                    const std::vector<double>& param) = 0;
	virtual void warp(Image* image, int frameId) const = 0;
	virtual void inverseWarp(Image* image, int frameId) const = 0;
};

class OperatorWarpRefImage : public Operator
{
public:
	OperatorWarpRefImage(int p_frameId = 0) : m_frameId(p_frameId) {}
	void setFrameId(int p_frameId) { m_frameId = p_frameId; }
	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

protected:
	int m_frameId;
};
