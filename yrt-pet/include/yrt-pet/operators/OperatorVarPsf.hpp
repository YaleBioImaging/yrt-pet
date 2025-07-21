/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/operators/Operator.hpp"

#include <vector>

namespace yrt
{

class ConvolutionKernel
{
public:
	using KernelArray = Array3D<float>;
	// Sizes are assumed to be odd
	size_t getHalfSizeX() const;
	size_t getHalfSizeY() const;
	size_t getHalfSizeZ() const;
	const KernelArray& getArray() const;
	ConvolutionKernel(const ConvolutionKernel& p_kernel);

protected:
	ConvolutionKernel() = default;

	KernelArray psfKernel;
};

class ConvolutionKernelGaussian : public ConvolutionKernel
{
public:
	ConvolutionKernelGaussian(float p_sigmaX, float p_sigmaY, float p_sigmaZ,
	                          float p_nStdX, float p_nStdY, float p_nStdZ,
	                          const ImageParams& pr_imageParams);
	void setSigmas(float p_sigmaX, float p_sigmaY, float p_sigmaZ,
	               float p_nStdX, float p_nStdY, float p_nStdZ,
	               const ImageParams& pr_imageParams);

private:
	float m_sigmaX, m_sigmaY, m_sigmaZ;
	float m_nStdX, m_nStdY, m_nStdZ;
};

class OperatorVarPsf : public Operator
{
public:
	using ConvolutionKernelCollection =
	    std::vector<std::unique_ptr<ConvolutionKernel>>;

	OperatorVarPsf(const ImageParams& p_imageParams);
	OperatorVarPsf(const std::string& imageVarPsf_fname,
	               const ImageParams& p_imageParams);
	~OperatorVarPsf() override = default;

	void readFromFile(const std::string& imageVarPsf_fname);

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;
	template <bool IS_FWD>
	void varconvolve(const Image* in, Image* out) const;
	void setRangeAndGap(float xRange, float xGap, float yRange, float yGap,
	                    float zRange, float zGap);
	void setKernelCollection(const ConvolutionKernelCollection& p_kernelLUT);

protected:
	const ConvolutionKernel& findNearestKernel(float x, float y, float z) const;

private:
	ConvolutionKernelCollection m_kernelLUT;
	ImageParams m_imageParams;
	// Ranges and gaps in mm
	float m_xRange;
	float m_xGap;
	float m_yRange;
	float m_yGap;
	float m_zRange;
	float m_zGap;
};

}  // namespace yrt
