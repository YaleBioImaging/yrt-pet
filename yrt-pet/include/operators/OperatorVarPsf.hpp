/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "operators/Operator.hpp"

#include <vector>

struct Sigma
{
	float x, y, z;
	float sigmax, sigmay, sigmaz;
	std::vector<float> psf_kernel;
};

class OperatorVarPsf : public Operator
{
public:
	OperatorVarPsf(const ImageParams& imgParams);
	OperatorVarPsf(const std::string& imageVarPsf_fname, const ImageParams& imgParams);
	// second constructor, use sigma as input
	~OperatorVarPsf() override = default;

	void readFromFile(const std::string& imageVarPsf_fname);

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;
	template <bool IS_FWD>
	void varconvolve(const Image* in, Image* out) const;
	std::vector<Sigma> sigma_lookup;
	const float kernel_width_control = 4.0;
	void precalculateKernel(Sigma& s);

protected:
	Sigma find_nearest_sigma(const std::vector<Sigma>& sigma_lookup, float x,
	                         float y, float z) const;

private:
	ImageParams m_imageParams;
	//mutable std::vector<float> m_tempBuffer;
	const float N = 0.0634936;  // 1/sqrt(8*pi*pi*pi)
	// in the futrue, these shold be included in the header of PSF LUT
	float x_range = 200;
	float x_gap = 50;
	float y_range = 200;
	float y_gap = 50;
	float z_range = 200;  // in mm
	float z_gap = 50;
	// declare x_dim here, put the calculation in the constructor
	float x_dim = x_range / x_gap;
	mutable std::vector<float> m_tempOut;
};