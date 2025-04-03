/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "operators/Operator.hpp"

#include <vector>

class OperatorPsf : public Operator
{
public:
	OperatorPsf();
	explicit OperatorPsf(const std::string& imagePsf_fname);
	~OperatorPsf() override = default;

	virtual void readFromFile(const std::string& imagePsf_fname);

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	virtual void convolve(const Image* in, Image* out,
	                      const std::vector<float>& kernelX,
	                      const std::vector<float>& kernelY,
	                      const std::vector<float>& kernelZ) const;

	const std::vector<float>& getKernelX() const;
	const std::vector<float>& getKernelY() const;
	const std::vector<float>& getKernelZ() const;
	const std::vector<float>& getKernelXFlipped() const;
	const std::vector<float>& getKernelYFlipped() const;
	const std::vector<float>& getKernelZFlipped() const;

protected:
	std::vector<float> m_kernelX;
	std::vector<float> m_kernelY;
	std::vector<float> m_kernelZ;
	std::vector<float> m_kernelX_flipped;
	std::vector<float> m_kernelY_flipped;
	std::vector<float> m_kernelZ_flipped;

private:
	void readFromFileInternal(const std::string& imagePsf_fname);
	mutable std::vector<float> m_buffer_tmp;
};
