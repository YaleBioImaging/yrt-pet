/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/recon/Corrector.hpp"

namespace yrt
{

class Corrector_CPU : public Corrector
{
public:
	explicit Corrector_CPU(const Scanner& pr_scanner);

	// Return sensitivity*attenuation. This computes the attenuation factor
	//  on-the-fly instead of using the precomputed ProjectionList. This is
	//  useful only for the sensitivity image generation
	float getMultiplicativeCorrectionFactor(const ProjectionData& measurements,
	                                        bin_t binId) const;

};
}  // namespace yrt
