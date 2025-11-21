/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/recon/Corrector_CPU.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"
#include "yrt-pet/datastruct/projection/BinFilter.hpp"
#include "yrt-pet/recon/OSEMUpdater.hpp"

namespace yrt
{

class ProjectionData;
class OSEM_CPU;
//TODO NOW (ALMOST): remove dependcy OSEMUpdater and useless overload
class OSEMUpdater_CPU : public OSEMUpdater
{
public:
	explicit OSEMUpdater_CPU(OSEM_CPU* pp_osem);
	~OSEMUpdater_CPU() override = default;

	/*
	 * This function computes the sensitivity image to use as denominator for
	 * the OSEM equation. It uses the current projector of the OSEM object,
	 * which stores a binIterator, which is chosen based on the subset. This
	 * means that this function has be called once for every subset (or every
	 * sensitivity image) in Histogram mode or only once in ListMode. (The
	 * sensitivity image generated will not have the PSF applied to it)
	 */
	void computeSensitivityImage(ImageBase& destImageBase) const override;
	void computeSensitivityImage(Image& destImage) const;

	/*
	 * This function computes the image that will be used in the EM update
	 * (after the PSF forward has been applied and before the PSF backwards is
	 * to be applied)
	 */
	void computeEMUpdateImage(const ImageBase& inputImage,
	                          ImageBase& destImage) const override;
	void computeEMUpdateImage(const Image& inputImage, Image& destImage) const;

private:
	OSEM_CPU* mp_osem;
};

}  // namespace yrt
