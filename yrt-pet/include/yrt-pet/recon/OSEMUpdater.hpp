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

class ProjectionData;
class OSEM;
class OSEM_CPU;

#if BUILD_CUDA
class OSEM_GPU;
#endif

class OSEMUpdater
{
public:
	OSEMUpdater() = default;
	virtual ~OSEMUpdater() = default;

	/*
	 * This function computes the sensitivity image to use as denominator for
	 * the OSEM equation. It uses the current projector of the OSEM object,
	 * which stores a binIterator, which is chosen based on the subset. This
	 * means that this function has be called once for every subset (or every
	 * sensitivity image) in Histogram mode or only once in ListMode. (The
	 * sensitivity image generated will not have the PSF applied to it)
	 */
	virtual void computeSensitivityImage(ImageBase& destImage) const = 0;

	/*
	 * This function computes the image that will be used in the EM update
	 * (after the PSF forward has been applied and before the PSF backwards is
	 * to be applied)
	 */
	virtual void computeEMUpdateImage(const ImageBase& inputImage,
	                          ImageBase& destImage) const = 0;

};

std::unique_ptr<OSEMUpdater> make_osem_updater(OSEM* pp_osem);

}