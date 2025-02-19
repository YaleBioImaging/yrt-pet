/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/SparseProjection.hpp"

#include "datastruct/image/Image.hpp"
#include "datastruct/projection/SparseHistogram.hpp"
#include "datastruct/projection/UniformHistogram.hpp"
#include "operators/OperatorProjector.hpp"
#include "utils/Globals.hpp"
#include "utils/ProgressDisplay.hpp"

#include "omp.h"

namespace Util
{
	void forwProjectToSparseHistogram(const Image& sourceImage,
	                                  const OperatorProjector& projector,
	                                  SparseHistogram& sparseHistogram)
	{
		// Iterate over all LORs
		const auto uniformHistogram =
		    std::make_unique<UniformHistogram>(projector.getScanner());
		const size_t numBins = uniformHistogram->count();

		SparseHistogram* sparseHistogram_ptr = &sparseHistogram;
		const UniformHistogram* uniformHistogram_ptr = uniformHistogram.get();
		const Image* sourceImage_ptr = &sourceImage;
		const OperatorProjector* projector_ptr = &projector;

		ProgressDisplay progress(numBins, 5);

		for (bin_t bin = 0; bin < numBins; ++bin)
		{
			progress.progress(bin);

			const det_pair_t detPair =
			    uniformHistogram_ptr->getDetectorPair(bin);
			const ProjectionProperties projectionProperties =
			    uniformHistogram_ptr->getProjectionProperties(bin);

			const float projValue = projector_ptr->forwardProjection(
			    sourceImage_ptr, projectionProperties);

			if (std::abs(projValue) > SMALL)
			{
				sparseHistogram_ptr->accumulate(detPair, projValue);
			}
		}
	}
}  // namespace Util
