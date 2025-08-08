#include "test_utils.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/ProjectionList.hpp"
#include "yrt-pet/datastruct/scanner/DetRegular.hpp"
#include "yrt-pet/utils/Assert.hpp"

#include <algorithm>

namespace yrt::util::test
{

double getRMSE(const Image& imgRef, const Image& img)
{
	const ImageParams& params = imgRef.getParams();
	const size_t numPixels =
	    static_cast<size_t>(params.nx * params.ny * params.nz);

	ASSERT(params.isSameAs(img.getParams()));

	const float* ptr_ref = imgRef.getRawPointer();
	const float* ptr = img.getRawPointer();
	double rmse = 0.0;

	for (size_t i = 0; i < numPixels; i++)
	{
		rmse += std::pow(ptr_ref[i] - ptr[i], 2.0);
	}

	rmse = std::sqrt(rmse / static_cast<double>(numPixels));

	return rmse;
}

double getRMSE(const ProjectionList& projListRef,
               const ProjectionList& projList)
{
	const size_t numBins = projListRef.count();
	ASSERT(numBins == projList.count());

	double rmse = 0.0;

	for (bin_t bin = 0; bin < numBins; ++bin)
	{
		rmse += std::pow(projList.getProjectionValue(bin) -
		                     projListRef.getProjectionValue(bin),
		                 2);
	}

	rmse = std::sqrt(rmse / static_cast<double>(numBins));

	return rmse;
}

template <bool EQUAL_NAN>
bool allclose(const ProjectionList& projValuesRef,
              const ProjectionList& projValues, float rtol, float atol)
{
	const size_t numBins = projValuesRef.count();
	ASSERT(projValues.count() == numBins);

	const float* valuesRef = projValuesRef.getRawPointer();
	const float* values = projValues.getRawPointer();

	return allclose<float, EQUAL_NAN>(valuesRef, values, numBins, rtol, atol);
}
template bool allclose<true>(const ProjectionList& projListRef,
                             const ProjectionList& projList, float rtol,
                             float atol);
template bool allclose<false>(const ProjectionList& projListRef,
                              const ProjectionList& projList, float rtol,
                              float atol);

template <bool EQUAL_NAN>
bool allclose(const Image& imageRef, const Image& image, float rtol, float atol)
{
	const auto params = imageRef.getParams();
	ASSERT(image.getParams().isSameDimensionsAs(params));
	const size_t numVoxels = params.nx * params.ny * params.nz;

	const float* valuesRef = imageRef.getRawPointer();
	const float* values = image.getRawPointer();

	return allclose<float, EQUAL_NAN>(valuesRef, values, numVoxels, rtol, atol);
}
template bool allclose<true>(const Image& imageRef, const Image& image,
                             float rtol, float atol);
template bool allclose<false>(const Image& imageRef, const Image& image,
                              float rtol, float atol);

template <typename TFloat, bool EQUAL_NAN>
bool allclose(const TFloat* valuesRef, const TFloat* values, size_t numValues,
              TFloat rtol, TFloat atol)
{
	for (bin_t i = 0; i < numValues; ++i)
	{
		const TFloat a = valuesRef[i];
		const TFloat b = values[i];

		if constexpr (EQUAL_NAN)
		{
			if (std::isnan(a) && std::isnan(b))
			{
				// If they're both NaNs, no need to do the numeric check
				continue;
			}
			if (std::isnan(a) != std::isnan(b))
			{
				// If one is NaN and the other isn't, return false immediately
				return false;
			}
		}
		else
		{
			if (std::isnan(a) || std::isnan(b))
			{
				// if EQUAL_NAN is disabled, and one of the two numbers are
				//  NaNs, return false immediately
				return false;
			}
		}

		// Numeric check (equation taken directly from NumPy documentation for
		//  numpy.allclose)
		const bool check = std::abs(a - b) <= (atol + rtol * std::abs(b));
		if (!check)
		{
			return false;
		}
	}
	return true;
}
template bool allclose<float, false>(const float* valuesRef,
                                     const float* values, size_t numValues,
                                     float rtol, float atol);
template bool allclose<double, false>(const double* valuesRef,
                                      const double* values, size_t numValues,
                                      double rtol, double atol);
template bool allclose<float, true>(const float* valuesRef, const float* values,
                                    size_t numValues, float rtol, float atol);
template bool allclose<double, true>(const double* valuesRef,
                                     const double* values, size_t numValues,
                                     double rtol, double atol);

std::unique_ptr<ImageOwned>
    makeImageWithRandomPrism(const ImageParams& params,
                             std::default_random_engine* p_engine)
{
	std::unique_ptr<std::default_random_engine> engine;
	std::default_random_engine* usedEngine;
	if (p_engine == nullptr)
	{
		unsigned int rseed = static_cast<unsigned int>(std::time(nullptr));
		engine = std::make_unique<std::default_random_engine>(rseed);
		usedEngine = engine.get();
	}
	else
	{
		usedEngine = p_engine;
	}
	constexpr float MaxPrismValue = 10.0f;

	std::uniform_int_distribution<int> prismPositionDistributionX(0, params.nx);
	std::uniform_int_distribution<int> prismPositionDistributionY(0, params.ny);
	std::uniform_int_distribution<int> prismPositionDistributionZ(0, params.nz);
	std::uniform_real_distribution<float> prismValueDistribution(0.1f,
																 MaxPrismValue);

	auto image = std::make_unique<ImageOwned>(params);
	image->allocate();
	image->setValue(0.0f);

	bool mustTryAgain = false;

	int prismBeginX, prismBeginY, prismBeginZ, prismEndX, prismEndY, prismEndZ;

	do
	{
		prismBeginX = prismPositionDistributionX(*usedEngine);
		prismBeginY = prismPositionDistributionY(*usedEngine);
		prismBeginZ = prismPositionDistributionZ(*usedEngine);
		prismEndX = prismPositionDistributionX(*usedEngine);
		prismEndY = prismPositionDistributionY(*usedEngine);
		prismEndZ = prismPositionDistributionZ(*usedEngine);

		auto [prismBeginX_n, prismEndX_n] = std::minmax(prismBeginX, prismEndX);
		prismBeginX = prismBeginX_n;
		prismEndX = prismEndX_n;
		auto [prismBeginY_n, prismEndY_n] = std::minmax(prismBeginY, prismEndY);
		prismBeginY = prismBeginY_n;
		prismEndY = prismEndY_n;
		auto [prismBeginZ_n, prismEndZ_n] = std::minmax(prismBeginZ, prismEndZ);
		prismBeginZ = prismBeginZ_n;
		prismEndZ = prismEndZ_n;

		// In case randomness made it so that the prism is of null value
		mustTryAgain = prismBeginX == prismEndX || prismBeginY == prismEndY ||
		               prismBeginZ == prismEndZ;
	} while (mustTryAgain);

	ASSERT(prismEndX > prismBeginX);
	ASSERT(prismEndY > prismBeginY);
	ASSERT(prismEndZ > prismBeginZ);

	float* image_ptr = image->getRawPointer();

	for (int i_x = prismBeginX; i_x < prismEndX; i_x++)
	{
		for (int i_y = prismBeginY; i_y < prismEndY; i_y++)
		{
			for (int i_z = prismBeginZ; i_z < prismEndZ; i_z++)
			{
				const size_t flatIdx = image->unravel(i_z, i_y, i_x);
				image_ptr[flatIdx] = prismValueDistribution(*usedEngine);
			}
		}
	}

	const float voxelSum = image->voxelSum();
	ASSERT_MSG(voxelSum > 0.0f, "Failure to generate random prism image");

	return image;
}

std::unique_ptr<yrt::Scanner> makeScanner()
{
	// Fake small scanner
	auto scanner = std::make_unique<Scanner>("FakeScanner", 200, 1, 1, 10, 200,
	                                         48, 12, 2, 8, 6, 4);
	const auto detRegular = std::make_shared<DetRegular>(scanner.get());
	detRegular->generateLUT();
	scanner->setDetectorSetup(detRegular);

	// Sanity check
	if (!scanner->isValid())
	{
		throw std::runtime_error(
		    "Unknown error in yrt::util::test::makeScanner");
	}

	return scanner;
}

}  // namespace yrt::util::test
