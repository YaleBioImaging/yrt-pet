#include "test_utils.hpp"

#include "datastruct/image/Image.hpp"
#include "datastruct/projection/ProjectionList.hpp"
#include "datastruct/scanner/DetRegular.hpp"
#include "utils/Assert.hpp"

double TestUtils::getRMSE(const Image& imgRef, const Image& img)
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

double TestUtils::getRMSE(const ProjectionList& projListRef,
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

std::unique_ptr<Scanner> TestUtils::makeScanner()
{
	// Fake small scanner
	auto scanner = std::make_unique<Scanner>("FakeScanner", 200, 1, 1, 10, 200,
	                                         24, 9, 2, 4, 6, 4);
	const auto detRegular = std::make_shared<DetRegular>(scanner.get());
	detRegular->generateLUT();
	scanner->setDetectorSetup(detRegular);

	// Sanity check
	if (!scanner->isValid())
	{
		throw std::runtime_error("Unknown error in TestUtils::makeScanner");
	}

	return scanner;
}
