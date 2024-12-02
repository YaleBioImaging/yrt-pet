
/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "datastruct/image/Image.hpp"

void checkTwoImages(const Image& img1, const Image& img2)
{
	const ImageParams& params1 = img1.getParams();
	const ImageParams& params2 = img2.getParams();
	REQUIRE(params1.isSameAs(params2));

	const float* i1_ptr = img1.getRawPointer();
	const float* i2_ptr = img2.getRawPointer();
	const int numVoxels = params1.nx * params1.ny * params1.nz;
	for (int i = 0; i < numVoxels; i++)
	{
		CHECK(i1_ptr[i] == Approx(i2_ptr[i]));
	}
}

TEST_CASE("image-readwrite", "[image]")
{
	ImageParams params1{50, 50, 50, 50, 50, 50};
	ImageOwned img1{params1};
	img1.allocate();
	img1.writeToFile("tmp.nii");

	ImageOwned img2{"tmp.nii"};
	ImageParams params2 = img2.getParams();
	REQUIRE(params2.isSameAs(params1));

	checkTwoImages(img1, img2);

	params1.serialize("tmp_params.json");
	ImageParams params3{"tmp_params.json"};
	REQUIRE(params1.isSameAs(params3));

	ImageOwned img3{params3, "tmp.nii"};
	REQUIRE(params1.isSameAs(img3.getParams()));

	checkTwoImages(img1, img3);

	std::remove("tmp.nii");
	std::remove("tmp_params.json");
}
