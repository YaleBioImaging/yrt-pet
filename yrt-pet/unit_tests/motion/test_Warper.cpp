/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/image/Image.hpp"
#include "motion/ImageWarperMatrix.hpp"

#include "catch.hpp"
#include <cmath>
#include <vector>

float l1DistBetweenTwoImage(Image* im1, Image* im2)
{
	float l1Dist = 0.0f;
	const ImageParams& im1_params = im1->getParams();
	const ImageParams& im2_params = im2->getParams();
	if ((im1_params.nx != im2_params.nx) || (im1_params.ny != im2_params.ny) ||
	    (im1_params.nz != im2_params.nz))
	{
		// Random value to show that the image are not similar.
		return 3.14159;
	}
	Array3DAlias<float> x1 = im1->getArray();
	Array3DAlias<float> x2 = im2->getArray();
	for (int i = 0; i < im1_params.nx; i++)
	{
		for (int j = 0; j < im1_params.ny; j++)
		{
			for (int k = 0; k < im1_params.nz; k++)
			{
				l1Dist += std::abs(x1[k][j][i] - x2[k][j][i]);
			}
		}
	}
	return l1Dist;
}


TEST_CASE("Warper-basic_iso_unWeighted", "[warper]")
{
	// Setup of the test.
	std::vector<int> imDim{10, 10, 10};
	std::vector<float> imSize{10, 10, 10};

	ImageParams img_params(imDim[0], imDim[1], imDim[2], imSize[0], imSize[1],
	                       imSize[2], 0.0, 0.0, 0.0);
	auto refImage = std::make_unique<ImageOwned>(img_params);
	refImage->allocate();
	refImage->setValue(0.0);
	Array3DAlias<float> x_ref = refImage->getArray();
	x_ref[2][4][2] = 1.0;

	int nbFrame = 4;
	int refFrameId = 1;
	float refFrameTimeStart = 0.33;
	float refFrameWeight = 0.25;

	float frame_0_TimeStart = 0.0;
	float frame_0_Weight = 0.40;
	// Translation of one pixel in each dimension.
	std::vector<double> frame_0_warpParm{1.0, 0.0, 0.0, 0.0, 0.5, 2.0, 1.75};

	float frame_2_TimeStart = 0.66;
	float frame_2_Weight = 0.30;
	// Only rotation.
	std::vector<double> frame_2_warpParm{
	    sqrt(2.0) / 2.0, 0.0, 0.0, sqrt(2.0) / 2.0, 0.0, 0.0, 0.0};

	float frame_3_TimeStart = 1.0;
	float frame_3_Weight = 0.05;
	// Rotation and translation.
	std::vector<double> frame_3_warpParm{
	    sqrt(2.0) / 2.0, 0.0, 0.0, sqrt(2.0) / 2.0, 0.5, 2.0, 1.75};

	ImageWarperMatrix warper;
	warper.setImageHyperParam(imDim, imSize);
	warper.setMotionHyperParam(nbFrame);
	warper.initParamContainer();
	warper.setReferenceFrameParam(refFrameId, refFrameTimeStart,
	                              refFrameWeight);

	warper.setFrameParam(0, frame_0_warpParm, frame_0_TimeStart,
	                     frame_0_Weight);
	warper.setFrameParam(2, frame_2_warpParm, frame_2_TimeStart,
	                     frame_2_Weight);
	warper.setFrameParam(3, frame_3_warpParm, frame_3_TimeStart,
	                     frame_3_Weight);

	warper.setRefImage(refImage.get());

	// Where the results is saved.
	auto warpedImage = std::make_unique<ImageOwned>(img_params);
	warpedImage->allocate();
	warpedImage->setValue(0.0);

	SECTION("warp_refImage_unchanged")
	{
		warper.warpRefImage(warpedImage.get(), refFrameId);

		float l1Dist;
		l1Dist = l1DistBetweenTwoImage(refImage.get(), warpedImage.get());

		REQUIRE(0.0f == l1Dist);
	}

	SECTION("warp_translation_integer")
	{
		warper.warpRefImage(warpedImage.get(), 0);

		auto compImage = std::make_unique<ImageOwned>(img_params);
		compImage->allocate();
		compImage->setValue(0.0);
		auto compImage_arr = compImage->getArray();
		compImage_arr[3][6][2] = 0.5 * 0.25;
		compImage_arr[3][6][3] = 0.5 * 0.25;
		compImage_arr[4][6][2] = 0.5 * 0.75;
		compImage_arr[4][6][3] = 0.5 * 0.75;

		float l1Dist;
		l1Dist = l1DistBetweenTwoImage(compImage.get(), warpedImage.get());

		REQUIRE(0.0f == Approx(l1Dist));
	}

	SECTION("warp_rotation")
	{
		warper.warpRefImage(warpedImage.get(), 2);

		auto compImage = std::make_unique<ImageOwned>(img_params);
		compImage->allocate();
		compImage->setValue(0.0);
		auto compImage_arr = compImage->getArray();
		compImage_arr[2][2][5] = 1.0;

		float l1Dist;
		l1Dist = l1DistBetweenTwoImage(compImage.get(), warpedImage.get());

		REQUIRE(l1Dist < 1e-6);
	}

	SECTION("warp_rotation_translation")
	{
		warper.warpRefImage(warpedImage.get(), 3);

		auto compImage = std::make_unique<ImageOwned>(img_params);
		compImage->allocate();
		compImage->setValue(0.0);
		auto compImage_arr = compImage->getArray();
		compImage_arr[3][2][3] = 0.5 * 0.25;
		compImage_arr[3][3][3] = 0.5 * 0.25;
		compImage_arr[4][2][3] = 0.5 * 0.75;
		compImage_arr[4][3][3] = 0.5 * 0.75;

		float l1Dist;
		l1Dist = l1DistBetweenTwoImage(compImage.get(), warpedImage.get());

		REQUIRE(l1Dist < 1e-6);
	}

	// Where the results is saved.
	auto invWarpedImage = std::make_unique<ImageOwned>(img_params);
	invWarpedImage->allocate();
	auto invWarpedImage_arr = invWarpedImage->getArray();

	SECTION("invWarp_refFrame_unchanged")
	{
		invWarpedImage->copyFromImage(refImage.get());
		warper.warpImageToRefFrame(invWarpedImage.get(), refFrameId);

		float l1Dist;
		l1Dist = l1DistBetweenTwoImage(refImage.get(), invWarpedImage.get());

		REQUIRE(l1Dist < 1e-6);
	}

	SECTION("invWarp_translation")
	{
		invWarpedImage->setValue(0.0);
		invWarpedImage_arr[2][4][2] = 1.0;
		warper.warpImageToRefFrame(invWarpedImage.get(), 0);

		auto compImage = std::make_unique<ImageOwned>(img_params);
		compImage->allocate();
		compImage->setValue(0.0);
		auto compImage_arr = compImage->getArray();
		compImage_arr[0][2][1] = 0.5 * 0.75;
		compImage_arr[1][2][1] = 0.5 * 0.25;
		compImage_arr[0][2][2] = 0.5 * 0.75;
		compImage_arr[1][2][2] = 0.5 * 0.25;

		float l1Dist;
		l1Dist = l1DistBetweenTwoImage(invWarpedImage.get(), compImage.get());

		REQUIRE(l1Dist < 1e-6);
	}

	SECTION("invWarp_rotation")
	{
		// rework on it
		invWarpedImage->setValue(0.0);
		invWarpedImage_arr[2][2][5] = 1.0;
		warper.warpImageToRefFrame(invWarpedImage.get(), 2);

		auto compImage = std::make_unique<ImageOwned>(img_params);
		compImage->allocate();
		auto compImage_arr = compImage->getArray();
		compImage_arr[2][4][2] = 1.0f;

		float l1Dist;
		l1Dist = l1DistBetweenTwoImage(invWarpedImage.get(), compImage.get());

		REQUIRE(l1Dist < 1e-6f);
	}

	// Not ready!!!
	// SECTION("invWarp_rotation_translation")
	// {
	// 	imageToInvWarp[2][0][1] = 0.398807;
	// 	imageToInvWarp[2][1][1] = 0.201194;
	// 	imageToInvWarp[3][0][1] = 0.227549;
	// 	imageToInvWarp[3][1][1] = 0.114273;
	// 	warper.warpImageToRefFrame(imageToInvWarp, 3);
	//
	// 	float l1Dist;
	// 	l1Dist = l1DistBetweenTwoImage(imageToInvWarp, validImage);
	//
	// 	REQUIRE(l1Dist < 1e-5);
	// }


	// for (int i = 0; i < imDim[0]; i++)
	// {
	// 	for (int j = 0; j < imDim[1]; j++)
	// 	{
	// 		for (int k = 0; k < imDim[2]; k++)
	// 		{
	// 			cout << i << " " << j << " " << k << " "
	// 			     << imageToInvWarp->image[k][j][i] << endl;
	// 		}
	// 	}
	// }
}
