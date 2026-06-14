/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/RawParameters.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/image/ImageParams.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"


namespace yrt
{

RawScannerParams getRawScannerParams(const Scanner& scanner)
{
	RawScannerParams params;
	params.crystalSize_trans = scanner.crystalSize_trans;
	params.crystalSize_z = scanner.crystalSize_z;
	params.numDets = scanner.getNumDets();
	params.crystalDepth = scanner.crystalDepth;
	return params;
}

RawImageParams getRawImageParams(const ImageParams& imgParams)
{
	RawImageParams params;

	params.nx = imgParams.nx;
	params.ny = imgParams.ny;
	params.nz = imgParams.nz;

	params.length_x = imgParams.length_x;
	params.length_y = imgParams.length_y;
	params.length_z = imgParams.length_z;

	params.vx = imgParams.vx;
	params.vy = imgParams.vy;
	params.vz = imgParams.vz;

	params.off_x = imgParams.off_x;
	params.off_y = imgParams.off_y;
	params.off_z = imgParams.off_z;

	params.fovRadius = imgParams.fovRadius;

	return params;
}

RawImage getRawImage(Image& img)
{
	return {getRawImageParams(img.getParams()), img.getRawPointer()};
}

RawImage getRawImage(ImageDevice& img)
{
	return {getRawImageParams(img.getParams()), img.getDevicePointer()};
}

RawImageConst getRawImage(const Image& img)
{
	return {getRawImageParams(img.getParams()), img.getRawPointer()};
}

RawImageConst getRawImage(const ImageDevice& img)
{
	return {getRawImageParams(img.getParams()), img.getDevicePointer()};
}

}  // namespace yrt
