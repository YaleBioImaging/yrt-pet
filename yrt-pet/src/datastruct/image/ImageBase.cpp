/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/image/ImageBase.hpp"
#include "yrt-pet/datastruct/image/ImageUtils.cuh"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/JSONUtils.hpp"

#include "nlohmann/json.hpp"
#include <fstream>

using json = nlohmann::json;

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace py::literals;

namespace yrt
{

void py_setup_imageparams(py::module& m)
{
	auto c = py::class_<ImageParams>(m, "ImageParams");
	c.def(py::init<>());
	c.def(py::init<int, int, int, float, float, float, float, float, float>(),
	      "nx"_a, "ny"_a, "nz"_a, "length_x"_a, "length_y"_a, "length_z"_a,
	      "offset_x"_a = 0., "offset_y"_a = 0., "offset_z"_a = 0., "num_frames"_a = 1);
	c.def(py::init<std::string>());
	c.def(py::init<const ImageParams&>());
	c.def_readwrite("nx", &ImageParams::nx);
	c.def_readwrite("ny", &ImageParams::ny);
	c.def_readwrite("nz", &ImageParams::nz);
	c.def_readwrite("num_frames", &ImageParams::num_frames);

	c.def_readwrite("length_x", &ImageParams::length_x);
	c.def_readwrite("length_y", &ImageParams::length_y);
	c.def_readwrite("length_z", &ImageParams::length_z);

	c.def_readwrite("off_x", &ImageParams::off_x);
	c.def_readwrite("off_y", &ImageParams::off_y);
	c.def_readwrite("off_z", &ImageParams::off_z);

	// Automatically populated fields
	c.def_readwrite("vx", &ImageParams::vx);
	c.def_readwrite("vy", &ImageParams::vy);
	c.def_readwrite("vz", &ImageParams::vz);
	c.def_readwrite("fov_radius", &ImageParams::fovRadius);

	c.def("setup", &ImageParams::setup);
	c.def("serialize", &ImageParams::serialize);
	c.def("deserialize", &ImageParams::deserialize);

	c.def("isSameAs", &ImageParams::isSameAs);
	c.def("isSameOffsetsAs", &ImageParams::isSameOffsetsAs);
	c.def("isSameLengthsAs", &ImageParams::isSameLengthsAs);
	c.def("isSameDimensionsAs", &ImageParams::isSameDimensionsAs);
	c.def("indexToPositionInZDimension",
	      &ImageParams::indexToPositionInDimension<0>);
	c.def("indexToPositionInYDimension",
	      &ImageParams::indexToPositionInDimension<1>);
	c.def("indexToPositionInXDimension",
	      &ImageParams::indexToPositionInDimension<2>);
	c.def("indexToPosition", &ImageParams::indexToPosition, "ix"_a, "iy"_a,
	      "iz"_a);

	c.def(py::pickle(
	    [](const ImageParams& g)
	    {
		    nlohmann::json geom_json;
		    g.writeToJSON(geom_json);
		    std::stringstream oss;
		    oss << geom_json;
		    return oss.str();
	    },
	    [](const std::string& s)
	    {
		    nlohmann::json geom_json;
		    geom_json = json::parse(s);
		    ImageParams g;
		    g.readFromJSON(geom_json);
		    return g;
	    }));
}

void py_setup_imagebase(py::module& m)
{
	auto c = py::class_<ImageBase, Variable>(m, "ImageBase");

	c.def("getParams", &ImageBase::getParams);
	c.def("getRadius", &ImageBase::getRadius);
	c.def("setParams", &ImageBase::setParams, "params"_a);

	c.def("setValue", &ImageBase::setValue, "initValue"_a);
	c.def("addFirstImageToSecond", &ImageBase::addFirstImageToSecond,
	      "second"_a);
	c.def("applyThreshold", &ImageBase::applyThreshold, "maskImage"_a,
	      "threshold"_a, "val_le_scale"_a, "val_le_off"_a, "val_gt_scale"_a,
	      "val_gt_off"_a);
	c.def("writeToFile", &ImageBase::writeToFile, "filename"_a);
	c.def("updateEMThreshold", &ImageBase::updateEMThreshold, "update_img"_a,
	      "norm_img"_a, "threshold"_a);
}

}  // namespace yrt
#endif

namespace yrt
{
ImageParams::ImageParams()
    : nx(-1),
      ny(-1),
      nz(-1),
      length_x(-1.0f),
      length_y(-1.0f),
      length_z(-1.0f),
      vx(-1.0f),
      vy(-1.0f),
      vz(-1.0f),
      off_x(0.0f),
      off_y(0.0f),
      off_z(0.0f),
      num_frames(1),
      fovRadius(-1.0f)
{
}

ImageParams::ImageParams(int nxi, int nyi, int nzi, float length_xi,
                         float length_yi, float length_zi, float offset_xi,
                         float offset_yi, float offset_zi, frame_t num_framesi)
    : nx(nxi),
      ny(nyi),
      nz(nzi),
      length_x(length_xi),
      length_y(length_yi),
      length_z(length_zi),
      vx(-1.0f),
      vy(-1.0f),
      vz(-1.0f),
      off_x(offset_xi),
      off_y(offset_yi),
      off_z(offset_zi),
      num_frames(num_framesi)
{
	setup();
}

ImageParams::ImageParams(const ImageParams& in)
{
	copy(in);
}

ImageParams& ImageParams::operator=(const ImageParams& in)
{
	copy(in);
	return *this;
}

void ImageParams::copy(const ImageParams& in)
{
	nx = in.nx;
	ny = in.ny;
	nz = in.nz;
	length_x = in.length_x;
	length_y = in.length_y;
	length_z = in.length_z;
	off_x = in.off_x;
	off_y = in.off_y;
	off_z = in.off_z;
	vx = in.vx;
	vy = in.vy;
	vz = in.vz;
	num_frames = in.num_frames;
	setup();
}

ImageParams::ImageParams(const std::string& fname) : ImageParams{}
{
	deserialize(fname);
}

void ImageParams::setup()
{
	ASSERT_MSG(nx > 0, "ImageParams object incomplete in X dimension");
	ASSERT_MSG(ny > 0, "ImageParams object incomplete in Y dimension");
	ASSERT_MSG(nz > 0, "ImageParams object incomplete in Z dimension");

	completeDimInfo<0>();
	completeDimInfo<1>();
	completeDimInfo<2>();

	fovRadius = std::max(length_x / 2.0f, length_y / 2.0f);
	fovRadius -= std::max(vx, vy) / 1000.0f;
}

template <int Dim>
void ImageParams::completeDimInfo()
{
	const int* n = nullptr;
	float* v = nullptr;
	float* length = nullptr;
	float* off = nullptr;

	static_assert(Dim >= 0 && Dim < 3);
	if constexpr (Dim == 0)
	{
		v = &vx;
		n = &nx;
		length = &length_x;
		off = &off_x;
	}
	if constexpr (Dim == 1)
	{
		v = &vy;
		n = &ny;
		length = &length_y;
		off = &off_y;
	}
	if constexpr (Dim == 2)
	{
		v = &vz;
		n = &nz;
		length = &length_z;
		off = &off_z;
	}

	ASSERT(v != nullptr);
	ASSERT(n != nullptr);
	ASSERT(length != nullptr);

	if (*v > 0.0f && *length > 0.0f)
	{
		return;
	}
	if (*v < 0.0f && *length > 0.0f)
	{
		*v = *length / static_cast<float>(*n);
	}
	else if (*v > 0.0f && *length < 0.0f)
	{
		*length = *v * static_cast<float>(*n);
	}
	else if (*v < 0.0f && *length < 0.0f)
	{
		throw std::invalid_argument(
		    "Image parameters incorrectly defined in X");
	}

	// Force offset to be zero in case it is lower than 0.1 micrometers
	if (std::abs(*off) < PositioningPrecision)
	{
		*off = 0.0f;
	}
}

void ImageParams::serialize(const std::string& fname) const
{
	std::ofstream output(fname);
	json geom_json;
	writeToJSON(geom_json);
	output << geom_json << std::endl;
}

void ImageParams::writeToJSON(json& j) const
{
	j["VERSION"] = IMAGEPARAMS_FILE_VERSION;
	j["nx"] = nx;
	j["ny"] = ny;
	j["nz"] = nz;
	j["vx"] = vx;
	j["vy"] = vy;
	j["vz"] = vz;
	j["off_x"] = off_x;
	j["off_y"] = off_y;
	j["off_z"] = off_z;
	j["num_frames"] = num_frames;
}

void ImageParams::deserialize(const std::string& fname)
{
	std::ifstream ifs(fname);
	if (!ifs.is_open())
	{
		throw std::runtime_error("Error opening ImageParams file: " + fname);
	}
	auto ss = std::ostringstream{};
	ss << ifs.rdbuf();
	std::string fileContents = ss.str();
	json j;
	try
	{
		j = json::parse(fileContents);
	}
	catch (json::exception& e)
	{
		throw std::invalid_argument("Error in ImageParams JSON file parsing");
	}
	readFromJSON(j);
}

void ImageParams::readFromJSON(json& j)
{
	float version;

	util::getParam<float>(
	    &j, &version, {"VERSION", "GCIMAGEPARAMS_FILE_VERSION"}, -1.0, true,
	    "Error in ImageParams file version : Version unspecified");

	if (version > IMAGEPARAMS_FILE_VERSION + SMALL_FLT)
	{
		throw std::logic_error("Error in ImageParams file version : Wrong "
		                       "version. Current version: " +
		                       std::to_string(IMAGEPARAMS_FILE_VERSION) +
		                       ", Given version: " + std::to_string(version));
	}

	util::getParam<int>(
	    &j, &nx, "nx", 0, true,
	    "Error in ImageParams file version : \'nx\' unspecified");

	util::getParam<int>(
	    &j, &ny, "ny", 0, true,
	    "Error in ImageParams file version : \'ny\' unspecified");

	util::getParam<int>(
	    &j, &nz, "nz", 0, true,
	    "Error in ImageParams file version : \'nz\' unspecified");

	util::getParam<frame_t>(&j, &num_frames, {"num_frames"}, 1, false);

	util::getParam<float>(&j, &off_x, {"off_x", "offset_x"}, 0.0, false);

	util::getParam<float>(&j, &off_y, {"off_y", "offset_y"}, 0.0, false);

	util::getParam<float>(&j, &off_z, {"off_z", "offset_z"}, 0.0, false);

	length_x = readLengthFromJSON(j, "length_x", "vx", nx);
	length_y = readLengthFromJSON(j, "length_y", "vy", ny);
	length_z = readLengthFromJSON(j, "length_z", "vz", nz);

	setup();
}

float ImageParams::readLengthFromJSON(nlohmann::json& j,
                                      const std::string& length_name,
                                      const std::string& v_name, int n)
{
	float given_v;
	if (!util::getParam<float>(&j, &given_v, v_name, -1.0, false))
	{
		float length;
		util::getParam<float>(&j, &length, length_name, -1.0, true,
		                      "You need to specify either the voxel size (vx, "
		                      "vy or vz) or the length (length_x, length_y or "
		                      "length_z) for all three dimensions.");
		return length;
	}
	return given_v * n;
}

bool ImageParams::isValid() const
{
	return nx > 0 && ny > 0 && nz > 0 && length_x > 0. && length_y > 0. &&
	       length_z > 0.;
}
// todo: Add time dimension to checks? Will change implementations during chekcs
bool ImageParams::isSameDimensionsAs(const ImageParams& other) const
{
	return nx == other.nx && ny == other.ny && nz == other.nz;
}

bool ImageParams::isSameLengthsAs(const ImageParams& other) const
{
	return APPROX_EQ_THRESH(length_x, other.length_x, PositioningPrecision) &&
	       APPROX_EQ_THRESH(length_y, other.length_y, PositioningPrecision) &&
	       APPROX_EQ_THRESH(length_z, other.length_z, PositioningPrecision);
}

bool ImageParams::isSameOffsetsAs(const ImageParams& other) const
{
	return APPROX_EQ_THRESH(off_x, other.off_x, PositioningPrecision) &&
	       APPROX_EQ_THRESH(off_y, other.off_y, PositioningPrecision) &&
	       APPROX_EQ_THRESH(off_z, other.off_z, PositioningPrecision);
}

bool ImageParams::isSameAs(const ImageParams& other) const
{
	return isSameDimensionsAs(other) && isSameLengthsAs(other) &&
	       isSameOffsetsAs(other);
}

template <int Dimension>
float ImageParams::indexToPositionInDimension(int index) const
{
	static_assert(Dimension >= 0 && Dimension < 3);
	float voxelSize, length, offset;
	if constexpr (Dimension == 0)
	{
		voxelSize = vz;
		length = length_z;
		offset = off_z;
	}
	else if constexpr (Dimension == 1)
	{
		voxelSize = vy;
		length = length_y;
		offset = off_y;
	}
	else if constexpr (Dimension == 2)
	{
		voxelSize = vx;
		length = length_x;
		offset = off_x;
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}
	return util::indexToPosition(index, voxelSize, length, offset);
}
template float ImageParams::indexToPositionInDimension<0>(int index) const;
template float ImageParams::indexToPositionInDimension<1>(int index) const;
template float ImageParams::indexToPositionInDimension<2>(int index) const;

Vector3D ImageParams::indexToPosition(int ix, int iy, int iz) const
{
	const float posZ = indexToPositionInDimension<0>(iz);
	const float posY = indexToPositionInDimension<1>(iy);
	const float posX = indexToPositionInDimension<2>(ix);
	return Vector3D{posX, posY, posZ};
}

ImageBase::ImageBase(const ImageParams& imgParams) : m_params(imgParams) {}

const ImageParams& ImageBase::getParams() const
{
	return m_params;
}

void ImageBase::setParams(const ImageParams& newParams)
{
	m_params = newParams;
}

size_t ImageBase::unravel(int iz, int iy, int ix, frame_t it) const
{
	return ix + (iy + (iz + m_params.nz * it) * m_params.ny) * m_params.nx;
}

float ImageBase::getRadius() const
{
	return m_params.fovRadius;
}
}  // namespace yrt
