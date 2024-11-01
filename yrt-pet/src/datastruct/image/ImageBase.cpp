/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/image/ImageBase.hpp"
#include "geometry/Constants.hpp"
#include "utils/JSONUtils.hpp"

#include "nlohmann/json.hpp"
#include <fstream>

using json = nlohmann::json;

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_imagebase(py::module& m)
{
	auto c = py::class_<ImageBase, Variable>(m, "ImageBase");

	c.def("getParams", &ImageBase::getParams);
	c.def("getRadius", &ImageBase::getRadius);
	c.def("setParams", &ImageBase::setParams, py::arg("params"));

	c.def("setValue", &ImageBase::setValue, py::arg("initValue"));
	c.def("addFirstImageToSecond", &ImageBase::addFirstImageToSecond,
	      py::arg("second"));
	c.def("applyThreshold", &ImageBase::applyThreshold, py::arg("maskImage"),
	      py::arg("threshold"), py::arg("val_le_scale"), py::arg("val_le_off"),
	      py::arg("val_gt_scale"), py::arg("val_gt_off"));
	c.def("writeToFile", &ImageBase::writeToFile, py::arg("filename"));
	c.def("updateEMThreshold", &ImageBase::updateEMThreshold,
	      py::arg("update_img"), py::arg("norm_img"), py::arg("threshold"));
}

void py_setup_imageparams(py::module& m)
{
	auto c = py::class_<ImageParams>(m, "ImageParams");
	c.def(py::init<>());
	c.def(py::init<int, int, int, float, float, float, float, float, float>(),
	      py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("length_x"),
	      py::arg("length_y"), py::arg("length_z"), py::arg("offset_x") = 0.,
	      py::arg("offset_y") = 0., py::arg("offset_z") = 0.);
	c.def(py::init<std::string>());
	c.def(py::init<const ImageParams&>());
	c.def_readwrite("nx", &ImageParams::nx);
	c.def_readwrite("ny", &ImageParams::ny);
	c.def_readwrite("nz", &ImageParams::nz);

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
#endif


ImageParams::ImageParams()
    : nx(-1),
      ny(-1),
      nz(-1),
      length_x(-1.0),
      length_y(-1.0),
      length_z(-1.0),
      off_x(0.0),
      off_y(0.0),
      off_z(0.0),
      vx(-1.0),
      vy(-1.0),
      vz(-1.0)
{
}

ImageParams::ImageParams(int nxi, int nyi, int nzi, float length_xi,
                         float length_yi, float length_zi, float offset_xi,
                         float offset_yi, float offset_zi)
    : nx(nxi),
      ny(nyi),
      nz(nzi),
      length_x(length_xi),
      length_y(length_yi),
      length_z(length_zi),
      off_x(offset_xi),
      off_y(offset_yi),
      off_z(offset_zi)
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
	setup();
}

ImageParams::ImageParams(const std::string& fname)
{
	deserialize(fname);
}

void ImageParams::setup()
{
	vx = length_x / static_cast<float>(nx);
	vy = length_y / static_cast<float>(ny);
	vz = length_z / static_cast<float>(nz);
	fovRadius = static_cast<float>(std::max(length_x / 2, length_y / 2));
	fovRadius -= static_cast<float>(std::max(vx, vy) / 1000);
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

	Util::getParam<float>(
	    &j, &version, {"VERSION", "GCIMAGEPARAMS_FILE_VERSION"}, -1.0, true,
	    "Error in ImageParams file version : Version unspecified");

	if (version > IMAGEPARAMS_FILE_VERSION + SMALL_FLT)
	{
		throw std::logic_error("Error in ImageParams file version : Wrong "
		                       "version. Current version: " +
		                       std::to_string(IMAGEPARAMS_FILE_VERSION) +
		                       ", Given version: " + std::to_string(version));
	}

	Util::getParam<int>(
	    &j, &nx, "nx", 0, true,
	    "Error in ImageParams file version : \'nx\' unspecified");

	Util::getParam<int>(
	    &j, &ny, "ny", 0, true,
	    "Error in ImageParams file version : \'ny\' unspecified");

	Util::getParam<int>(
	    &j, &nz, "nz", 0, true,
	    "Error in ImageParams file version : \'nz\' unspecified");

	Util::getParam<float>(&j, &off_x, {"off_x", "offset_x"}, 0.0, false);

	Util::getParam<float>(&j, &off_y, {"off_y", "offset_y"}, 0.0, false);

	Util::getParam<float>(&j, &off_z, {"off_z", "offset_z"}, 0.0, false);

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
	if (!Util::getParam<float>(&j, &given_v, v_name, -1.0, false))
	{
		float length;
		Util::getParam<float>(&j, &length, length_name, -1.0, true,
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

bool ImageParams::isSameDimensionsAs(const ImageParams& other) const
{
	return nx == other.nx && ny == other.ny && nz == other.nz;
}

bool ImageParams::isSameLengthsAs(const ImageParams& other) const
{
	return APPROX_EQ_THRESH(length_x, other.length_x, 1e-4) &&
	       APPROX_EQ_THRESH(length_y, other.length_y, 1e-4) &&
	       APPROX_EQ_THRESH(length_z, other.length_z, 1e-4);
}

bool ImageParams::isSameOffsetsAs(const ImageParams& other) const
{
	return APPROX_EQ_THRESH(off_x, other.off_x, 1e-4) &&
	       APPROX_EQ_THRESH(off_y, other.off_y, 1e-4) &&
	       APPROX_EQ_THRESH(off_z, other.off_z, 1e-4);
}

bool ImageParams::isSameAs(const ImageParams& other) const
{
	return isSameDimensionsAs(other) && isSameLengthsAs(other) &&
	       isSameOffsetsAs(other);
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

float ImageBase::getRadius() const
{
	return m_params.fovRadius;
}
