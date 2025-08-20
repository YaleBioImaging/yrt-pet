/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/scanner/DetCoord.hpp"

#include "yrt-pet/utils/Array.hpp"
#include <memory>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

#include <fstream>

#if BUILD_PYBIND11

namespace yrt
{

void py_setup_detcoord(py::module& m)
{
	auto c =
	    pybind11::class_<DetCoord, DetectorSetup, std::shared_ptr<DetCoord>>(
	        m, "DetCoord");

	c.def("setXpos", &DetCoord::setXpos);
	c.def("setYpos", &DetCoord::setYpos);
	c.def("setZpos", &DetCoord::setZpos);
	c.def("setXorient", &DetCoord::setXorient);
	c.def("setYorient", &DetCoord::setYorient);
	c.def("setZorient", &DetCoord::setZorient);

	c.def("getXposArray",
	      [](const DetCoord& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* posArr = self.getXposArrayRef();
		      auto buf_info =
		          py::buffer_info(posArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {posArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getYposArray",
	      [](const DetCoord& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* posArr = self.getYposArrayRef();
		      auto buf_info =
		          py::buffer_info(posArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {posArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getZposArray",
	      [](const DetCoord& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* posArr = self.getZposArrayRef();
		      auto buf_info =
		          py::buffer_info(posArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {posArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getXorientArray",
	      [](const DetCoord& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* orientArr = self.getXorientArrayRef();
		      auto buf_info =
		          py::buffer_info(orientArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {orientArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getYorientArray",
	      [](const DetCoord& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* orientArr = self.getYorientArrayRef();
		      auto buf_info =
		          py::buffer_info(orientArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {orientArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getZorientArray",
	      [](const DetCoord& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* orientArr = self.getZorientArrayRef();
		      auto buf_info =
		          py::buffer_info(orientArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {orientArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });


	auto c_owned =
	    pybind11::class_<DetCoordOwned, DetCoord,
	                     std::shared_ptr<DetCoordOwned>>(m, "DetCoordOwned");
	c_owned.def(py::init<>());
	c_owned.def(py::init<const std::string&>());
	c_owned.def("readFromFile", &DetCoordOwned::readFromFile);
	c_owned.def("allocate", &DetCoordOwned::allocate);

	auto c_alias =
	    pybind11::class_<DetCoordAlias, DetCoord,
	                     std::shared_ptr<DetCoordAlias>>(m, "DetCoordAlias");
	c_alias.def(py::init<>());
	c_alias.def(
	    "bind",
	    [](DetCoordAlias& self, py::buffer& xpos, py::buffer& ypos,
	       py::buffer& zpos, py::buffer& xorient, py::buffer& yorient,
	       py::buffer& zorient)
	    {
		    py::buffer_info xpos_info = xpos.request();
		    py::buffer_info zpos_info = ypos.request();
		    py::buffer_info ypos_info = zpos.request();
		    py::buffer_info xorient_info = xorient.request();
		    py::buffer_info zorient_info = yorient.request();
		    py::buffer_info yorient_info = zorient.request();
		    if (xpos_info.format != py::format_descriptor<float>::format() ||
		        xpos_info.ndim != 1)
			    throw std::invalid_argument(
			        "The XPos array has to be a 1-dimensional float32 array");
		    if (ypos_info.format != py::format_descriptor<float>::format() ||
		        ypos_info.ndim != 1)
			    throw std::invalid_argument(
			        "The YPos array has to be a 1-dimensional float32 array");
		    if (zpos_info.format != py::format_descriptor<float>::format() ||
		        zpos_info.ndim != 1)
			    throw std::invalid_argument(
			        "The ZPos array has to be a 1-dimensional float32 array");
		    if (xorient_info.format != py::format_descriptor<float>::format() ||
		        xorient_info.ndim != 1)
			    throw std::invalid_argument("The XOrient array has to be a "
			                                "1-dimensional float32 array");
		    if (yorient_info.format != py::format_descriptor<float>::format() ||
		        yorient_info.ndim != 1)
			    throw std::invalid_argument("The YOrient array has to be a "
			                                "1-dimensional float32 array");
		    if (zorient_info.format != py::format_descriptor<float>::format() ||
		        zorient_info.ndim != 1)
			    throw std::invalid_argument("The ZOrient array has to be a "
			                                "1-dimensional float32 array");
		    if (xpos_info.shape[0] != ypos_info.shape[0] ||
		        xpos_info.shape[0] != zpos_info.shape[0] ||
		        xpos_info.shape[0] != xorient_info.shape[0] ||
		        xpos_info.shape[0] != yorient_info.shape[0] ||
		        xpos_info.shape[0] != zorient_info.shape[0])
			    throw std::invalid_argument(
			        "All the arrays given have to have the same size");

		    static_cast<Array1DAlias<float>*>(self.getXposArrayRef())
		        ->bind(reinterpret_cast<float*>(xpos_info.ptr),
		               xpos_info.shape[0]);
		    static_cast<Array1DAlias<float>*>(self.getYposArrayRef())
		        ->bind(reinterpret_cast<float*>(ypos_info.ptr),
		               ypos_info.shape[0]);
		    static_cast<Array1DAlias<float>*>(self.getZposArrayRef())
		        ->bind(reinterpret_cast<float*>(zpos_info.ptr),
		               zpos_info.shape[0]);

		    static_cast<Array1DAlias<float>*>(self.getXorientArrayRef())
		        ->bind(reinterpret_cast<float*>(xorient_info.ptr),
		               xorient_info.shape[0]);
		    static_cast<Array1DAlias<float>*>(self.getYorientArrayRef())
		        ->bind(reinterpret_cast<float*>(yorient_info.ptr),
		               yorient_info.shape[0]);
		    static_cast<Array1DAlias<float>*>(self.getZorientArrayRef())
		        ->bind(reinterpret_cast<float*>(zorient_info.ptr),
		               zorient_info.shape[0]);
	    });
}
}  // namespace yrt

#endif

namespace yrt
{

DetCoord::DetCoord() = default;

DetCoordOwned::DetCoordOwned() : DetCoord()
{
	mp_Xpos = std::make_unique<Array1D<float>>();
	mp_Ypos = std::make_unique<Array1D<float>>();
	mp_Zpos = std::make_unique<Array1D<float>>();
	mp_Xorient = std::make_unique<Array1D<float>>();
	mp_Yorient = std::make_unique<Array1D<float>>();
	mp_Zorient = std::make_unique<Array1D<float>>();
}

DetCoordOwned::DetCoordOwned(const std::string& filename) : DetCoordOwned()
{
	readFromFile(filename);
}

DetCoordAlias::DetCoordAlias() : DetCoord()
{
	mp_Xpos = std::make_unique<Array1DAlias<float>>();
	mp_Ypos = std::make_unique<Array1DAlias<float>>();
	mp_Zpos = std::make_unique<Array1DAlias<float>>();
	mp_Xorient = std::make_unique<Array1DAlias<float>>();
	mp_Yorient = std::make_unique<Array1DAlias<float>>();
	mp_Zorient = std::make_unique<Array1DAlias<float>>();
}

void DetCoordOwned::allocate(size_t numDets)
{
	reinterpret_cast<Array1D<float>*>(mp_Xpos.get())->allocate(numDets);
	reinterpret_cast<Array1D<float>*>(mp_Ypos.get())->allocate(numDets);
	reinterpret_cast<Array1D<float>*>(mp_Zpos.get())->allocate(numDets);
	reinterpret_cast<Array1D<float>*>(mp_Xorient.get())->allocate(numDets);
	reinterpret_cast<Array1D<float>*>(mp_Yorient.get())->allocate(numDets);
	reinterpret_cast<Array1D<float>*>(mp_Zorient.get())->allocate(numDets);
}

void DetCoord::writeToFile(const std::string& detCoord_fname) const
{
	std::ofstream file;
	file.open(detCoord_fname.c_str(), std::ios::binary | std::ios::out);
	if (!file.is_open())
	{
		throw std::runtime_error("Error in opening of file " + detCoord_fname +
		                         ".");
	}
	for (size_t j = 0; j < getNumDets(); j++)
	{
		float Xpos10 = (*mp_Xpos)[j];
		float Ypos10 = (*mp_Ypos)[j];
		float Zpos10 = (*mp_Zpos)[j];

		file.write((char*)(&(Xpos10)), sizeof(float));
		file.write((char*)(&(Ypos10)), sizeof(float));
		file.write((char*)(&(Zpos10)), sizeof(float));

		file.write((char*)(&((*mp_Xorient)[j])), sizeof(float));
		file.write((char*)(&((*mp_Yorient)[j])), sizeof(float));
		file.write((char*)(&((*mp_Zorient)[j])), sizeof(float));
	}
}

void DetCoordOwned::readFromFile(const std::string& filename)
{
	// File format:
	// <float><float><float><float><float><float>
	// <float><float><float><float><float><float>
	// <float><float><float><float><float><float>
	// ...
	std::ifstream fin(filename.c_str(), std::ios::in | std::ios::binary);
	if (!fin.good())
	{
		throw std::runtime_error("Error reading input file " + filename);
	}

	// first check that file has the right size:
	fin.seekg(0, std::ios::end);
	size_t end = fin.tellg();
	fin.seekg(0, std::ios::beg);
	size_t begin = fin.tellg();
	size_t fileSize = end - begin;

	size_t numElem = fileSize / sizeof(float);

	if (fileSize <= 0 || fileSize % sizeof(float) != 0 || numElem % 6 != 0)
	{
		throw std::logic_error("Error: Input file has incorrect size");
	}

	size_t numDets = numElem / 6;
	allocate(numDets);

	auto buff = std::make_unique<float[]>(numElem);

	fin.read(reinterpret_cast<char*>(buff.get()), numElem * sizeof(float));

	// Get raw pointers
	float* xPos_ptr = mp_Xpos->getRawPointer();
	float* yPos_ptr = mp_Ypos->getRawPointer();
	float* zPos_ptr = mp_Zpos->getRawPointer();
	float* xOrient_ptr = mp_Xorient->getRawPointer();
	float* yOrient_ptr = mp_Yorient->getRawPointer();
	float* zOrient_ptr = mp_Zorient->getRawPointer();
	const float* buff_ptr = buff.get();

	util::parallel_for_chunked(numDets, globals::numThreads(),
	                           [xPos_ptr, yPos_ptr, zPos_ptr, xOrient_ptr,
	                            yOrient_ptr, zOrient_ptr,
	                            buff_ptr](size_t i, size_t /*tid*/)
	                           {
		                           xPos_ptr[i] = buff_ptr[6 * i + 0];
		                           yPos_ptr[i] = buff_ptr[6 * i + 1];
		                           zPos_ptr[i] = buff_ptr[6 * i + 2];
		                           xOrient_ptr[i] = buff_ptr[6 * i + 3];
		                           yOrient_ptr[i] = buff_ptr[6 * i + 4];
		                           zOrient_ptr[i] = buff_ptr[6 * i + 5];
	                           });

	fin.close();
}

void DetCoordAlias::bind(DetCoord* p_detCoord)
{
	bind(p_detCoord->getXposArrayRef(), p_detCoord->getYposArrayRef(),
	     p_detCoord->getZposArrayRef(), p_detCoord->getXorientArrayRef(),
	     p_detCoord->getYorientArrayRef(), p_detCoord->getZorientArrayRef());
}

void DetCoordAlias::bind(Array1DBase<float>* p_Xpos, Array1DBase<float>* p_Ypos,
                         Array1DBase<float>* p_Zpos,
                         Array1DBase<float>* p_Xorient,
                         Array1DBase<float>* p_Yorient,
                         Array1DBase<float>* p_Zorient)
{
	bool isNotNull = true;

	static_cast<Array1DAlias<float>*>(mp_Xpos.get())->bind(*p_Xpos);
	static_cast<Array1DAlias<float>*>(mp_Ypos.get())->bind(*p_Ypos);
	static_cast<Array1DAlias<float>*>(mp_Zpos.get())->bind(*p_Zpos);
	static_cast<Array1DAlias<float>*>(mp_Xorient.get())->bind(*p_Xorient);
	static_cast<Array1DAlias<float>*>(mp_Yorient.get())->bind(*p_Yorient);
	static_cast<Array1DAlias<float>*>(mp_Zorient.get())->bind(*p_Zorient);

	isNotNull &= (mp_Xpos->getRawPointer() != nullptr);
	isNotNull &= (mp_Ypos->getRawPointer() != nullptr);
	isNotNull &= (mp_Zpos->getRawPointer() != nullptr);
	isNotNull &= (mp_Xorient->getRawPointer() != nullptr);
	isNotNull &= (mp_Yorient->getRawPointer() != nullptr);
	isNotNull &= (mp_Zorient->getRawPointer() != nullptr);
	if (!isNotNull)
	{
		throw std::runtime_error(
		    "An error occured during the binding of the DetCoord");
	}
}

// GETTERS AND SETTERS
float DetCoord::getXpos(det_id_t detID) const
{
	return (*mp_Xpos)[detID];
}
float DetCoord::getYpos(det_id_t detID) const
{
	return (*mp_Ypos)[detID];
}
float DetCoord::getZpos(det_id_t detID) const
{
	return (*mp_Zpos)[detID];
}
float DetCoord::getXorient(det_id_t detID) const
{
	return (*mp_Xorient)[detID];
}
float DetCoord::getYorient(det_id_t detID) const
{
	return (*mp_Yorient)[detID];
}
float DetCoord::getZorient(det_id_t detID) const
{
	return (*mp_Zorient)[detID];
}

void DetCoord::setXpos(det_id_t detID, float f)
{
	(*mp_Xpos)[detID] = f;
}
void DetCoord::setYpos(det_id_t detID, float f)
{
	(*mp_Ypos)[detID] = f;
}
void DetCoord::setZpos(det_id_t detID, float f)
{
	(*mp_Zpos)[detID] = f;
}
void DetCoord::setXorient(det_id_t detID, float f)
{
	(*mp_Xorient)[detID] = f;
}
void DetCoord::setYorient(det_id_t detID, float f)
{
	(*mp_Yorient)[detID] = f;
}
void DetCoord::setZorient(det_id_t detID, float f)
{
	(*mp_Zorient)[detID] = f;
}

size_t DetCoord::getNumDets() const
{
	return this->mp_Xpos->getSize(0);
}
}  // namespace yrt
