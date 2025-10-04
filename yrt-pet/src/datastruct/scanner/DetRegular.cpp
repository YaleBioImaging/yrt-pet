/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/scanner/DetRegular.hpp"

#include "yrt-pet/geometry/Constants.hpp"

#include <cstdlib>
#include <format>
#include <fstream>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

namespace yrt
{

DetRegular::DetRegular(Scanner* pp_scanner) : mp_scanner(pp_scanner)
{
	mp_Xpos = std::make_unique<Array1D<float>>();
	mp_Ypos = std::make_unique<Array1D<float>>();
	mp_Zpos = std::make_unique<Array1D<float>>();
	mp_Xorient = std::make_unique<Array1D<float>>();
	mp_Yorient = std::make_unique<Array1D<float>>();
	mp_Zorient = std::make_unique<Array1D<float>>();
}

void DetRegular::generateLUT()
{
	size_t numBlocks = mp_scanner->detsPerRing / mp_scanner->detsPerBlock;
	float blockLength =
	    mp_scanner->crystalSize_trans * mp_scanner->detsPerBlock;
	float edgeLengthMax =
	    2 * mp_scanner->scannerRadius * std::tan(PI / numBlocks);
	ASSERT_MSG(mp_scanner->crystalSize_trans * mp_scanner->detsPerBlock <=
	               edgeLengthMax,
	           ("Overlap between crystals detected, given the crystal size and "
	            "number of crystals per block, the scanner radius should be at "
	            "least " +
	            std::to_string(mp_scanner->crystalSize_trans *
	                           mp_scanner->detsPerBlock /
	                           (2 * std::tan(PI / numBlocks))) +
	            " mm")
	               .c_str());
	allocate();
	for (size_t doi = 0; doi < mp_scanner->numDOI; doi++)
	{
		float blockDistance =
		    mp_scanner->scannerRadius + mp_scanner->crystalDepth * doi;
		for (size_t ring = 0; ring < mp_scanner->numRings; ring++)
		{
			// Gap between each ring (Currently, equal distance)
			float zGap = mp_scanner->axialFOV / ((float)mp_scanner->numRings);
			float zPos = -mp_scanner->axialFOV / 2.0 + (ring + 0.5f) * zGap;
			for (size_t block = 0; block < numBlocks; block++)
			{
				float blockAngle = block * TWOPI / numBlocks;
				float xBlock =
				    (blockDistance + 0.5f * mp_scanner->crystalDepth) *
				    std::cos(blockAngle);
				float yBlock =
				    (blockDistance + 0.5f * mp_scanner->crystalDepth) *
				    std::sin(blockAngle);
				for (size_t det = 0; det < mp_scanner->detsPerBlock; det++)
				{
					// relative_det_pos is a number between 0 and 1 describing
					// how far we've gone in the current block
					float relativeDetPos = 0.0;
					if (mp_scanner->detsPerBlock != 1)
					{
						relativeDetPos =
						    -0.5 +
						    (det + 0.5) / (float)mp_scanner->detsPerBlock;
					}
					float relativeXDet =
					    relativeDetPos * std::cos(blockAngle + PI / 2.0);
					float relativeYDet =
					    relativeDetPos * std::sin(blockAngle + PI / 2.0);

					float xDet = xBlock + blockLength * relativeXDet;
					float yDet = yBlock + blockLength * relativeYDet;

					size_t idx =
					    det + block * mp_scanner->detsPerBlock +
					    ring * (mp_scanner->detsPerRing) +
					    doi * (mp_scanner->numRings * mp_scanner->detsPerRing);

					setXpos(idx, xDet);
					setYpos(idx, yDet);
					setZpos(idx, zPos);
					setXorient(idx, std::cos(blockAngle));
					setYorient(idx, std::sin(blockAngle));
					setZorient(idx, 0.0);
				}
			}
		}
	}
}

void DetRegular::allocate()
{
	size_t num_dets = mp_scanner->getTheoreticalNumDets();
	mp_Xpos->allocate(num_dets);
	mp_Ypos->allocate(num_dets);
	mp_Zpos->allocate(num_dets);
	mp_Xorient->allocate(num_dets);
	mp_Yorient->allocate(num_dets);
	mp_Zorient->allocate(num_dets);
}

void DetRegular::writeToFile(const std::string& detReg_fname) const
{
	std::ofstream file;
	file.open(detReg_fname.c_str(), std::ios::binary | std::ios::out);
	if (!file.is_open())
	{
		throw std::runtime_error("Error opening file " + detReg_fname);
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

// GETTERS AND SETTERS
float DetRegular::getXpos(det_id_t detID) const
{
	return (*mp_Xpos)[detID];
}
float DetRegular::getYpos(det_id_t detID) const
{
	return (*mp_Ypos)[detID];
}
float DetRegular::getZpos(det_id_t detID) const
{
	return (*mp_Zpos)[detID];
}
float DetRegular::getXorient(det_id_t detID) const
{
	return (*mp_Xorient)[detID];
}
float DetRegular::getYorient(det_id_t detID) const
{
	return (*mp_Yorient)[detID];
}
float DetRegular::getZorient(det_id_t detID) const
{
	return (*mp_Zorient)[detID];
}

void DetRegular::setXpos(det_id_t detID, float f)
{
	(*mp_Xpos)[detID] = f;
}
void DetRegular::setYpos(det_id_t detID, float f)
{
	(*mp_Ypos)[detID] = f;
}
void DetRegular::setZpos(det_id_t detID, float f)
{
	(*mp_Zpos)[detID] = f;
}
void DetRegular::setXorient(det_id_t detID, float f)
{
	(*mp_Xorient)[detID] = f;
}
void DetRegular::setYorient(det_id_t detID, float f)
{
	(*mp_Yorient)[detID] = f;
}
void DetRegular::setZorient(det_id_t detID, float f)
{
	(*mp_Zorient)[detID] = f;
}

size_t DetRegular::getNumDets() const
{
	return this->mp_Xpos->getSize(0);
}

}  // namespace yrt

#if BUILD_PYBIND11

namespace yrt
{

void py_setup_detregular(py::module& m)
{
	auto c = pybind11::class_<DetRegular, DetectorSetup,
	                          std::shared_ptr<DetRegular>>(m, "DetRegular");
	c.def(py::init<Scanner*>());

	c.def("generateLUT", &DetRegular::generateLUT);
	c.def("setXpos", &DetRegular::setXpos);
	c.def("setYpos", &DetRegular::setYpos);
	c.def("setZpos", &DetRegular::setZpos);
	c.def("setXorient", &DetRegular::setXorient);
	c.def("setYorient", &DetRegular::setYorient);
	c.def("setZorient", &DetRegular::setZorient);
	c.def("getScanner", &DetRegular::getScanner);

	c.def("getXposArray",
	      [](const DetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* posArr = self.getXposArrayRef();
		      auto buf_info =
		          py::buffer_info(posArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {posArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getYposArray",
	      [](const DetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* posArr = self.getYposArrayRef();
		      auto buf_info =
		          py::buffer_info(posArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {posArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getZposArray",
	      [](const DetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* posArr = self.getZposArrayRef();
		      auto buf_info =
		          py::buffer_info(posArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {posArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getXorientArray",
	      [](const DetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* orientArr = self.getXorientArrayRef();
		      auto buf_info =
		          py::buffer_info(orientArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {orientArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getYorientArray",
	      [](const DetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* orientArr = self.getYorientArrayRef();
		      auto buf_info =
		          py::buffer_info(orientArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {orientArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getZorientArray",
	      [](const DetRegular& self) -> py::array_t<float>
	      {
		      Array1DBase<float>* orientArr = self.getZorientArrayRef();
		      auto buf_info =
		          py::buffer_info(orientArr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {orientArr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
}
}  // namespace yrt

#endif
