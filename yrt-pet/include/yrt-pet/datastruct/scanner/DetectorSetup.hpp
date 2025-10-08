/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/scanner/DetectorMask.hpp"
#include "yrt-pet/geometry/Vector3D.hpp"
#include "yrt-pet/utils/Types.hpp"

#include <memory>
#include <string>

namespace yrt
{
class DetectorSetup
{
public:
	virtual ~DetectorSetup() = default;
	virtual size_t getNumDets() const = 0;
	virtual float getXpos(det_id_t id) const = 0;
	virtual float getYpos(det_id_t id) const = 0;
	virtual float getZpos(det_id_t id) const = 0;
	virtual float getXorient(det_id_t id) const = 0;
	virtual float getYorient(det_id_t id) const = 0;
	virtual float getZorient(det_id_t id) const = 0;
	virtual bool isDetectorAllowed(det_id_t det) const;
	virtual void writeToFile(const std::string& detCoord_fname) const = 0;
	virtual Vector3D getPos(det_id_t id) const;
	virtual Vector3D getOrient(det_id_t id) const;
	virtual bool hasMask() const;

	DetectorMask& getMask();

	void addMask(const std::string& mask_fname);
	void addMask(const DetectorMask& mask);

protected:
	std::unique_ptr<DetectorMask> mp_mask;
};
}  // namespace yrt
