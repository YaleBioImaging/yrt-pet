/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/ScatterSpace.hpp"

#include "yrt-pet/geometry/Constants.hpp"

#define FORBID_INPUT_ERROR_MESSAGE                                          \
	"The scatter space should not be used to gather LORs, do projections, " \
	"or reconstructions. It is meant only for corrections."

namespace yrt
{

ScatterSpace::ScatterSpace(const Scanner& pr_scanner, const std::string& fname)
    : Histogram(pr_scanner), mp_values(nullptr)
{
	readFromFile(fname);

	const auto dims = mp_values->getDims();

	m_numTOFBins = dims[0];
	m_numPlanes = dims[1];
	m_numAngles = dims[2];

	initStepSizes();
}

ScatterSpace::ScatterSpace(const Scanner& pr_scanner, size_t p_numTOFBins,
                           size_t p_numPlanes, size_t p_numAngles)
    : Histogram(pr_scanner),
      m_numTOFBins(p_numTOFBins),
      m_numPlanes(p_numPlanes),
      m_numAngles(p_numAngles)
{
	ASSERT_MSG(m_numTOFBins > 0, "Number of TOF bins must be non-null");
	ASSERT_MSG(m_numPlanes > 0, "Number of planes must be non-null");
	ASSERT_MSG(m_numAngles > 1, "Number of angles must be more than 1");

	initStepSizes();

	// Allocate the scatter space
	mp_values = std::make_unique<Array5D<float>>();
	mp_values->allocate(m_numTOFBins, m_numPlanes, m_numAngles, m_numPlanes,
	                    m_numAngles);
	mp_values->fill(0.0f);
}

void ScatterSpace::readFromFile(const std::string& fname)
{
	if (mp_values == nullptr)
	{
		mp_values = std::make_unique<Array5D<float>>();
	}
	mp_values->readFromFile(fname);
}

void ScatterSpace::writeToFile(const std::string& fname) const
{
	mp_values->writeToFile(fname);
}

ScatterSpace::ScatterSpaceIndex
    ScatterSpace::getNearestNeighborIndex(const ScatterSpacePosition& pos) const
{
	return {getTOFBin(pos.tof_ps), getPlaneIndex(pos.planePosition1),
	        getAngleIndex(pos.angle1), getPlaneIndex(pos.planePosition2),
	        getAngleIndex(pos.angle2)};
}

float ScatterSpace::getNearestNeighborValue(
    const ScatterSpacePosition& pos) const
{
	return getValue(getNearestNeighborIndex(pos));
}

float ScatterSpace::getLinearInterpolationValue(
    const ScatterSpacePosition& pos) const
{
	// TODO NOW: Rename these local variables

	// Clamp and wrap input
	const float clamped_tof = clampTOF(pos.tof_ps);
	const float clamped_plane1 = clampPlanePosition(pos.planePosition1);
	const float wrapped_angle1 = wrapAngle(pos.angle1);
	const float clamped_plane2 = clampPlanePosition(pos.planePosition2);
	const float wrapped_angle2 = wrapAngle(pos.angle2);

	// Get the logical indices but in float
	const float tof_idx = clamped_tof / m_TOFBinStep_ps - 0.5f;
	const float plane_start = -getAxialFOV() / 2.0f;
	const float plane1_idx =
	    (clamped_plane1 - plane_start) / m_planeStep - 0.5f;
	const float angle1_idx = wrapped_angle1 / m_angleStep - 0.5f;
	const float plane2_idx =
	    (clamped_plane2 - plane_start) / m_planeStep - 0.5f;
	const float angle2_idx = wrapped_angle2 / m_angleStep - 0.5f;

	// Logical indices but in signed int
	const int tof_i = static_cast<int>(std::floor(tof_idx));
	const int plane1_i = static_cast<int>(std::floor(plane1_idx));
	const int angle1_i = static_cast<int>(std::floor(angle1_idx));
	const int plane2_i = static_cast<int>(std::floor(plane2_idx));
	const int angle2_i = static_cast<int>(std::floor(angle2_idx));

	// Fraction within the sample
	float tof_frac = tof_idx - tof_i;
	float plane1_frac = plane1_idx - plane1_i;
	const float angle1_frac = angle1_idx - angle1_i;
	float plane2_frac = plane2_idx - plane2_i;
	const float angle2_frac = angle2_idx - angle2_i;

	// From the logical indices, gather the neighbor sample (in each dimension)
	const int tof_i0 = std::clamp(tof_i, 0, static_cast<int>(m_numTOFBins) - 1);
	const int tof_i1 =
	    std::clamp(tof_i + 1, 0, static_cast<int>(m_numTOFBins) - 1);
	const int plane1_i0 =
	    std::clamp(plane1_i, 0, static_cast<int>(m_numPlanes) - 1);
	const int plane1_i1 =
	    std::clamp(plane1_i + 1, 0, static_cast<int>(m_numPlanes) - 1);
	const int plane2_i0 =
	    std::clamp(plane2_i, 0, static_cast<int>(m_numPlanes) - 1);
	const int plane2_i1 =
	    std::clamp(plane2_i + 1, 0, static_cast<int>(m_numPlanes) - 1);

	// Wrap angle indices and get the neighboring sample
	// Edge case:
	//  Might have to do a linear interpolation between the first angle sample
	//  and the last one
	const size_t angle1_i0 = wrapAngleIndex(angle1_i);
	const size_t angle1_i1 = wrapAngleIndex(angle1_i + 1);
	const size_t angle2_i0 = wrapAngleIndex(angle2_i);
	const size_t angle2_i1 = wrapAngleIndex(angle2_i + 1);

	// Edge case for boundaries (if the clamping happened)
	if (tof_i0 == tof_i1)
	{
		tof_frac = 0.0f;
	}
	if (plane1_i0 == plane1_i1)
	{
		plane1_frac = 0.0f;
	}
	if (plane2_i0 == plane2_i1)
	{
		plane2_frac = 0.0f;
	}

	// Get the raw data pointer for faster access
	const float* data = mp_values->getRawPointer();

	// Precompute strides for each dimension
	const size_t stride_TOF =
	    m_numPlanes * m_numAngles * m_numPlanes * m_numAngles;
	const size_t stride_plane1 = m_numAngles * m_numPlanes * m_numAngles;
	const size_t stride_angle1 = m_numPlanes * m_numAngles;
	const size_t stride_plane2 = m_numAngles;
	const size_t stride_angle2 = 1;

	// Precompute weights for all 32 corners using a 5D tensor product
	float weights[32];
	size_t indices[32];

	int idx = 0;
	for (int t = 0; t < 2; ++t)
	{
		const float w_t = (t == 0) ? (1.0f - tof_frac) : tof_frac;
		const size_t tof_index = (t == 0) ? tof_i0 : tof_i1;

		for (int p1 = 0; p1 < 2; ++p1)
		{
			const float w_p1 = (p1 == 0) ? (1.0f - plane1_frac) : plane1_frac;
			const size_t plane1_index = (p1 == 0) ? plane1_i0 : plane1_i1;

			for (int a1 = 0; a1 < 2; ++a1)
			{
				const float w_a1 =
				    (a1 == 0) ? (1.0f - angle1_frac) : angle1_frac;
				const size_t angle1_index = (a1 == 0) ? angle1_i0 : angle1_i1;

				for (int p2 = 0; p2 < 2; ++p2)
				{
					const float w_p2 =
					    (p2 == 0) ? (1.0f - plane2_frac) : plane2_frac;
					const size_t plane2_index =
					    (p2 == 0) ? plane2_i0 : plane2_i1;

					for (int a2 = 0; a2 < 2; ++a2)
					{
						const float w_a2 =
						    (a2 == 0) ? (1.0f - angle2_frac) : angle2_frac;
						const size_t angle2_index =
						    (a2 == 0) ? angle2_i0 : angle2_i1;

						// Compute linear index
						const size_t linear_idx = tof_index * stride_TOF +
						                          plane1_index * stride_plane1 +
						                          angle1_index * stride_angle1 +
						                          plane2_index * stride_plane2 +
						                          angle2_index * stride_angle2;

						indices[idx] = linear_idx;
						weights[idx] = w_t * w_p1 * w_a1 * w_p2 * w_a2;
						idx++;
					}
				}
			}
		}
	}

	// Sum weighted contributions
	float result = 0.0f;
	for (int i = 0; i < 32; ++i)
	{
		result += data[indices[i]] * weights[i];
	}

	return result;
}

float ScatterSpace::getTOF_ps(size_t TOFBin) const
{
	return m_TOFBinStep_ps * (static_cast<float>(TOFBin) + 0.5f);
}

float ScatterSpace::getPlanePosition(size_t planeIndex) const
{
	return m_planeStep * (static_cast<float>(planeIndex) + 0.5f);
}

float ScatterSpace::getAngle(size_t angleIndex) const
{
	return m_angleStep * (static_cast<float>(angleIndex) + 0.5f);
}

size_t ScatterSpace::getTOFBin(float tof_ps) const
{
	// This function assumes that tof_ps is between 0 and getMaxTOF_ps()
	const float clampedTOF_ps = clampTOF(tof_ps);
	const float tofBin_flt = clampedTOF_ps / m_TOFBinStep_ps - 0.5f;
	return static_cast<size_t>(std::round(tofBin_flt));
}

size_t ScatterSpace::getPlaneIndex(float planePosition) const
{
	const float clampedPlanePosition = clampTOF(planePosition);
	const float startingPlane = -getAxialFOV() / 2.0f;
	const float planeIndex_flt =
	    (clampedPlanePosition - startingPlane) / m_planeStep - 0.5f;
	return static_cast<size_t>(std::round(planeIndex_flt));
}

size_t ScatterSpace::getAngleIndex(float angle) const
{
	const float wrappedAngle = wrapAngle(angle);
	const float angleIndex_flt = wrappedAngle / m_angleStep - 0.5f;
	return static_cast<size_t>(std::round(angleIndex_flt));
}

float ScatterSpace::getValue(const ScatterSpaceIndex& idx) const
{
	return getValue(idx.tofBin, idx.planeIndex1, idx.angleIndex1,
	                idx.planeIndex2, idx.angleIndex2);
}

float ScatterSpace::getValue(size_t tofBin, size_t planeIndex1,
                             size_t angleIndex1, size_t planeIndex2,
                             size_t angleIndex2) const
{
	return mp_values->get(
	    {tofBin, planeIndex1, angleIndex1, planeIndex2, angleIndex2});
}

void ScatterSpace::setValue(const ScatterSpaceIndex& idx, float value)
{
	setValue(idx.tofBin, idx.planeIndex1, idx.angleIndex1, idx.planeIndex2,
	         idx.angleIndex2, value);
}

void ScatterSpace::setValue(size_t tofBin, size_t planeIndex1,
                            size_t angleIndex1, size_t planeIndex2,
                            size_t angleIndex2, float value)
{
	mp_values->set({tofBin, planeIndex1, angleIndex1, planeIndex2, angleIndex2},
	               value);
}

void ScatterSpace::symmetrize()
{
	// TODO NOW: Here, make it so that for all elements of this scatter space,
	//  the value at ((a1,z1), (a2,z2), tof) is the same as for
	//  ((a2,z2), (a1,z1), tof) when m_numTOFBins==1
}

float ScatterSpace::clampTOF(float tof_ps) const
{
	const float halfTOFBinStep_ps = getTOFBinStep_ps() / 2;
	const float maxTOFSample = getMaxTOF_ps() - halfTOFBinStep_ps;
	const float minTOFSample = halfTOFBinStep_ps;  // ps
	return std::clamp(tof_ps, minTOFSample, maxTOFSample);
}

float ScatterSpace::clampPlanePosition(float planePosition) const
{
	const float maxPlanePosition = getAxialFOV() / 2 - getPlaneStep() / 2;
	const float minPlanePosition = -maxPlanePosition;
	return std::clamp(planePosition, minPlanePosition, maxPlanePosition);
}

float ScatterSpace::wrapAngle(float angle)
{
	// Wrap to [0, 2pi[
	angle = std::fmod(angle, TWOPI_FLT);
	if (angle < 0)
	{
		angle += TWOPI_FLT;
	}
	return angle;
}

size_t ScatterSpace::wrapAngleIndex(int angleIndex) const
{
	int angleIndexModulo = angleIndex % static_cast<int>(m_numAngles);

	// YN: Unsure about this. Leaving it here just in case
	if (angleIndexModulo < 0)
	{
		angleIndexModulo += static_cast<int>(m_numAngles);
	}

	return static_cast<size_t>(angleIndexModulo);
}

size_t ScatterSpace::getNumTOFBins() const
{
	return m_numTOFBins;
}

size_t ScatterSpace::getNumPlanes() const
{
	return m_numPlanes;
}

size_t ScatterSpace::getNumAngles() const
{
	return m_numAngles;
}

float ScatterSpace::getTOFBinStep_ps() const
{
	return m_TOFBinStep_ps;
}

float ScatterSpace::getPlaneStep() const
{
	return m_planeStep;
}

float ScatterSpace::getAngleStep() const
{
	return m_angleStep;
}

float ScatterSpace::getAxialFOV() const
{
	return mr_scanner.axialFOV;
}

float ScatterSpace::getRadius() const
{
	return mr_scanner.scannerRadius;
}

float ScatterSpace::getDiameter() const
{
	return getRadius() * 2;
}

float ScatterSpace::getMaxTOF_ps() const
{
	return getDiameter() / SPEED_OF_LIGHT_MM_PS_FLT;
}

size_t ScatterSpace::count() const
{
	return mp_values->getSizeTotal();
}

float ScatterSpace::getProjectionValue(bin_t id) const
{
	return mp_values->getFlat(id);
}

void ScatterSpace::setProjectionValue(bin_t id, float val)
{
	mp_values->setFlat(id, val);
}

float ScatterSpace::getProjectionValueFromHistogramBin(
    histo_bin_t histoBinId) const
{
	// TODO NOW:
	//  - Gather the LOR associated with the given detector pair
	//  - Compute the properties of the LOR
	//  - Also Gather the TOF property in case it's available (from histo_bin_t)
	//  - Do linear interpolation and return the value
	return 0.0f;
}

det_id_t ScatterSpace::getDetector1(bin_t /*id*/) const
{
	throw std::runtime_error(FORBID_INPUT_ERROR_MESSAGE);
}

det_id_t ScatterSpace::getDetector2(bin_t /*id*/) const
{
	throw std::runtime_error(FORBID_INPUT_ERROR_MESSAGE);
}

det_pair_t ScatterSpace::getDetectorPair(bin_t /*id*/) const
{
	throw std::runtime_error(FORBID_INPUT_ERROR_MESSAGE);
}

std::unique_ptr<BinIterator> ScatterSpace::getBinIter(int /*numSubsets*/,
                                                      int /*idxSubset*/) const
{
	throw std::runtime_error(FORBID_INPUT_ERROR_MESSAGE);
}

void ScatterSpace::initStepSizes()
{
	m_TOFBinStep_ps = getMaxTOF_ps() / static_cast<float>(m_numTOFBins);
	m_angleStep = TWOPI_FLT / static_cast<float>(m_numAngles);
	m_planeStep = getAxialFOV() / static_cast<float>(m_numPlanes);
}

}  // namespace yrt
