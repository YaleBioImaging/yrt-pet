/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/utils/Types.hpp"

#include <experimental/type_traits>

namespace yrt
{

class ProjectionPropertiesT
{
	template <bool HasTOF, bool HasOrient>
	static void getProjectionProperties(
	    const ProjectionData& projData, bin_t bin,
	    std::set<ProjectionPropertiesVariable>& variables,
	    ConstraintParams& consInfo, ProjectionProperties& projProps);
};

namespace ProjProps
{

// Properties
struct Dets { det_pair_t detPair; };
struct LOR { Line3D lor; };
struct Orientation { Vector3D orient1; Vector3D orient2; };
struct TOFValue { float tofValue; };
struct AdditiveCorrection { float additiveCorrection; };
struct ACFInVivo { float acfInVivo; };

// Combined type
template <typename... Fields>
struct ProjectionProps : Fields...
{
};

// Get type from properties
template <typename... Fields>
struct GetProductType;

template <typename First, typename... Rest>
struct GetProductType<First, Rest...>
{
	using type = ProjectionProps<First, Rest...>;
};

// Test functions

template <typename T>
void has_det() {}
template <typename T, typename U>
auto has_det(T*) -> decltype(has_det(std::declval<U*>()), void(),
                             std::true_type());
template <typename T>
struct TestHasDet
{
	static constexpr bool value = has_det<T>(0).value;
};

template <typename T>
void has_orientation() { }
template <typename T, typename U>
auto has_orientation(T*) -> decltype(has_orientation(std::declval<U*>()), void(),
                                     std::true_type());
template <typename T>
struct TestHasOrientation {
  static constexpr bool value = has_orientation<T>(0).value;
};

template <typename T>
void has_tof() { }
template <typename T, typename U>
auto has_tof(T*) -> decltype(has_tof(std::declval<U*>()), void(), std::true_type());
template <typename T>
struct TestHasTOF {
  static constexpr bool value = has_tof<T>(0).value;
};

}  // namespace ProjProps

}  // namespace yrt
