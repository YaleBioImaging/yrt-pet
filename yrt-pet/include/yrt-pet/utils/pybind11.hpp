/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#if BUILD_PYBIND11

#define PY_PUB_MEMBER(CLASS, NAME) .def_readwrite(#NAME, &CLASS::NAME)
#define PY_PUB_MEMBER_CONST(CLASS, NAME)                       \
	.def_property(                                             \
	    #NAME, [](const CLASS& g) { return g.NAME; }, nullptr, \
	    pybind11::return_value_policy::copy)
#define PY_PUB_METHOD_NOARGS(CLASS, NAME) .def(#NAME, &CLASS::NAME)
#define PY_PUB_METHOD_SIGNATURE(CLASS, NAME, SIG) .def(#NAME, SIG& CLASS::NAME)
#define PY_PUB_METHOD_SIGNATURE_ARGS(CLASS, NAME, SIG, ARGS) \
	.def(#NAME, SIG& CLASS::NAME, ARGS)
#define GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#define PY_PUB_METHOD(...)                                     \
	GET_MACRO(__VA_ARGS__, PY_PUB_METHOD_SIGNATURE_ARGS,       \
	          PY_PUB_METHOD_SIGNATURE, PY_PUB_METHOD_NOARGS, ) \
	(__VA_ARGS__)
#define PY_PUB_STATIC_METHOD(CLASS, NAME) .def_static(#NAME, &CLASS::NAME)

#define STRCAT3(x, y, z) x##y##z
#define PY_DECLARE_ARRAY(TYPE, NDIM) \
	declare_array<STRCAT3(Array, NDIM, D) < TYPE>, TYPE, NDIM > (m, #TYPE)


#endif  // if BUILD_PYBIND11
