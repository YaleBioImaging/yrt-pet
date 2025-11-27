project(unit_tests)

set(SOURCES_COMMON
    test_base.cpp
    test_utils.cpp)

set(SOURCES_DATASTRUCT
    ${SOURCES_COMMON}
    datastruct/test_Array.cpp
    datastruct/test_CSV.cpp
    datastruct/test_SparseHistogram.cpp
    datastruct/test_Histogram3D.cpp
    datastruct/test_LORMotion.cpp
    datastruct/test_Image.cpp
    datastruct/test_ListModeLUT.cpp
    datastruct/test_Scanner.cpp
    datastruct/test_ProjectionProperties.cpp
    operators/test_TimeOfFlight.cpp
    utils/test_BinIterator.cpp
    utils/test_BinFilter.cpp)

if (${USE_CUDA})
    set(SOURCES_DATASTRUCT "${SOURCES_DATASTRUCT};utils/test_CUDA.cu")
endif ()

define_target_exe(test_runner_datastruct "${SOURCES_DATASTRUCT}")
if (${USE_CUDA})
    set_property(TARGET test_runner_datastruct PROPERTY
        CUDA_ARCHITECTURES ${YRTPET_CUDA_ARCHITECTURES})
endif ()
target_include_directories(test_runner_datastruct PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(test_runner_datastruct PUBLIC Catch)
add_test(test_runner_datastruct test_runner_datastruct)

set(SOURCES_ALGORITHMS
    ${SOURCES_COMMON}
    operators/test_DD.cpp
    operators/test_SparseProjection.cpp
    operators/test_Siddon.cpp
    operators/test_PSF.cpp
)

if (${USE_CUDA})
    set(SOURCES_ALGORITHMS "${SOURCES_ALGORITHMS};operators/test_Siddon_GPU.cu")
endif ()

define_target_exe(test_runner_algorithms "${SOURCES_ALGORITHMS}")
if (${USE_CUDA})
    set_property(TARGET test_runner_algorithms PROPERTY
        CUDA_ARCHITECTURES ${YRTPET_CUDA_ARCHITECTURES})
endif ()
target_include_directories(test_runner_datastruct PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(test_runner_algorithms PUBLIC Catch)
add_test(test_runner_algorithms test_runner_algorithms)

set(SOURCES_MISC
    ${SOURCES_COMMON}
    utils/test_Utilities.cpp)
define_target_exe(test_runner_misc "${SOURCES_MISC}")
if (${USE_CUDA})
    set_property(TARGET test_runner_misc PROPERTY
        CUDA_ARCHITECTURES ${YRTPET_CUDA_ARCHITECTURES})
endif ()
target_include_directories(test_runner_datastruct PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(test_runner_misc PUBLIC Catch)
add_test(test_runner_misc test_runner_misc)

if (${BUILD_PYBIND11})
    set(SOURCE_PYTEST
        test_datastruct.py)

    add_pytest(YRTPET_UNIT "${SOURCE_PYTEST}"
        ${CMAKE_SOURCE_DIR}/unit_tests
        ${CMAKE_BINARY_DIR}/unit_tests)

    add_test(NAME pytest_tests
        COMMAND pytest ${CMAKE_BINARY_DIR}/unit_tests)
endif ()
