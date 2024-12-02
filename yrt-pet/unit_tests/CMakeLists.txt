project(unit_tests)

set(SOURCES_COMMON
        test_base.cpp
        test_utils.cpp)

set(SOURCES_DATASTRUCT
        ${SOURCES_COMMON}
        recon/test_Array.cpp
        recon/test_CSV.cpp
        recon/test_SparseHistogram.cpp
        recon/test_Histogram3D.cpp
        recon/test_Image.cpp
        recon/test_ListMode.cpp
        recon/test_Scanner.cpp
        recon/test_TimeOfFlight.cpp
        utils/test_BinIterator.cpp)

if (${USE_CUDA})
    set(SOURCES_DATASTRUCT "${SOURCES_DATASTRUCT};utils/test_CUDA.cu")
endif ()

define_target_exe(test_runner_datastruct "${SOURCES_DATASTRUCT}")
if (${USE_CUDA})
    set_property(TARGET test_runner_datastruct PROPERTY CUDA_ARCHITECTURES ${YRTPET_CUDA_ARCHITECTURES})
endif ()
target_include_directories(test_runner_datastruct PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(test_runner_datastruct PUBLIC Catch)
add_test(test_runner_datastruct test_runner_datastruct)

set(SOURCES_ALGORITHMS
        ${SOURCES_COMMON}
        recon/test_DD.cpp
        recon/test_Siddon.cpp
        motion/test_Warper.cpp)

define_target_exe(test_runner_algorithms "${SOURCES_ALGORITHMS}")
if (${USE_CUDA})
    set_property(TARGET test_runner_algorithms PROPERTY CUDA_ARCHITECTURES ${YRTPET_CUDA_ARCHITECTURES})
endif ()
target_include_directories(test_runner_datastruct PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(test_runner_algorithms PUBLIC Catch)
add_test(test_runner_algorithms test_runner_algorithms)
