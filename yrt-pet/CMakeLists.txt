# Jinsong Ouyang, 03-2019

cmake_minimum_required(VERSION 3.28)
project(yrt-pet VERSION 0.0.1 LANGUAGES CXX)


# # Dependencies
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

# CMake tools
include(ExternalProject)

# Program requirements


# User options
option(BUILD_PYBIND11 "Build python bindings using pybind11" ON)
option(BUILD_TESTS "Determines whether to build tests." ON)
option(USE_CUDA "Enable CUDA" ON)
option(BUILD_SHARED_LIB "Additionally build YRT-PET as a shared library " OFF)
option(DOWNLOAD_CATCH2 "Determines whether to download the Catch2 library" ON)
option(DOWNLOAD_CXXOPTS "Determines whether to download the CxxOpts library" ON)
option(DOWNLOAD_JSON "Determines whether to download the JSON library" ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# # Parameters

# cuda
if (USE_CUDA)
    # Set cuda architectures
    set(YRTPET_CUDA_ARCHITECTURES "native")
    if (NOT "$ENV{CUDAARCHS}" STREQUAL "")
        set(YRTPET_CUDA_ARCHITECTURES $ENV{CUDAARCHS})
        message("Found CUDA Architectures from CUDAARCHS environment variable")
    elseif (NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(YRTPET_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        message("Found CUDA Architectures from CMAKE_CUDA_ARCHITECTURES build option")
    else ()
        message("Using CUDA Architecture from native GPU")
    endif ()

    set(CMAKE_CUDA_ARCHITECTURES ${YRTPET_CUDA_ARCHITECTURES})

    find_package(CUDAToolkit)
    enable_language(CUDA)
    add_compile_definitions(BUILD_CUDA)

endif (USE_CUDA)

# Common variables
set(YRTPET_COMPILE_FEATURES cxx_std_20)
set(YRTPET_LINK_LIBRARIES ZLIB::ZLIB ${CMAKE_THREAD_LIBS_INIT})
set(YRTPET_INCLUDE_DIRECTORIES)
set(YRTPET_LIB_NAME yrtpet)

# Default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

# Base directory for plugins
set(PLUGINS_BASE_DIR "${CMAKE_SOURCE_DIR}/plugins")

# # Helper functions

# Add compile_features, include_directory and link_libraries
function(setup_target TARGET JSON CXXOPTS PYBIND11)
    target_compile_features(${TARGET} PUBLIC ${YRTPET_COMPILE_FEATURES})
    # Workaround for filesystem feature with gcc-8
    string(CONCAT LINK_EXTRA "$<$<AND:$<CXX_COMPILER_ID:GNU>,"
        "$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>")
    target_link_libraries(${TARGET} PRIVATE ${LINK_EXTRA})
    target_include_directories(${TARGET} PUBLIC ${YRTPET_INCLUDE_DIRECTORIES}
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:include/>)
    target_link_libraries(${TARGET} PUBLIC ${YRTPET_LINK_LIBRARIES})
    target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-pedantic>
        "$<$<COMPILE_LANGUAGE:CUDA>:>")
    if (${JSON})
        target_link_libraries(${TARGET} PUBLIC JSON)
    endif ()
    if (${CXXOPTS})
        target_link_libraries(${TARGET} PUBLIC CxxOpts)
    endif ()
    if (${PYBIND11} AND BUILD_PYBIND11)
        target_include_directories(${TARGET}
            PUBLIC ${pybind11_INCLUDE_DIR} ${Python_INCLUDE_DIRS})
        target_link_libraries(${TARGET} PUBLIC ${Python_LIBRARIES})
        target_compile_definitions(${TARGET} PUBLIC BUILD_PYBIND11)
    endif ()
    add_compile_definitions(HAVE_ZLIB)
endfunction()

# Wrapper for library target
function(define_yrtpet_lib SOURCES)
    add_library(${YRTPET_LIB_NAME} STATIC)
    target_sources(${YRTPET_LIB_NAME} PRIVATE ${SOURCES})
    set(JSON ON)
    set(CXXOPTS OFF)
    setup_target(${YRTPET_LIB_NAME} ${JSON} ${CXXOPTS} ${BUILD_PYBIND11})
    target_compile_options(${YRTPET_LIB_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
        "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>")
    install(TARGETS ${YRTPET_LIB_NAME}
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
    if (BUILD_SHARED_LIB)
        add_library(${YRTPET_LIB_NAME}_shared SHARED)
        target_sources(${YRTPET_LIB_NAME}_shared PRIVATE ${SOURCES})
        setup_target(${YRTPET_LIB_NAME}_shared ${JSON} ${CXXOPTS} ${BUILD_PYBIND11})
        target_compile_options(${YRTPET_LIB_NAME}_shared PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
                "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>")
        install(TARGETS ${YRTPET_LIB_NAME}_shared
                ARCHIVE DESTINATION lib
                INCLUDES DESTINATION include)
    endif()
endfunction()

# Add plugin sources to library
function(add_plugin_sources)
    target_sources(${YRTPET_LIB_NAME} PRIVATE ${ARGN})
    if (BUILD_SHARED_LIB)
        target_sources(${YRTPET_LIB_NAME}_shared PRIVATE ${ARGN})
    endif()
endfunction()

# Wrapper for executable target (link against library)
function(define_target_exe TARGET)
    add_executable(${TARGET} ${ARGN})
    set(CXXOPTS ${ARGC} GREATER 2 AND ${ARGV2})
    set(JSON OFF)
    setup_target(${TARGET} ${JSON} ${CXXOPTS} ${BUILD_PYBIND11})

    target_link_options(${TARGET} PUBLIC
        -Wl,--whole-archive $<TARGET_FILE:${YRTPET_LIB_NAME}> -Wl,--no-whole-archive
    )
    target_link_libraries(${TARGET} PUBLIC ${YRTPET_LIB_NAME})
    install(TARGETS ${TARGET} RUNTIME DESTINATION bin)
endfunction()

if (BUILD_PYBIND11)
    # Wrapper for pybind11 target (link against library)
    function(define_target_pybind11 TARGET SOURCES PYFOLD)
        pybind11_add_module(${TARGET} ${SOURCES})
        set(JSON ON)
        set(CXXOPTS OFF)
        setup_target(${TARGET} ${JSON} ${CXXOPTS} ON)
        target_link_libraries(${TARGET} PUBLIC ${YRTPET_LIB_NAME})
        set_target_properties(${TARGET} PROPERTIES
            LIBRARY_OUTPUT_NAME ${TARGET}
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pyyrtpet/${TARGET}_wrapper/")
        add_custom_target(copy_py_${TARGET} ALL)
        if (EXISTS ${PYFOLD}/python/${TARGET})
            add_custom_command(TARGET copy_py_${TARGET}
                COMMAND
                ${CMAKE_COMMAND} -E copy_directory
                ${PYFOLD}/python/${TARGET}
                ${CMAKE_BINARY_DIR}/pyyrtpet/${TARGET}_wrapper)
        endif ()
        add_custom_target(copy_py_init_${TARGET} ALL)
        if (EXISTS ${PYFOLD}/python/__init__.py)
            add_custom_command(TARGET copy_py_init_${TARGET}
                COMMAND
                ${CMAKE_COMMAND} -E copy
                ${PYFOLD}/python/__init__.py
                ${CMAKE_BINARY_DIR}/pyyrtpet/__init__.py)
        endif ()
    endfunction()

    # Add plugin sources to pybind11 library
    function(add_pybind11_plugin_sources)
        target_sources(py${YRTPET_LIB_NAME} PRIVATE ${ARGN})
    endfunction()

    # Add integration test
    function(add_pytest NAME SOURCES INPUT_DIR OUTPUT_DIR)
        add_custom_target(copy_pytest_${NAME} ALL)
        foreach (FILE ${SOURCES})
            add_custom_command(TARGET copy_pytest_${NAME}
                COMMAND
                ${CMAKE_COMMAND} -E copy
                ${INPUT_DIR}/${FILE}
                ${OUTPUT_DIR}/${FILE})
        endforeach ()
    endfunction()
endif ()

# Add target from external header only library
function(define_external_target_header TARGET
    URL MD5SUM INCLUDE_PATH FLAG_DOWNLOAD)
    if (${FLAG_DOWNLOAD})
        ExternalProject_Add(${TARGET}-External
            URL ${URL}
            URL_MD5 ${MD5SUM}
            PREFIX "${CMAKE_BINARY_DIR}/external/${TARGET}-External"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_BINARY_DIR}/external/${TARGET}-External/src/${TARGET}-External/
            ${CMAKE_BINARY_DIR}/external/${TARGET}/
            TEST_COMMAND "")
    endif ()
    add_library(${TARGET} INTERFACE)
    if (${FLAG_DOWNLOAD})
        add_dependencies(${TARGET} ${TARGET}-External)
    endif ()
    target_include_directories(${TARGET}
        INTERFACE ${CMAKE_BINARY_DIR}/external/${TARGET}/${INCLUDE_PATH})
endfunction()

# Function to check if a directory contains a CMakeLists.txt
function(check_and_add_subdirectory dir)
    if (IS_DIRECTORY "${dir}" OR IS_SYMLINK "${dir}")
        if (EXISTS "${dir}/CMakeLists.txt")
            message(STATUS "Adding plugin from directory ${dir}")
            add_subdirectory("${dir}")
        else ()
            message(WARNING "Directory ${dir} does not contain a CMakeLists.txt. Skipping.")
        endif ()
    endif ()
endfunction()

# # External dependencies

# CxxOpts dependency
define_external_target_header(CxxOpts
    https://github.com/jarro2783/cxxopts/archive/v3.1.1.tar.gz
    61b8bf0d8ab97fd55d67c7c25687b26d
    include
    ${DOWNLOAD_CXXOPTS})

# JSON dependency
define_external_target_header(JSON
    https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    c23a33f04786d85c29fda8d16b5f0efd
    include
    ${DOWNLOAD_JSON})

# pybind11
if (BUILD_PYBIND11)
    # Find Pybind11
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)
endif (BUILD_PYBIND11)

# # Targets
add_subdirectory(src)
add_subdirectory(executables)

# # Build plugins

# Gather all subdirectories in the plugins directory
file(GLOB PLUGIN_DIRS ${PLUGINS_BASE_DIR}/*)

# Loop through the plugin directories and add them as subdirectories
foreach (plugin ${PLUGIN_DIRS})
    check_and_add_subdirectory("${plugin}")
endforeach ()

# # Testing
if (BUILD_TESTS)
    include(CTest)
    # Code coverage
    # if(uppercase_CMAKE_BUILD_TYPE MATCHES DEBUG)
    # 	list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake-modules")
    # 	if(CMAKE_COMPILER_IS_GNUCXX)
    # 	  include(CodeCoverage)
    # 	  append_coverage_compiler_flags()
    # 	  setup_target_for_coverage_lcov(NAME coverage
    # 		EXECUTABLE ctest
    # 		DEPENDENCIES
    # 	    test_runner)
    # 	endif()
    # endif()

    # Enable tests
    enable_testing()

    # Add Catch2 dependency
    define_external_target_header(Catch
        https://github.com/catchorg/Catch2/archive/v2.13.10.tar.gz
        7a4dd2fd14fb9f46198eb670ac7834b7
        single_include/catch2
        ${DOWNLOAD_CATCH2})

    # Test target
    add_subdirectory(unit_tests)
    add_subdirectory(integration_tests)
endif ()
