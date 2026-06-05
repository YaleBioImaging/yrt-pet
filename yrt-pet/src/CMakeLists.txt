project(src)

# List source files explicitly
set(SOURCES
    datastruct/projection/BinIterator.cpp
    datastruct/projection/DynamicFraming.cpp
    datastruct/projection/Histogram.cpp
    datastruct/projection/Histogram3D.cpp
    datastruct/projection/ListMode.cpp
    datastruct/projection/ListModeLUT.cpp
    datastruct/projection/ListModeLUTDOI.cpp
    datastruct/projection/LORMotion.cpp
    datastruct/projection/ProjectionData.cpp
    datastruct/projection/BinFilter.cpp
    datastruct/projection/BinLoader.cpp
    datastruct/projection/Constraints.cpp
    datastruct/projection/ProjectionList.cpp
    datastruct/projection/SparseHistogram.cpp
    datastruct/projection/UniformHistogram.cpp
    datastruct/projection/ProjectionProperties.cpp
    datastruct/scanner/DetCoord.cpp
    datastruct/scanner/DetectorMask.cpp
    datastruct/scanner/DetectorSetup.cpp
    datastruct/scanner/DetRegular.cpp
    datastruct/scanner/Scanner.cpp
    datastruct/image/Image.cpp
    datastruct/image/ImageBase.cpp
    datastruct/image/ImageParams.cpp
    datastruct/image/nifti/nifti1_io.cpp
    datastruct/image/nifti/znzlib.cpp
    datastruct/IO.cpp
    geometry/Cylinder.cpp
    geometry/Line3D.cpp
    geometry/Matrix.cpp
    geometry/MultiRayGenerator.cpp
    geometry/Plane.cpp
    geometry/TransformUtils.cpp
    geometry/TubeOfResponse.cpp
    geometry/Vector3D.cpp
    kernel/Kernel.cpp
    kinetic/SRTM.cpp
    operators/Operator.cpp
    operators/ProjectorParams.cpp
    operators/OperatorProjectorBase.cpp
    operators/OperatorProjector.cpp
    operators/ProjectorUpdater.cpp
    operators/OperatorPsf.cpp
    operators/OperatorVarPsf.cpp
    operators/Projector.cpp
    operators/ProjectorSiddon.cpp
    operators/ProjectorDD.cpp
    operators/ProjectionPsfManager.cpp
    operators/SparseProjection.cpp
    operators/TimeOfFlight.cpp
    operators/Variable.cpp
    recon/Corrector.cpp
    recon/Corrector_CPU.cpp
    recon/OSEM.cpp
    recon/OSEM_CPU.cpp
    recon/LREM.cpp
    recon/LREM_CPU.cpp
    scatter/Crystal.cpp
    scatter/ScatterEstimator.cpp
    scatter/ScatterSpace.cpp
    scatter/SingleScatterSimulator.cpp
    utils/Array.cpp
    utils/Assert.cpp
    utils/FileReader.cpp
    utils/Globals.cpp
    utils/Logger.cpp
    utils/ProgressDisplay.cpp
    utils/ProgressDisplayMultiThread.cpp
    utils/RangeList.cpp
    utils/ReconstructionUtils.cpp
    utils/Timer.cpp
    utils/Tools.cpp
    utils/Utilities.cpp
    utils/Version.cpp)

# cuda
if (USE_CUDA)
    set(SOURCES "${SOURCES}"
        datastruct/scanner/ScannerDevice.cu
        datastruct/image/ImageDevice.cu
        datastruct/image/ImageSpaceKernels.cu
        datastruct/projection/LORsDevice.cu
        datastruct/projection/ProjectionListDevice.cu
        operators/ProjectionSpaceKernels.cu
        operators/DeviceSynchronized.cu
        operators/OperatorProjectorDevice.cu
        operators/OperatorPsfDevice.cu
        operators/OperatorVarPsfDevice.cu
        operators/OperatorProjectorDD_GPU.cu
        operators/OperatorProjectorSiddon_GPU.cu
        operators/ProjectionPsfManagerDevice.cu
        recon/Corrector_GPU.cu
        recon/OSEM_GPU.cu
        recon/LREM_GPU.cu
        utils/GPUStream.cu
        utils/GPUTypes.cu
        utils/GPUUtils.cu
        utils/GPUKernelUtils.cu
        utils/PageLockedBuffer.cu
        utils/ReconstructionUtilsDevice.cu
    )
endif (USE_CUDA)

if (USE_METAL)
    set(YRTPET_METAL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/backends/metal")
    set(YRTPET_METAL_SMOKE_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/SmokeKernels.metal")
    set(YRTPET_METAL_SMOKE_AIR
        "${YRTPET_METAL_OUTPUT_DIR}/SmokeKernels.air")
    set(YRTPET_METAL_SMOKE_METALLIB
        "${YRTPET_METAL_OUTPUT_DIR}/SmokeKernels.metallib")
    set(YRTPET_METAL_MODULE_CACHE_DIR
        "${YRTPET_METAL_OUTPUT_DIR}/ModuleCache")

    add_custom_command(
        OUTPUT "${YRTPET_METAL_SMOKE_AIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${YRTPET_METAL_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${YRTPET_METAL_MODULE_CACHE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E env
            "CLANG_MODULE_CACHE_PATH=${YRTPET_METAL_MODULE_CACHE_DIR}"
            ${YRTPET_XCRUN_EXECUTABLE} -sdk macosx metal
            "-fmodules-cache-path=${YRTPET_METAL_MODULE_CACHE_DIR}"
            -fmetal-math-mode=safe
            -fmetal-math-fp32-functions=precise
            -ffp-contract=off
            -c "${YRTPET_METAL_SMOKE_SOURCE}"
            -o "${YRTPET_METAL_SMOKE_AIR}"
        DEPENDS "${YRTPET_METAL_SMOKE_SOURCE}"
        VERBATIM)
    add_custom_command(
        OUTPUT "${YRTPET_METAL_SMOKE_METALLIB}"
        COMMAND ${YRTPET_XCRUN_EXECUTABLE} -sdk macosx metallib
            "${YRTPET_METAL_SMOKE_AIR}"
            -o "${YRTPET_METAL_SMOKE_METALLIB}"
        DEPENDS "${YRTPET_METAL_SMOKE_AIR}"
        VERBATIM)
    add_custom_target(yrtpet_metal_kernels
        DEPENDS "${YRTPET_METAL_SMOKE_METALLIB}")

    set(YRTPET_METAL_SMOKE_METALLIB "${YRTPET_METAL_SMOKE_METALLIB}"
        CACHE INTERNAL "Path to the Metal smoke test metallib")

    list(APPEND SOURCES
        backends/metal/ExperimentalBackend.cpp
        backends/metal/ImageMetal.cpp
        backends/metal/ImageOps.cpp
        backends/metal/ImageSpaceKernels.cpp
        backends/metal/JosephProjectorKernels.cpp
        backends/metal/JosephProjectorMetal.cpp
        backends/metal/JosephProjectorOps.cpp
        backends/metal/MetalBackend.cpp
        backends/metal/MetalContext.cpp
        backends/metal/MetalSmoke.cpp
        backends/metal/MotionOps.cpp
        backends/metal/OperatorProjectorMetalBridge.cpp
        backends/metal/OperatorPsfMetal.cpp
        backends/metal/PsfFileOps.cpp
        backends/metal/PsfOps.cpp
        backends/metal/ProjectionBatchMetal.cpp
        backends/metal/ProjectionGeometryKernels.cpp
        backends/metal/ProjectionGeometryOps.cpp
        backends/metal/ProjectionVectorMetal.cpp
        backends/metal/ProjectionVectorOps.cpp
        backends/metal/ProjectionVectorKernels.cpp
        backends/metal/SiddonProjectorKernels.cpp
        backends/metal/SiddonProjectorMetal.cpp
        backends/metal/SiddonProjectorOps.cpp)
    set_source_files_properties(backends/metal/MetalBackend.cpp
        PROPERTIES COMPILE_OPTIONS "-fblocks")
endif ()

define_yrtpet_lib("${SOURCES}")

if (USE_CUDA)
    set_property(TARGET ${YRTPET_LIB_NAME} PROPERTY CUDA_ARCHITECTURES ${YRTPET_CUDA_ARCHITECTURES})
endif ()

if (USE_METAL)
    target_compile_definitions(${YRTPET_LIB_NAME} PRIVATE
        YRTPET_METAL_SMOKE_METALLIB="${YRTPET_METAL_SMOKE_METALLIB}")
    if (BUILD_SHARED_LIB)
        target_compile_definitions(${YRTPET_LIB_NAME}_shared PRIVATE
            YRTPET_METAL_SMOKE_METALLIB="${YRTPET_METAL_SMOKE_METALLIB}")
    endif ()
endif ()

# pybind111
set(SOURCES_PYBIND11
    pybind11.cpp)

if (BUILD_PYBIND11)
    define_target_pybind11(py${YRTPET_LIB_NAME} "${SOURCES_PYBIND11}" ${CMAKE_SOURCE_DIR})
endif (BUILD_PYBIND11)
