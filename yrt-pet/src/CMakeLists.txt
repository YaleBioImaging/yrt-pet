project(src)

# List source files explicitly
set(SOURCES
    datastruct/projection/BinIterator.cpp
    datastruct/projection/Histogram.cpp
    datastruct/projection/Histogram3D.cpp
    datastruct/projection/ListMode.cpp
    datastruct/projection/ListModeLUT.cpp
    datastruct/projection/ListModeLUTDOI.cpp
    datastruct/projection/LORMotion.cpp
    datastruct/projection/ProjectionData.cpp
    datastruct/projection/BinFilter.cpp
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
    operators/OperatorProjectorBase.cpp
    operators/OperatorProjector.cpp
    operators/OperatorProjectorSiddon.cpp
    operators/OperatorProjectorDD.cpp
    operators/OperatorPsf.cpp
    operators/OperatorVarPsf.cpp
    operators/ProjectionPsfManager.cpp
    operators/SparseProjection.cpp
    operators/TimeOfFlight.cpp
    operators/Variable.cpp
    recon/Corrector.cpp
    recon/Corrector_CPU.cpp
    recon/OSEMUpdater_CPU.cpp
    recon/OSEM.cpp
    recon/OSEM_CPU.cpp
    scatter/Crystal.cpp
    scatter/ScatterEstimator.cpp
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
    utils/Utilities.cpp)

# cuda
if (USE_CUDA)
    set(SOURCES "${SOURCES}"
        datastruct/scanner/ScannerDevice.cu
        datastruct/image/ImageDevice.cu
        datastruct/image/ImageSpaceKernels.cu
        datastruct/projection/LORsDevice.cu
        datastruct/projection/ProjectionDataDevice.cu
        datastruct/projection/ProjectionSpaceKernels.cu
        operators/DeviceSynchronized.cu
        operators/OperatorProjectorDevice.cu
        operators/OperatorPsfDevice.cu
            operators/OperatorVarPsfDevice.cu
        operators/OperatorProjectorDD_GPUKernels.cu
        operators/OperatorProjectorSiddon_GPUKernels.cu
        operators/OperatorProjectorDD_GPU.cu
        operators/OperatorProjectorSiddon_GPU.cu
        operators/ProjectionPsfManagerDevice.cu
        recon/Corrector_GPU.cu
        recon/OSEMUpdater_GPU.cu
        recon/OSEM_GPU.cu
        utils/GPUStream.cu
        utils/GPUUtils.cu
        utils/GPUKernelUtils.cu
        utils/PageLockedBuffer.cu
        utils/ReconstructionUtilsDevice.cu
    )
endif (USE_CUDA)

define_yrtpet_lib("${SOURCES}")

if (USE_CUDA)
    set_property(TARGET ${YRTPET_LIB_NAME} PROPERTY CUDA_ARCHITECTURES ${YRTPET_CUDA_ARCHITECTURES})
endif ()

# pybind111
set(SOURCES_PYBIND11
    pybind11.cpp)

if (BUILD_PYBIND11)
    define_target_pybind11(py${YRTPET_LIB_NAME} "${SOURCES_PYBIND11}" ${CMAKE_SOURCE_DIR})
endif (BUILD_PYBIND11)
