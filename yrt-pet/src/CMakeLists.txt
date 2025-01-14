project(src)

# List source files explicitly
set(SOURCES
        datastruct/projection/Histogram.cpp
        datastruct/projection/Histogram3D.cpp
        datastruct/projection/UniformHistogram.cpp
        datastruct/projection/ListModeLUT.cpp
        datastruct/projection/ListModeLUTDOI.cpp
        datastruct/projection/LORMotion.cpp
        datastruct/projection/BinIterator.cpp
        datastruct/projection/ProjectionList.cpp
        datastruct/projection/SparseHistogram.cpp
        datastruct/projection/ProjectionData.cpp
        datastruct/projection/ListMode.cpp
        datastruct/scanner/DetCoord.cpp
        datastruct/scanner/DetRegular.cpp
        datastruct/scanner/DetectorSetup.cpp
        datastruct/scanner/Scanner.cpp
        datastruct/image/ImageBase.cpp
        datastruct/image/Image.cpp
        datastruct/image/nifti/nifti1_io.cpp
        datastruct/image/nifti/znzlib.cpp
        datastruct/IO.cpp
        geometry/Matrix.cpp
        geometry/Vector3D.cpp
        geometry/Line3D.cpp
        geometry/MultiRayGenerator.cpp
        geometry/TubeOfResponse.cpp
        geometry/Cylinder.cpp
        geometry/Plane.cpp
        kernel/Kernel.cpp
        kinetic/SRTM.cpp
        motion/ImageWarperMatrix.cpp
        motion/ImageWarperTemplate.cpp
        motion/ImageWarperFunction.cpp
        operators/Operator.cpp
        operators/OperatorProjectorBase.cpp
        operators/OperatorProjector.cpp
        operators/OperatorProjectorSiddon.cpp
        operators/OperatorProjectorDD.cpp
        operators/OperatorPsf.cpp
        operators/ProjectionPsfManager.cpp
        operators/TimeOfFlight.cpp
        recon/Corrector.cpp
        recon/Corrector_CPU.cpp
        recon/OSEMUpdater_CPU.cpp
        recon/OSEM.cpp
        recon/OSEM_CPU.cpp
        operators/Variable.cpp
        scatter/ScatterEstimator.cpp
        scatter/SingleScatterSimulator.cpp
        scatter/Crystal.cpp
        utils/Assert.cpp
        utils/Utilities.cpp
        utils/RangeList.cpp
        utils/Tools.cpp
        utils/ProgressDisplay.cpp
        utils/ProgressDisplayMultiThread.cpp
        utils/ReconstructionUtils.cpp
        utils/Array.cpp
        utils/FileReader.cpp
        utils/Globals.cpp)

# cuda
if (USE_CUDA)
    set(SOURCES "${SOURCES}"
            datastruct/scanner/ScannerDevice.cu
            datastruct/image/ImageDevice.cu
            datastruct/image/ImageSpaceKernels.cu
            datastruct/projection/LORsDevice.cu
            datastruct/projection/ProjectionDataDevice.cu
            datastruct/projection/ProjectionSpaceKernels.cu
            operators/OperatorDevice.cu
            operators/OperatorPsfDevice.cu
            operators/ProjectionPsfManagerDevice.cu
            operators/OperatorProjectorDD_GPUKernels.cu
            operators/OperatorProjectorDD_GPU.cu
            recon/OSEM_GPU.cu
            utils/PageLockedBuffer.cu
            utils/GPUStream.cu
            utils/GPUUtils.cu
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
