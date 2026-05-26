project(executables)

define_target_exe(yrtpet_reconstruct recon/Reconstruct.cpp PluginOptionsHelper.cpp ArgumentReader.cpp)
define_target_exe(yrtpet_forward_project recon/ForwardProject.cpp ArgumentReader.cpp)
define_target_exe(yrtpet_backproject recon/Backproject.cpp PluginOptionsHelper.cpp ArgumentReader.cpp)
define_target_exe(yrtpet_estimate_scatter recon/EstimateScatter.cpp PluginOptionsHelper.cpp ArgumentReader.cpp)

define_target_exe(yrtpet_convert_to_histogram conversion/ConvertToHistogram.cpp PluginOptionsHelper.cpp ArgumentReader.cpp)
define_target_exe(yrtpet_convert_to_listmodelut conversion/ConvertToListModeLUT.cpp PluginOptionsHelper.cpp ArgumentReader.cpp)
define_target_exe(yrtpet_convert_to_low_count_listmodelut conversion/ConvertToLowCountListModeLUT.cpp PluginOptionsHelper.cpp ArgumentReader.cpp)
define_target_exe(yrtpet_cumulate_histograms conversion/CumulateHistograms.cpp PluginOptionsHelper.cpp ArgumentReader.cpp)
define_target_exe(yrtpet_histogram_to_listmode conversion/Histogram3DToListMode.cpp ArgumentReader.cpp)

define_target_exe(yrtpet_build_k kernel/BuildK.cpp)

define_target_exe(yrtpet_accumulate_to_detectors utils/AccumulateToDetectors.cpp PluginOptionsHelper.cpp ArgumentReader.cpp)
define_target_exe(yrtpet_post_recon_motion_correction utils/PostReconMotionCorrection.cpp)

if (USE_METAL)
    define_target_exe(yrtpet_metal_smoke utils/MetalSmoke.cpp)
    add_dependencies(yrtpet_metal_smoke yrtpet_metal_kernels)
    define_target_exe(yrtpet_metal_backend_sample
        utils/MetalBackendSample.cpp)
    add_dependencies(yrtpet_metal_backend_sample yrtpet_metal_kernels)
    define_target_exe(yrtpet_metal_psf_compare utils/MetalPsfCompare.cpp)
    add_dependencies(yrtpet_metal_psf_compare yrtpet_metal_kernels)
    define_target_exe(yrtpet_metal_projector_compare
        utils/MetalProjectorCompare.cpp)
    add_dependencies(yrtpet_metal_projector_compare yrtpet_metal_kernels)
    define_target_exe(yrtpet_metal_osem_compare
        utils/MetalOsemCompare.cpp)
    add_dependencies(yrtpet_metal_osem_compare yrtpet_metal_kernels)
endif ()
