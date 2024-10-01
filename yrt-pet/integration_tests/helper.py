import os
import sys

import numpy as np

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(fold_py))
import pyyrtpet as yrt


# %% Test folders
def get_test_folders():
    env_data = "YRTPET_TEST_DATA"
    env_out = "YRTPET_TEST_OUT"

    fold_data_from_env = os.getenv(env_data)
    fold_out_from_env = os.getenv(env_out)
    if fold_data_from_env is None or fold_out_from_env is None:
        raise RuntimeError("environment variables " + env_data
                           + " and " + env_out + " need to be set")

    fold_bin = os.path.join(os.path.dirname(__file__), '../executables')
    return fold_data_from_env, fold_out_from_env, fold_bin


# %% Helper test functions
# Note: this works only for ListModes
def _test_reconstruction(img_params, scanner, dataset, sens_img,
                         out_img_file: str, ref_img_file: str,
                         attenuationImage=None, warper=None,
                         num_MLEM_iterations=30, num_OSEM_subsets=1,
                         hard_threshold=1.0, num_threads=-1,
                         tof_width_ps=None, tof_n_std=None,
                         proj_psf_fname=None, num_rays=1):
    out_img = yrt.ImageOwned(img_params)
    out_img.allocate()
    out_img.setValue(0.0)

    osem = yrt.createOSEM(scanner)
    osem.imageParams = img_params
    osem.num_MLEM_iterations = num_MLEM_iterations
    osem.num_OSEM_subsets = num_OSEM_subsets
    osem.hardThreshold = hard_threshold
    osem.numRays = num_rays
    yrt.Globals.set_num_threads(num_threads)
    osem.setDataInput(dataset)
    if tof_width_ps is not None and tof_n_std is not None:
        osem.addTOF(tof_width_ps, tof_n_std)
    if proj_psf_fname is not None:
        osem.addProjPSF(proj_psf_fname)

    osem.registerSensitivityImages([sens_img])

    if warper is not None:
        osem.warper = warper
    if attenuationImage is not None:
        osem.attenuationImage = attenuationImage
    osem.outImage = out_img

    # Launch Reconstruction
    if warper is None:
        osem.reconstruct()
    else:
        osem.reconstructWithWarperMotion()
    out_img.writeToFile(out_img_file)

    ref_img = yrt.ImageOwned(img_params, ref_img_file)
    rmse = get_rmse(np.array(out_img, copy=False),
                    np.array(ref_img, copy=False))
    assert rmse < 10**-4


def _test_subsets(scanner: yrt.Scanner, img_params: yrt.ImageParams,
                  projData: yrt.ProjectionData, **args):
    k = yrt.ProjectionOper(scanner, img_params, projData, **args)

    x = np.random.random([img_params.nz, img_params.ny, img_params.nx])
    y = np.random.random(projData.count())

    Ax = k.A(x)

    num_subsets = 4
    Ax_sub = np.zeros_like(Ax)
    for subset in range(num_subsets):
        k_sub = yrt.ProjectionOper(
            scanner,
            img_params, projData,
            idx_subset=subset, num_subsets=num_subsets,
            **args)
        Ax_s = k_sub.A(x)
        Ax_sub += Ax_s
        # Adjoint test
        if subset == 2:
            Aty_s = k_sub.At(y)
            np.testing.assert_allclose(
                np.sum(Ax_s * y), np.sum(x * Aty_s), atol=0, rtol=1e-3)
    # Check that combination of subsets yields full projection
    np.testing.assert_allclose(Ax, Ax_sub)


def _test_adjoint(scanner: yrt.Scanner, img_params: yrt.ImageParams,
                  projData: yrt.ProjectionData, **args):
    k = yrt.ProjectionOper(scanner, img_params, projData, **args)

    x = np.random.random([img_params.nz, img_params.ny, img_params.nx])
    y = np.random.random(projData.count())

    Ax = k.A(x)
    ATy = k.At(y)

    lhs = np.sum(Ax * y)
    rhs = np.sum(x * ATy)
    np.testing.assert_allclose(lhs, rhs, atol=0, rtol=1e-3)


def join_file_path_recursive(prefix, path):
    if isinstance(path, (list, tuple)):
        for i in range(len(path)):
            path[i] = join_file_path_recursive(prefix, path[i])
        return path
    return os.path.join(prefix, path)


def join_file_paths(dataset_paths, out_paths, ref_paths, util_paths,
                    fold_data: str, fold_out: str):
    for key, value in dataset_paths.items():
        dataset_paths[key] = join_file_path_recursive(
            fold_data, dataset_paths[key])
    for key, value in util_paths.items():
        util_paths[key] = join_file_path_recursive(fold_data, util_paths[key])
    for key, value in out_paths.items():
        out_paths[key] = join_file_path_recursive(fold_out, out_paths[key])
    for key, value in ref_paths.items():
        ref_paths[key] = join_file_path_recursive(
            os.path.join(fold_data, 'ref'), ref_paths[key])


# Test helpers
def get_test_summary(x0, x1):
    x0_max = np.max(np.abs(x0))
    x0_mean = np.mean(x0)
    x0_median = np.median(x0)

    rmse = np.sqrt(np.mean((x0 - x1)**2))
    nrmse = rmse / np.sqrt(np.mean(x0**2))
    linf = np.max(np.abs(x0 - x1))
    npix_diff = np.size(np.nonzero(x0 != x1)[0])
    return {'x0_max': x0_max,
            'x0_mean': x0_mean,
            'x0_median': x0_median,
            'rmse': rmse,
            'nrmse': nrmse,
            'linf': linf,
            'npix_diff': npix_diff}


def get_npix_diff(x0, x1):
    return np.size(np.nonzero(x0 != x1)[0])


def get_linf(x0, x1):
    return np.max(np.abs(x0 - x1))


def get_rmse(x0, x1):
    return np.sqrt(np.mean((x0 - x1)**2))


def get_nrmse(x0, x1):
    rmse = get_rmse(x0, x1)
    return rmse / np.sqrt(np.mean(x0**2))


# %% Datasets

dataset_paths = {'test_mlem_simple': 'offset_UMHotSpot_noMotion.lmDat',
                 'test_mlem_piston': ['offset_UMHotSpot_piston.lmDat',
                                      'piston_ref0.twp'],
                 'test_mlem_wobble': ['offset_UMHotSpot_wobble.lmDat',
                                      'wobble_ref0.twp'],
                 'test_mlem_yesMan': ['offset_UMHotSpot_yesMan.lmDat',
                                      'yesMan_ref0.twp'],
                 'test_bwd': 'offset_UMHotSpot_noMotion.lmDat',
                 'test_post_recon_mc_piston': [
                     'images/pistonGt_part1Of5.fraw.img',
                     'images/pistonGt_part2Of5.fraw.img',
                     'images/pistonGt_part3Of5.fraw.img',
                     'images/pistonGt_part4Of5.fraw.img',
                     'images/pistonGt_part5Of5.fraw.img',
                     'piston_ref0.twp'],
                 'test_post_recon_mc_wobble': [
                     'images/wobbleGt_part1Of5.fraw.img',
                     'images/wobbleGt_part2Of5.fraw.img',
                     'images/wobbleGt_part3Of5.fraw.img',
                     'images/wobbleGt_part4Of5.fraw.img',
                     'images/wobbleGt_part5Of5.fraw.img',
                     'wobble_ref0.twp'],
                 'test_psf': ['psf_im_in.img', 'psfKernelImageSpace.csv'],
                 'test_flat_panel_mlem_tof': [
                     'out_Large_panels_4min_singTres49ps_0_gc_R30_r0.lmDat',
                     'test_flat_sens_S.img', 'test_flat_sens_DD_GPU.img'],
                 'test_subsets_savant': 'offset_UMHotSpot_noMotion.lmDat',
                 'test_adjoint_uhr2d': 'shepp_logan_2d.lmDat',
                 'test_osem_his_2d': 'shepp_logan_2d.his',
                 'test_osem_siddon_multi_ray':
                     'offset_UMHotSpot_noMotion.lmDat'}

out_paths = {'test_mlem_simple': 'test_mlem_simple.img',
             'test_mlem_piston': 'test_mlem_piston.img',
             'test_mlem_wobble': 'test_mlem_wobble.img',
             'test_mlem_yesMan': 'test_mlem_yesMan.img',
             'test_sens': 'test_sens.img',
             'test_bwd': 'test_bwd.img',
             'test_post_recon_mc_piston': 'test_post_recon_mc_piston.img',
             'test_post_recon_mc_wobble': 'test_post_recon_mc_wobble.img',
             'test_psf': 'psf_im_out.img',
             'test_flat_panel_mlem_tof': 'test_flat_img_S.img',
             'test_flat_panel_mlem_tof_exec': 'test_flat_img_DD_GPU.img',
             'test_osem_his_2d': ['test_osem_his_2d.img',
                                  'UHR2D-SensImg-5Subsets.img',
                                  ['UHR2D-SensImg-5Subsets_subset' + str(i) +
                                   '.img' for i in range(5)]],
             'test_dd_cpu_gpu_similarity':
                 ['test_dd_cpu.img', 'test_dd_gpu.img',
                  'test_dd_cpu_sens.img', 'test_dd_gpu_sens.img'],
             'test_dd_cpu_gpu_similarity_exec':
                 ['test_dd_cpu_exec.img', 'test_dd_gpu_exec.img',
                  'test_dd_cpu_sens_exec.img', 'test_dd_gpu_sens_exec.img'],
             'test_osem_siddon_multi_ray': 'test_osem_siddon_multi_ray.img'}

ref_paths = {'test_mlem_simple': 'test_mlem_simple_ref.img',
             'test_mlem_piston': 'test_mlem_piston_ref.img',
             'test_mlem_wobble': 'test_mlem_wobble_ref.img',
             'test_mlem_yesMan': 'test_mlem_yesMan_ref.img',
             'test_sens': 'test_sens_ref.img',
             'test_bwd': 'test_bwd_ref.img',
             'test_post_recon_mc_piston': 'test_post_recon_mc_piston_ref.img',
             'test_post_recon_mc_wobble': 'test_post_recon_mc_wobble_ref.img',
             'test_psf': 'psf_im_ref.img',
             'test_flat_panel_mlem_tof': ['test_flat_img_S.img',
                                          'test_flat_img_DD_GPU.img'],
             'test_osem_his_2d': ['test_osem_his_2d.img',
                                  ['UHR2D-SensImg-5Subsets_subset' + str(i) +
                                   '.img' for i in range(5)]],
             'test_osem_siddon_multi_ray': 'test_osem_siddon_multi_ray.img'}

util_paths = {'img_params_500': 'config/img_params_500.json',
              'img_params_3.0': 'config/img_params_3.0.json',
              'img_params_192': 'config/img_params_192.json',
              'img_params_2mm': 'config/img_params_2mm.json',
              'img_params_2d': 'config/img_params_2d.json',
              'SAVANT_json': 'config/SAVANT.json',
              'UHR2D_json': 'config/UHR2D.json',
              'Geometry_2panels_large_3x3x20mm_rot_gc_json':
                  'config/Geometry_2panels_large_3x3x20mm_rot_gc.json',
              'SensImageSAVANT500': 'images/SensImageSAVANT500.img',
              'sens_SAVANT_multi_ray_500':
                  'images/sens_SAVANT_multi_ray_500.img'}

# Preend data path
fold_data, fold_out, fold_bin = get_test_folders()
join_file_paths(dataset_paths, out_paths, ref_paths, util_paths,
                fold_data, fold_out)
