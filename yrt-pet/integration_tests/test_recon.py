#!/bin/env python
"""Integration tests for YRT-PET"""

# %% Imports

import os
import sys
import pytest
import numpy as np

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(fold_py))
import pyyrtpet as yrt

yrt.setNumThreads(-1)

import helper as _helper

fold_data = _helper.fold_data
fold_out = _helper.fold_out
fold_bin = _helper.fold_bin


# Nomenclature of the tests:
#  test_<scanner>_<dataset>_<algorithms tested>_<running conditions>
#  Examples of "algorithms tested": osem, dd (distance-driven), bwd
#  Examples of running conditions: gpu/cpu, exec
#  If the projector is not specified, it is assumed to be Siddon
#  If neither CPU or GPU is specified, test is assumed to be ran on the CPU
# %% Tests

def test_multithreading():
    yrt.setNumThreads(8)
    num_threads = yrt.getNumThreads()
    assert (num_threads == 8)
    yrt.setNumThreads(-1)


def test_savant_sim_ultra_micro_hotspot_nomotion_mlem():
    fold_savant_sim = os.path.join(fold_data, 'savant_sim')
    scanner = yrt.Scanner(os.path.join(fold_savant_sim, 'SAVANT_sim.json'))
    listmode = yrt.ListModeLUTOwned(scanner, os.path.join(
        fold_savant_sim, 'ultra_micro_hotspot', 'nomotion.lmDat'))
    image_params = yrt.ImageParams(
        os.path.join(fold_savant_sim, 'img_params_500.json'))
    sens_image = yrt.ImageOwned(image_params, os.path.join(
        fold_savant_sim, 'images', 'sens_image_siddon.nii'))

    osem = yrt.createOSEM(scanner, False)
    osem.setSensitivityImage(sens_image)  # One subset
    osem.setDataInput(listmode)
    osem.num_MLEM_iterations = 30
    osem.num_OSEM_subsets = 1
    recon_image = osem.reconstruct()

    recon_image.writeToFile(os.path.join(
        fold_out, 'test_savant_sim_ultra_micro_hotspot_nomotion_mlem.nii.gz'))

    recon_image_np = np.array(recon_image, copy=False)
    ref_image = yrt.ImageOwned(image_params, os.path.join(
        fold_savant_sim, 'ref', 'ultra_micro_hotspot_nomotion_mlem.nii'))
    ref_image_np = np.array(ref_image, copy=False)

    np.testing.assert_allclose(recon_image_np, ref_image_np,
                               atol=0, rtol=5e-2)


def _test_savant_sim_ultra_micro_hotspot_motion_mlem_gpu_exec(
        keyword: str, proj_keyword: str):
    if not yrt.compiledWithCuda():
        pytest.skip('Code not compiled with cuda. Skipping...')

    proj_name_upper = proj_keyword.upper()

    fold_savant_sim = os.path.join(fold_data, 'savant_sim')
    out_path = os.path.join(fold_out, 'test_savant_sim_ultra_micro_hotspot_' +
                            keyword + '_mlem_' + proj_keyword +
                            '_gpu_exec.nii.gz')

    exec_str = os.path.join(fold_bin, 'yrtpet_reconstruct')
    exec_str += ' -s ' + os.path.join(fold_savant_sim, 'SAVANT_sim.json')
    exec_str += ' -i ' + os.path.join(fold_savant_sim,
                                      'ultra_micro_hotspot',
                                      keyword + '.lmDat')
    exec_str += ' -f LM'
    exec_str += ' -p ' + os.path.join(fold_savant_sim, 'img_params_500.json')
    exec_str += ' --lor_motion ' + os.path.join(fold_savant_sim,
                                                'ultra_micro_hotspot',
                                                keyword + '.vc')
    exec_str += ' -o ' + out_path
    exec_str += ' --projector ' + proj_name_upper + ' --gpu'

    ret = os.system(exec_str)
    assert ret == 0

    out_image = yrt.ImageOwned(out_path)
    out_image_np = np.array(out_image, copy=False)

    ref_image = yrt.ImageOwned(
        os.path.join(fold_savant_sim,
                     'ref',
                     'ultra_micro_hotspot_' + keyword + '_mlem_' \
                     + proj_keyword + '.nii.gz'))
    ref_image_np = np.array(ref_image, copy=False)

    # Allow up to 40 voxels to mismatch
    assert (1 - np.isclose(out_image_np, ref_image_np,
                           atol=0, rtol=1e-3)).sum() < 40


def test_savant_sim_ultra_micro_hotspot_piston_mlem_dd_gpu_exec():
    _test_savant_sim_ultra_micro_hotspot_motion_mlem_gpu_exec(
        'piston', 'dd')


def test_savant_sim_ultra_micro_hotspot_yesman_mlem_dd_gpu_exec():
    _test_savant_sim_ultra_micro_hotspot_motion_mlem_gpu_exec(
        'yesman', 'dd')


def test_savant_sim_ultra_micro_hotspot_wobble_mlem_dd_gpu_exec():
    _test_savant_sim_ultra_micro_hotspot_motion_mlem_gpu_exec(
        'wobble', 'dd')


def test_savant_sim_ultra_micro_hotspot_piston_mlem_siddon_gpu_exec():
    _test_savant_sim_ultra_micro_hotspot_motion_mlem_gpu_exec(
        'piston', 'siddon')


def test_savant_sim_ultra_micro_hotspot_yesman_mlem_siddon_gpu_exec():
    _test_savant_sim_ultra_micro_hotspot_motion_mlem_gpu_exec(
        'yesman', 'siddon')


def test_savant_sim_ultra_micro_hotspot_wobble_mlem_siddon_gpu_exec():
    _test_savant_sim_ultra_micro_hotspot_motion_mlem_gpu_exec(
        'wobble', 'siddon')


def test_savant_sim_sens_image_siddon():
    fold_savant_sim = os.path.join(fold_data, 'savant_sim')
    scanner = yrt.Scanner(os.path.join(fold_savant_sim, 'SAVANT_sim.json'))
    image_params = yrt.ImageParams(
        os.path.join(fold_savant_sim, 'img_params_500.json'))
    osem = yrt.createOSEM(scanner, False)
    osem.setImageParams(image_params)
    [sens_image] = osem.generateSensitivityImages()
    sens_image.writeToFile(
        os.path.join(fold_out, 'test_savant_sim_sens_image_siddon.nii.gz'))

    ref_image = yrt.ImageOwned(
        os.path.join(fold_savant_sim, 'images', 'sens_image_siddon.nii'))

    np.testing.assert_allclose(np.array(sens_image, copy=False),
                               np.array(ref_image, copy=False),
                               atol=0, rtol=1e-4)


def test_savant_sim_sens_image_siddon_exec():
    fold_savant_sim = os.path.join(fold_data, 'savant_sim')

    img_params_path = os.path.join(fold_savant_sim, 'img_params_500.json')
    out_image_path = os.path.join(
        fold_out, 'test_savant_sim_sens_image_siddon_exec.nii.gz')
    ref_image_path = os.path.join(fold_savant_sim, 'images',
                                  'sens_image_siddon.nii')

    exec_str = os.path.join(fold_bin, 'yrtpet_reconstruct') + ' --sens_only'
    exec_str += ' --scanner ' + os.path.join(fold_savant_sim,
                                             'SAVANT_sim.json')
    exec_str += ' --params ' + img_params_path
    exec_str += ' --out_sens ' + out_image_path
    ret = os.system(exec_str)
    assert ret == 0

    img_params = yrt.ImageParams(img_params_path)
    out_img = yrt.ImageOwned(img_params, out_image_path)
    ref_img = yrt.ImageOwned(img_params, ref_image_path)
    np.testing.assert_allclose(np.array(out_img, copy=False),
                               np.array(ref_img, copy=False),
                               atol=0, rtol=1e-4)


def test_savant_sim_ultra_micro_hotspot_nomotion_bwd():
    fold_savant_sim = os.path.join(fold_data, 'savant_sim')
    scanner = yrt.Scanner(os.path.join(fold_savant_sim, 'SAVANT_sim.json'))
    listmode = yrt.ListModeLUTOwned(scanner, os.path.join(
        fold_savant_sim, 'ultra_micro_hotspot', 'nomotion.lmDat'))
    image_params = yrt.ImageParams(
        os.path.join(fold_savant_sim, 'img_params_500.json'))
    bwd_image = yrt.ImageOwned(image_params)

    bwd_image.allocate()

    # Siddon backprojection by default
    yrt.backProject(scanner, bwd_image, listmode)

    # Force all valus less than one to be zero
    bwd_image.applyThreshold(bwd_image, 1, 0, 0, 1, 0)

    bwd_image.writeToFile(os.path.join(
        fold_out, 'test_savant_sim_ultra_micro_hotspot_nomotion_bwd.nii.gz'))

    ref_image = yrt.ImageOwned(os.path.join(
        fold_savant_sim, 'ref', 'ultra_micro_hotspot_nomotion_bwd.nii'))

    np.testing.assert_allclose(np.array(bwd_image, copy=False),
                               np.array(ref_image, copy=False),
                               atol=0, rtol=1e-5)


def _test_savant_sim_ultra_micro_hotspot_motion_post_recon_mc(keyword: str):
    fold_savant_sim = os.path.join(fold_data, 'savant_sim')
    img_params = yrt.ImageParams(
        os.path.join(fold_savant_sim, 'img_params_500.json'))

    file_format = os.path.join(fold_savant_sim, 'images',
                               keyword + '_ground_truth_part_{idx}_of_5.nii')
    file_list = list()
    for i in range(5):
        file_list.append(file_format.format(idx=(i + 1)))

    out_file = os.path.join(fold_out,
                            'test_savant_sim_ultra_micro_hotspot_' +
                            keyword + '_post_recon_mc.nii.gz')

    exec_str = os.path.join(fold_bin, 'yrtpet_post_recon_motion_correction')
    exec_str += ' --input ' + ','.join(file_list)
    exec_str += ' --lor_motion ' + os.path.join(fold_savant_sim,
                                                'ultra_micro_hotspot',
                                                keyword + '.vc')
    exec_str += ' --out ' + out_file
    print('Running: ' + exec_str)
    ret = os.system(exec_str)
    assert ret == 0

    out_img = yrt.ImageOwned(img_params, out_file)
    ref_img = yrt.ImageOwned(
        img_params, os.path.join(fold_savant_sim, 'ref',
                                 'ultra_micro_hotspot_' + keyword +
                                 '_post_recon_mc.nii.gz'))

    nrmse = _helper.get_nrmse(np.array(out_img, copy=False),
                              np.array(ref_img, copy=False))
    assert nrmse < 10e-6


def test_savant_sim_ultra_micro_hotspot_piston_post_recon_mc():
    _test_savant_sim_ultra_micro_hotspot_motion_post_recon_mc('piston')


def test_savant_sim_ultra_micro_hotspot_wobble_post_recon_mc():
    _test_savant_sim_ultra_micro_hotspot_motion_post_recon_mc('wobble')


def _test_psf(use_gpu: bool):
    img_params = yrt.ImageParams(50, 50, 25, 50, 50, 25, 0.0, 0.0, 0.0)
    image_in = yrt.ImageOwned(img_params,
                              os.path.join(fold_data, 'other', 'psf',
                                           'image_in.nii'))

    psf_file = os.path.join(fold_data, 'other', 'psf', 'kernel.csv')
    if use_gpu:
        psf_oper = yrt.OperatorPsfDevice(psf_file)
    else:
        psf_oper = yrt.OperatorPsf(psf_file)
    image_out = yrt.ImageOwned(img_params)
    image_out.allocate()

    psf_oper.applyA(image_in, image_out)

    device = 'gpu' if use_gpu else 'cpu'

    image_out.writeToFile(
        os.path.join(fold_out, 'test_psf_' + device + '.nii.gz'))
    image_ref = yrt.ImageOwned(img_params,
                               os.path.join(fold_data, 'other', 'ref',
                                            'psf_image_out.nii'))
    np.testing.assert_allclose(np.array(image_out, copy=False),
                               np.array(image_ref, copy=False),
                               atol=0, rtol=1e-5)


def test_psf_cpu():
    _test_psf(False)


def test_psf_gpu():
    if yrt.compiledWithCuda():
        _test_psf(True)
    else:
        pytest.skip('Code not compiled with cuda. Skipping...')


def _test_psf_adjoint(use_gpu: bool):
    rng = np.random.default_rng(13)

    nx = rng.integers(1, 30)
    ny = rng.integers(1, 30)
    nz = rng.integers(1, 20)
    sx = rng.random() * 5 + 0.01
    sy = rng.random() * 10 + 0.01
    sz = rng.random() * 10 + 0.01
    ox = 0.0
    oy = 0.0
    oz = 0.0
    img_params = yrt.ImageParams(nx, ny, nz, sx, sy, sz, ox, oy, oz)

    img_x = yrt.ImageAlias(img_params)
    img_y = yrt.ImageAlias(img_params)

    img_x_a = (rng.random([nz, ny, nx]) * 10 - 5).astype(np.float32)
    img_y_a = (rng.random([nz, ny, nx]) * 10 - 5).astype(np.float32)
    img_x.bind(img_x_a)
    img_y.bind(img_y_a)

    psf_file = os.path.join(fold_data, 'other', 'psf', 'kernel.csv')
    if use_gpu:
        oper_psf = yrt.OperatorPsfDevice(psf_file)
    else:
        oper_psf = yrt.OperatorPsf(psf_file)

    Ax = yrt.ImageOwned(img_params)
    Aty = yrt.ImageOwned(img_params)
    Ax.allocate()
    Ax.setValue(0.0)
    Aty.allocate()
    Aty.setValue(0.0)

    oper_psf.applyA(img_x, Ax)
    oper_psf.applyAH(img_y, Aty)

    dot_Ax_y = Ax.dotProduct(img_y)
    dot_x_Aty = img_x.dotProduct(Aty)

    assert np.abs(dot_Ax_y - dot_x_Aty) < 10 ** -4


def test_psf_adjoint_cpu():
    _test_psf_adjoint(False)


def test_psf_adjoint_gpu():
    if yrt.compiledWithCuda():
        _test_psf_adjoint(True)
    else:
        pytest.skip('Code not compiled with cuda. Skipping...')


def _test_savant_sim_ultra_micro_hotspot_nomotion_osem_6rays(use_gpu: bool):
    fold_savant_sim = os.path.join(fold_data, 'savant_sim')
    num_siddon_rays = 6
    img_params = yrt.ImageParams(
        os.path.join(fold_savant_sim, 'img_params_500.json'))
    scanner = yrt.Scanner(os.path.join(fold_savant_sim, 'SAVANT_sim.json'))
    dataset = yrt.ListModeLUTOwned(
        scanner,
        os.path.join(fold_savant_sim, 'ultra_micro_hotspot',
                     'nomotion.lmDat'))
    gpu_str = 'gpu' if use_gpu else 'cpu'
    sens_img = yrt.ImageOwned(
        img_params, os.path.join(fold_savant_sim, 'images',
                                 f'sens_image_siddon_6rays_{gpu_str}.nii.gz'))

    osem = yrt.createOSEM(scanner, use_gpu)
    osem.setImageParams(img_params)
    osem.num_MLEM_iterations = 3
    osem.num_OSEM_subsets = 12
    osem.numRays = num_siddon_rays
    osem.setDataInput(dataset)
    osem.setSensitivityImage(sens_img)

    out_img = osem.reconstruct()

    out_img_fname = 'test_savant_sim_ultra_micro_hotspot_nomotion_osem_6rays_'
    if use_gpu:
        out_img_fname += 'gpu'
    else:
        out_img_fname += 'cpu'
    out_img_fname += '.nii.gz'

    out_img.writeToFile(os.path.join(fold_out, out_img_fname))

    # The reference is different between CPU and GPU
    ref_img_fname = 'ultra_micro_hotspot_nomotion_osem_6rays_'
    if use_gpu:
        ref_img_fname += 'gpu'
    else:
        ref_img_fname += 'cpu'
    ref_img_fname += '.nii.gz'

    ref_img = yrt.ImageOwned(img_params,
                             os.path.join(fold_savant_sim,
                                          'ref', ref_img_fname))

    out_img_np = np.array(out_img, copy=False)
    ref_img_np = np.array(ref_img, copy=False)
    is_close = np.isclose(ref_img_np, out_img_np, atol=0, rtol=1e-3)
    num_mismatch = (~is_close).sum()
    print(f'Number of mismatched voxels: {num_mismatch}')

    # Allow up to 10 mismatches
    assert num_mismatch < 5, f'Too many mismatched voxels: {num_mismatch}'


def test_savant_sim_ultra_micro_hotspot_nomotion_osem_6rays_cpu():
    _test_savant_sim_ultra_micro_hotspot_nomotion_osem_6rays(False)


def test_savant_sim_ultra_micro_hotspot_nomotion_osem_6rays_gpu():
    _test_savant_sim_ultra_micro_hotspot_nomotion_osem_6rays(True)


def _test_savant_sim_ultra_micro_hotpot_nomotion_subsets(projector: str):
    fold_savant_sim = os.path.join(fold_data, 'savant_sim')
    scanner = yrt.Scanner(os.path.join(fold_savant_sim, 'SAVANT_sim.json'))
    img_params = yrt.ImageParams(
        os.path.join(fold_savant_sim, 'img_params_500.json'))
    lm = yrt.ListModeLUTOwned(scanner, os.path.join(fold_savant_sim,
                                                    'ultra_micro_hotspot',
                                                    'nomotion.lmDat'))
    _helper._test_subsets(scanner, img_params, lm, projector=projector)


def test_savant_sim_ultra_micro_hotpot_nomotion_subsets_siddon():
    _test_savant_sim_ultra_micro_hotpot_nomotion_subsets('Siddon')


def test_savant_sim_ultra_micro_hotpot_nomotion_subsets_dd():
    _test_savant_sim_ultra_micro_hotpot_nomotion_subsets('DD')


def test_savant_sim_ultra_micro_hotpot_nomotion_subsets_dd_gpu():
    if yrt.compiledWithCuda():
        _test_savant_sim_ultra_micro_hotpot_nomotion_subsets('DD_GPU')
    else:
        pytest.skip('Code not compiled with cuda. Skipping...')


def _test_uhr2d_shepp_logan_adjoint(projector: str, num_rays: int = 1):
    fold_uhr2d = os.path.join(fold_data, 'uhr2d')
    scanner = yrt.Scanner(os.path.join(fold_uhr2d, 'UHR2D.json'))
    img_params = yrt.ImageParams(os.path.join(fold_uhr2d,
                                              'img_params_2d.json'))
    his = yrt.ListModeLUTOwned(scanner,
                               os.path.join(fold_uhr2d, 'shepp_logan.lmDat'))
    _helper._test_adjoint(scanner, img_params, his, projector=projector,
                          num_rays=num_rays)


def test_uhr2d_shepp_logan_adjoint_siddon():
    _test_uhr2d_shepp_logan_adjoint('Siddon')


def test_uhr2d_shepp_logan_adjoint_siddon_4rays():
    _test_uhr2d_shepp_logan_adjoint('Siddon', 4)


def test_uhr2d_shepp_logan_adjoint_dd():
    _test_uhr2d_shepp_logan_adjoint('DD')


def test_uhr2d_shepp_logan_adjoint_dd_gpu():
    if yrt.compiledWithCuda():
        _test_uhr2d_shepp_logan_adjoint('DD_GPU')
    else:
        pytest.skip('Code not compiled with cuda. Skipping...')


def test_uhr2d_shepp_logan_osem_his_exec():
    fold_uhr2d = os.path.join(fold_data, 'uhr2d')

    out_recon_path = os.path.join(
        fold_out, 'test_uhr2d_shepp_logan_osem_his_exec_recon_image.nii')
    out_sens_path_prefix = os.path.join(
        fold_out, 'test_uhr2d_shepp_logan_osem_his_exec_sens_image')
    out_sens_path = out_sens_path_prefix + '.nii'

    recon_exec_str = os.path.join(fold_bin, 'yrtpet_reconstruct')
    recon_exec_str += ' --scanner ' + os.path.join(fold_uhr2d, 'UHR2D.json')
    recon_exec_str += ' --params ' + os.path.join(fold_uhr2d,
                                                  'img_params_2d.json')
    recon_exec_str += ' --out ' + out_recon_path
    recon_exec_str += ' --out_sens ' + out_sens_path
    recon_exec_str += ' --input ' + os.path.join(fold_uhr2d,
                                                 'shepp_logan.his')
    recon_exec_str += ' --format H'
    recon_exec_str += ' --num_subsets 5'
    recon_exec_str += ' --num_iterations 100'
    print('Running: ' + recon_exec_str)
    ret = os.system(recon_exec_str)
    assert ret == 0

    img_params = yrt.ImageParams(os.path.join(fold_uhr2d,
                                              'img_params_2d.json'))
    for i in range(5):
        ref_sens = yrt.ImageOwned(img_params, os.path.join(
            fold_uhr2d, 'ref',
            'sens_image_siddon_subset{idx}.nii'.format(idx=i)))
        out_sens = yrt.ImageOwned(img_params,
                                  out_sens_path_prefix +
                                  '_subset{idx}.nii'.format(idx=i))

        np.testing.assert_allclose(np.array(ref_sens, copy=False),
                                   np.array(out_sens, copy=False),
                                   atol=0, rtol=1e-5)

    ref_recon = yrt.ImageOwned(img_params,
                               os.path.join(fold_uhr2d,
                                            'ref',
                                            'shepp_logan_osem_his.nii'))
    out_recon = yrt.ImageOwned(img_params, out_recon_path)
    np.testing.assert_allclose(np.array(ref_recon, copy=False),
                               np.array(out_recon, copy=False),
                               atol=0, rtol=5e-3)


def test_large_flat_panel_xcat_osem_tof_siddon():
    fold_large_flat_panel = os.path.join(fold_data, 'large_flat_panel')
    scanner = yrt.Scanner(
        os.path.join(fold_large_flat_panel, 'large_flat_panel.json'))
    fold_xcat = os.path.join(fold_large_flat_panel, 'xcat')
    img_params = yrt.ImageParams(os.path.join(fold_xcat,
                                              'img_params_3mm.json'))
    dataset = yrt.ListModeLUTDOIOwned(
        scanner, os.path.join(fold_xcat, 'sim_4min_49ps.lmDoiDat'),
        flag_tof=True)
    sens_img = yrt.ImageOwned(
        img_params, os.path.join(fold_xcat, 'sens_image_siddon.nii'))

    osem = yrt.createOSEM(scanner)
    osem.setImageParams(img_params)
    osem.num_MLEM_iterations = 5
    osem.num_OSEM_subsets = 12
    osem.setDataInput(dataset)
    osem.addTOF(70, 5)
    osem.setSensitivityImage(sens_img)
    out_img = osem.reconstruct()

    out_img.writeToFile(os.path.join(
        fold_out, 'test_large_flat_panel_xcat_osem_tof_siddon.nii'))

    ref_img = yrt.ImageOwned(img_params, os.path.join(
        fold_large_flat_panel, 'ref', 'xcat_osem_siddon.nii'))

    np_out_img = np.array(out_img, copy=False)
    np_ref_img = np.array(ref_img, copy=False)

    nrmse = _helper.get_nrmse(np_out_img, np_ref_img)
    assert nrmse < 7e-4


def test_large_flat_panel_xcat_osem_tof_dd_gpu_exec():
    if not yrt.compiledWithCuda():
        pytest.skip('Code not compiled with cuda. Skipping...')
    fold_large_flat_panel = os.path.join(fold_data, 'large_flat_panel')
    fold_xcat = os.path.join(fold_large_flat_panel, 'xcat')

    out_path = os.path.join(
        fold_out, 'test_large_flat_panel_xcat_osem_tof_dd_gpu_exec.nii')

    exec_str = os.path.join(fold_bin, 'yrtpet_reconstruct')
    exec_str += ' --scanner ' + os.path.join(fold_large_flat_panel,
                                             'large_flat_panel.json')
    exec_str += ' --params ' + os.path.join(fold_xcat, 'img_params_3mm.json')
    exec_str += ' --input ' + os.path.join(fold_xcat,
                                           'sim_4min_49ps.lmDoiDat')
    exec_str += ' --format LM-DOI --projector DD --gpu'
    exec_str += ' --sens ' + os.path.join(fold_xcat, 'sens_image_dd.nii')
    exec_str += ' --flag_tof --tof_width_ps 70 --tof_n_std 5'
    exec_str += ' --num_iterations 5 --num_subsets 12'
    exec_str += ' --out ' + out_path
    ret = os.system(exec_str)
    assert ret == 0

    img_params = yrt.ImageParams(os.path.join(fold_xcat,
                                              'img_params_3mm.json'))
    ref_img = yrt.ImageOwned(img_params,
                             os.path.join(fold_large_flat_panel, 'ref',
                                          'xcat_osem_dd.nii'))
    out_img = yrt.ImageOwned(img_params, out_path)

    out_img_np = np.array(out_img, copy=False)
    ref_img_np = np.array(ref_img, copy=False)
    cur_nrmse = _helper.get_nrmse(out_img_np, ref_img_np)
    assert cur_nrmse < 2e-5


# %% Standalone command line

if __name__ == '__main__':
    print('Run \'pytest test_recon.py\' to launch integration tests')
