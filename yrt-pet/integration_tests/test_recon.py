#!/bin/env python
"""Integration tests for YRT-PET"""

# %% Imports

import os
import sys

import numpy as np

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(fold_py))
import pyyrtpet as yrt
yrt.Globals.set_num_threads(-1)

import helper as _helper

dataset_paths = _helper.dataset_paths
out_paths = _helper.out_paths
ref_paths = _helper.ref_paths
util_paths = _helper.util_paths
fold_data = _helper.fold_data
fold_out = _helper.fold_out
fold_bin = _helper.fold_bin

# %% Tests

def test_omp_multithreading():
    yrt.Globals.set_num_threads(8)
    num_threads = yrt.Globals.get_num_threads()
    assert(num_threads == 8)
    yrt.Globals.set_num_threads(-1)

def test_mlem_simple():
    img_params = yrt.ImageParams(util_paths['img_params_500'])
    scanner = yrt.Scanner(util_paths['SAVANT_json'])
    dataset = yrt.ListModeLUTOwned(scanner, dataset_paths['test_mlem_simple'])
    sens_img = yrt.ImageOwned(img_params, util_paths['SensImageSAVANT500'])
    _helper._test_reconstruction(
        img_params, scanner, dataset, sens_img,
        out_paths['test_mlem_simple'], ref_paths['test_mlem_simple'],
        num_MLEM_iterations=30, rtol=1e-3)


def _test_mlem_helper(dset):
    """Helper function for motion MLEM tests"""
    test_name = 'test_mlem_{}'.format(dset)
    img_params = yrt.ImageParams(util_paths['img_params_500'])
    scanner = yrt.Scanner(util_paths['SAVANT_json'])
    dataset = yrt.ListModeLUTOwned(scanner, dataset_paths[test_name][0])
    sens_img = yrt.ImageOwned(img_params, util_paths['SensImageSAVANT500'])

    warper = yrt.ImageWarperMatrix()
    warper.setImageHyperParam(img_params)
    warper.setFramesParamFromFile(dataset_paths[test_name][1])
    _helper._test_reconstruction(
        img_params, scanner, dataset, sens_img,
        out_paths[test_name], ref_paths[test_name],
        warper=warper, num_MLEM_iterations=10, hard_threshold=100.0)


def test_mlem_piston():
    _test_mlem_helper('piston')


def test_mlem_wobble():
    _test_mlem_helper('wobble')


def test_mlem_yesMan():
    _test_mlem_helper('yesMan')


def test_bwd():
    img_params = yrt.ImageParams(util_paths['img_params_500'])
    scanner = yrt.Scanner(util_paths['SAVANT_json'])
    dataset = yrt.ListModeLUTOwned(scanner, dataset_paths['test_bwd'])

    out_img = yrt.ImageOwned(img_params)
    out_img.allocate()
    out_img.setValue(0.0)

    yrt.backProject(scanner, out_img, dataset)
    out_img.applyThreshold(out_img, 1, 0, 0, 1, 0)

    out_img.writeToFile(out_paths['test_bwd'])

    ref_img = yrt.ImageOwned(img_params, ref_paths['test_bwd'])
    np.testing.assert_allclose(np.array(out_img, copy=False),
                               np.array(ref_img, copy=False),
                               atol=0, rtol=1e-5)

def test_sens():
    img_params = yrt.ImageParams(util_paths['img_params_500'])
    scanner = yrt.Scanner(util_paths['SAVANT_json'])

    osem = yrt.createOSEM(scanner)
    osem.setImageParams(img_params)

    out_imgs = osem.generateSensitivityImages()

    out_img = out_imgs[0]
    out_img.writeToFile(out_paths['test_sens'])

    ref_img = yrt.ImageOwned(img_params, ref_paths['test_sens'])
    np.testing.assert_allclose(np.array(out_img, copy=False),
                               np.array(ref_img, copy=False),
                               atol=0, rtol=1e-4)

def test_sens_exec():
    exec_str = os.path.join(fold_bin, 'yrtpet_reconstruct --sens_only')
    exec_str += ' --scanner ' + util_paths['SAVANT_json']
    exec_str += ' --params ' + util_paths['img_params_500']
    exec_str += ' --out_sens ' + out_paths['test_sens_exec']
    ret = os.system(exec_str)
    assert ret == 0

    img_params = yrt.ImageParams(util_paths['img_params_500'])
    out_img = yrt.ImageOwned(img_params, out_paths['test_sens_exec'])
    ref_img = yrt.ImageOwned(img_params, ref_paths['test_sens'])
    np.testing.assert_allclose(np.array(out_img, copy=False),
                               np.array(ref_img, copy=False),
                               atol=0, rtol=1e-4)

def _test_savant_motion_post_mc(test_name: str):
    img_params = yrt.ImageParams(util_paths['img_params_500'])
    file_list = dataset_paths[test_name]
    image_list = []
    for fname in file_list[:-1]:
        image_list.append(yrt.ImageOwned(img_params, fname))

    warper = yrt.ImageWarperFunction()
    warper.setImageHyperParam([img_params.nx, img_params.ny, img_params.nz],
                              [img_params.length_x, img_params.length_y,
                               img_params.length_z])
    warper.setFramesParamFromFile(dataset_paths[test_name][len(file_list) - 1])

    out_img = yrt.ImageOwned(img_params)
    out_img.allocate()
    out_img.setValue(0.0)
    for i, image in enumerate(image_list):
        warper.warpImageToRefFrame(image, i)
        image.addFirstImageToSecond(out_img)

    out_img.writeToFile(out_paths[test_name])

    ref_img = yrt.ImageOwned(img_params, ref_paths[test_name])
    nrmse = _helper.get_nrmse(np.array(out_img, copy=False),
                              np.array(ref_img, copy=False))
    assert nrmse < 10**-6


def test_post_recon_mc_piston():
    _test_savant_motion_post_mc('test_post_recon_mc_piston')


def test_post_recon_mc_wobble():
    _test_savant_motion_post_mc('test_post_recon_mc_wobble')


def test_psf():
    img_params = yrt.ImageParams(50, 50, 25, 50, 50, 25, 0.0, 0.0, 0.0)
    image_in = yrt.ImageOwned(img_params, dataset_paths['test_psf'][0])
    oper_psf = yrt.OperatorPsf(dataset_paths['test_psf'][1])
    image_out = yrt.ImageOwned(img_params)
    image_out.allocate()
    image_out.setValue(0.0)

    oper_psf.applyA(image_in, image_out)

    image_out.writeToFile(out_paths['test_psf'])
    image_ref = yrt.ImageOwned(img_params, ref_paths['test_psf'])
    np.testing.assert_allclose(np.array(image_out, copy=False),
                               np.array(image_ref, copy=False),
                               atol=0, rtol=1e-5)

def test_psf_gpu():
    img_params = yrt.ImageParams(50, 50, 25, 50, 50, 25, 0.0, 0.0, 0.0)
    image_in = yrt.ImageOwned(img_params, dataset_paths['test_psf'][0])
    oper_psf = yrt.OperatorPsfDevice(dataset_paths['test_psf'][1])
    image_out = yrt.ImageOwned(img_params)
    image_out.allocate()
    image_out.setValue(0.0)

    oper_psf.applyA(image_in, image_out)

    image_out.writeToFile(out_paths['test_psf_gpu'])
    image_ref = yrt.ImageOwned(img_params, ref_paths['test_psf'])
    np.testing.assert_allclose(np.array(image_out, copy=False),
                               np.array(image_ref, copy=False),
                               atol=0, rtol=1e-5)

def test_psf_adjoint():
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

    img_X = yrt.ImageAlias(img_params)
    img_Y = yrt.ImageAlias(img_params)

    img_X_a = (rng.random([nz, ny, nx]) * 10 - 5).astype(np.float32)
    img_Y_a = (rng.random([nz, ny, nx]) * 10 - 5).astype(np.float32)
    img_X.bind(img_X_a)
    img_Y.bind(img_Y_a)

    oper_psf = yrt.OperatorPsf( dataset_paths['test_psf'][1])

    Ax = yrt.ImageOwned(img_params)
    Aty = yrt.ImageOwned(img_params)
    Ax.allocate()
    Ax.setValue(0.0)
    Aty.allocate()
    Aty.setValue(0.0)

    oper_psf.applyA(img_X, Ax)
    oper_psf.applyAH(img_Y, Aty)

    dot_Ax_y = Ax.dotProduct(img_Y)
    dot_x_Aty = img_X.dotProduct(Aty)

    assert abs(dot_Ax_y - dot_x_Aty) < 10**-4


def test_flat_panel_mlem_tof():
    img_params = yrt.ImageParams(util_paths['img_params_3.0'])
    scanner = yrt.Scanner(
        util_paths['Geometry_2panels_large_3x3x20mm_rot_gc_json'])
    dataset = yrt.ListModeLUTDOIOwned(
        scanner, dataset_paths['test_flat_panel_mlem_tof'][0], True)
    sens_img = yrt.ImageOwned(img_params,
                              dataset_paths['test_flat_panel_mlem_tof'][1])

    _helper._test_reconstruction(
        img_params, scanner, dataset, sens_img,
        out_paths['test_flat_panel_mlem_tof'],
        ref_paths['test_flat_panel_mlem_tof'][0],
        num_MLEM_iterations=5, num_OSEM_subsets=12, num_threads=20,
        tof_width_ps=70, tof_n_std=5, rtol=None, nrmse=1e-6)


def test_flat_panel_mlem_tof_exec():
    exec_str = os.path.join(fold_bin, 'yrtpet_reconstruct')
    exec_str += ' --scanner ' + \
        util_paths['Geometry_2panels_large_3x3x20mm_rot_gc_json']
    exec_str += ' --params ' + util_paths['img_params_3.0']
    exec_str += ' --input ' + dataset_paths['test_flat_panel_mlem_tof'][0]
    exec_str += ' --format LM-DOI --projector DD_GPU'
    exec_str += ' --sens ' + dataset_paths['test_flat_panel_mlem_tof'][2]
    exec_str += ' --flag_tof --tof_width_ps 70 --tof_n_std 5'
    exec_str += ' --num_iterations 10 --num_threads 20'
    exec_str += ' --out ' + out_paths['test_flat_panel_mlem_tof_exec']
    ret = os.system(exec_str)
    assert ret == 0

    img_params = yrt.ImageParams(util_paths['img_params_3.0'])
    ref_img = yrt.ImageOwned(img_params,
                             ref_paths['test_flat_panel_mlem_tof'][1])
    out_img = yrt.ImageOwned(img_params,
                             out_paths['test_flat_panel_mlem_tof_exec'])

    np_out_img = np.array(out_img, copy=False)
    np_ref_img = np.array(ref_img, copy=False)
    cur_nrmse = _helper.get_nrmse(np_out_img, np_ref_img)
    assert cur_nrmse < 1e-6


def test_subsets_savant_siddon():
    scanner = yrt.Scanner(util_paths["SAVANT_json"])
    img_params = yrt.ImageParams(util_paths["img_params_500"])
    lm = yrt.ListModeLUTOwned(scanner, dataset_paths["test_subsets_savant"])
    _helper._test_subsets(scanner, img_params, lm, projector='Siddon')


def test_subsets_savant_dd():
    scanner = yrt.Scanner(util_paths["SAVANT_json"])
    img_params = yrt.ImageParams(util_paths["img_params_500"])
    lm = yrt.ListModeLUTOwned(scanner, dataset_paths["test_subsets_savant"])
    _helper._test_subsets(scanner, img_params, lm, projector='DD')


def test_adjoint_uhr2d_siddon():
    scanner = yrt.Scanner(util_paths["UHR2D_json"])
    img_params = yrt.ImageParams(util_paths["img_params_2d"])
    his = yrt.ListModeLUTOwned(scanner, dataset_paths["test_adjoint_uhr2d"])
    _helper._test_adjoint(scanner, img_params, his, projector='Siddon')


def test_adjoint_uhr2d_multi_ray_siddon():
    scanner = yrt.Scanner(util_paths["UHR2D_json"])
    img_params = yrt.ImageParams(util_paths["img_params_2d"])
    his = yrt.ListModeLUTOwned(scanner, dataset_paths["test_adjoint_uhr2d"])
    _helper._test_adjoint(scanner, img_params, his, projector='Siddon',
                          num_rays=4)


def test_adjoint_uhr2d_dd():
    scanner = yrt.Scanner(util_paths["UHR2D_json"])
    img_params = yrt.ImageParams(util_paths["img_params_2d"])
    his = yrt.ListModeLUTOwned(scanner, dataset_paths["test_adjoint_uhr2d"])
    _helper._test_adjoint(scanner, img_params, his, projector='DD')


def test_osem_his_2d():
    recon_exec_str = os.path.join(fold_bin, 'yrtpet_reconstruct')
    recon_exec_str += " --scanner " + util_paths['UHR2D_json']
    recon_exec_str += " --params " + util_paths["img_params_2d"]
    recon_exec_str += " --out " + out_paths['test_osem_his_2d'][0]
    recon_exec_str += " --out_sens " + out_paths['test_osem_his_2d'][1]
    recon_exec_str += " --input " + dataset_paths['test_osem_his_2d']
    recon_exec_str += " --format H"
    recon_exec_str += " --num_subsets 5"
    recon_exec_str += " --num_iterations 100"
    print("Running: " + recon_exec_str)
    ret = os.system(recon_exec_str)
    assert ret == 0

    img_params = yrt.ImageParams(util_paths['img_params_2d'])
    for i in range(5):
        ref_gensensimg = yrt.ImageOwned(img_params,
                                        ref_paths['test_osem_his_2d'][1][i])
        out_gensensimg = yrt.ImageOwned(img_params,
                                        out_paths['test_osem_his_2d'][2][i])
        np.testing.assert_allclose(np.array(ref_gensensimg, copy=False),
                                   np.array(out_gensensimg, copy=False),
                                   atol=0, rtol=1e-5)

    ref_gensensimg = yrt.ImageOwned(img_params,
                                    ref_paths['test_osem_his_2d'][0])
    out_gensensimg = yrt.ImageOwned(img_params,
                                    out_paths['test_osem_his_2d'][0])
    np.testing.assert_allclose(np.array(ref_gensensimg, copy=False),
                               np.array(out_gensensimg, copy=False),
                               atol=0, rtol=5e-3)


def test_osem_siddon_multi_ray():
    num_siddon_rays = 6
    img_params = yrt.ImageParams(util_paths['img_params_500'])
    scanner = yrt.Scanner(util_paths['SAVANT_json'])
    dataset = yrt.ListModeLUTOwned(
        scanner, dataset_paths['test_osem_siddon_multi_ray'])
    sens_img = yrt.ImageOwned(
        img_params, util_paths['sens_SAVANT_multi_ray_500'])

    _helper._test_reconstruction(
        img_params, scanner, dataset, sens_img,
        out_paths['test_osem_siddon_multi_ray'],
        ref_paths['test_osem_siddon_multi_ray'],
        num_MLEM_iterations=3, num_OSEM_subsets=12,
        num_rays=num_siddon_rays, rtol=1e-3)


# %% Standalone command line

if __name__ == '__main__':
    print('Run \'pytest test_recon.py\' to launch integration tests')
