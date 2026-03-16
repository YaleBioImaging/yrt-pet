#!/bin/env python
"""Integration tests for YRT-PET"""

# %% Imports

import os
import sys
import pytest
import numpy as np
import copy
import pickle

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(fold_py))
import pyyrtpet as yrt

# %% Get paths

# TODO NOW: This script should not depend on this folder
HOME_DIR = os.path.expanduser('~yd385')

fold_data = os.path.join(HOME_DIR, 'yrt-pet-integration_tests', 'test_data')
fold_ge2d_dyn = os.path.join(fold_data, "phantom_dynamic")
fold_ge2d = os.path.join(fold_data, "phantom_static")
scanner = yrt.Scanner(os.path.join(fold_ge2d_dyn, "UHR2D.json"))

rng = np.random.default_rng(0)


def create_operator_projector(proj_params, bin_iter=None,
                              use_gpu: bool = False):
    if use_gpu:
        oper = yrt.OperatorProjectorDevice.create(proj_params, bin_iter)
    else:
        oper = yrt.OperatorProjector(proj_params, bin_iter)
    return oper


# %% Helper function to load data and prepare projector/list-mode

def prepare_data(updater_type):
    out_lm = yrt.ListModeLUTOwned(scanner, os.path.join(fold_ge2d_dyn,
                                                        'shepp_logan_dyn.lmDat'))
    img_params = yrt.ImageParams(
        os.path.join(fold_ge2d_dyn, "img_params_2d_dyn.json"))
    rank = 1 if updater_type == 'DEFAULT3D' else 5
    T = 1 if updater_type == 'DEFAULT3D' else 20
    img_params.nt = rank if 'LR' in updater_type else T

    if updater_type in ['DEFAULT4D', 'LR']:
        HBasis_np = pickle.load(open(os.path.join(fold_ge2d_dyn,
                                                  f'HBasis_r{rank}.pik'),
                                     'rb'))['HBasis'].astype(np.float32)
        HBasis_np = np.require(HBasis_np, requirements=['C'])

        max_timestamp = np.max(out_lm.getTimestampArray())
        dynamic_framing_array = np.require(np.linspace(0, max_timestamp + 1,
                                                       num=T + 1,
                                                       dtype=np.uint32),
                                           requirements=['C'])

        df = yrt.DynamicFraming(dynamic_framing_array)
        out_lm.addDynamicFraming(df)
    else:
        df = None
        HBasis_np = None

    bin_iter = out_lm.getBinIter(1, 0)

    count = out_lm.count()
    assert count == bin_iter.size()

    proj_params = yrt.ProjectorParams(scanner)
    proj_params.updaterType = getattr(yrt.UpdaterType, updater_type)

    if updater_type == 'LR':
        proj_params.setHBasisFromNumpy(HBasis_np)

    return {'lm': out_lm, 'proj_params': proj_params, 'df': df,
            'bin_iter': bin_iter,
            'rank': rank, 'T': T, 'img_params': img_params, 'HBasis': HBasis_np}


def _ref_fwd_bwd_cpu(projector_type):
    # Prepare reference CPU forward and backward
    prep_dict_ref = prepare_data('DEFAULT4D')
    out_lm_ref = prep_dict_ref['lm']
    proj_params_ref = prep_dict_ref['proj_params']
    proj_params_ref.setProjector(projector_type)
    img_params_ref = prep_dict_ref['img_params']
    HBasis_np_ref = prep_dict_ref['HBasis']
    rank_ref = prep_dict_ref['rank']
    T_ref = prep_dict_ref['T']
    bin_iter = prep_dict_ref['bin_iter']

    oper_ref_cpu = create_operator_projector(proj_params_ref, bin_iter, False)
    count = out_lm_ref.count()

    # Prepare variables for projections
    y_proj_ref_np = np.require(np.zeros(count), requirements=['C'],
                               dtype=np.float32)
    W_np = np.abs(np.require(rng.standard_normal((rank_ref, img_params_ref.nz,
                                                  img_params_ref.ny,
                                                  img_params_ref.nx),
                                                 dtype=np.float32),
                             requirements=['C']))

    y_proj_ref = yrt.ProjectionListAlias(out_lm_ref)
    y_proj_ref.bind(y_proj_ref_np)
    x_ref_np = np.tensordot(HBasis_np_ref.T, W_np, (1, 0))
    x_ref_img = yrt.ImageAlias(img_params_ref)
    x_ref_img.bind(x_ref_np)

    x_ref_np_backward = np.require(np.zeros_like(x_ref_np).astype(np.float32),
                                   requirements=['C'])
    x_ref_img_backward = yrt.ImageAlias(img_params_ref)
    x_ref_img_backward.bind(x_ref_np_backward)

    # Compute and store A(W * H) in y_proj_ref
    oper_ref_cpu.applyA(x_ref_img, y_proj_ref)
    # Compute and store A^T(y) * H^T in x_ref_img_backward
    oper_ref_cpu.applyAH(y_proj_ref, x_ref_img_backward)

    W_ref_np_backward = np.tensordot(HBasis_np_ref, x_ref_np_backward, (1, 0))
    H_ref_np_backward = np.tensordot(x_ref_np_backward, W_np.T,
                                     ((-1, -2, -3), (0, 1, 2))).T

    return {'x_ref_np_backward': x_ref_np_backward, 'x_ref_np': x_ref_np,
            'W_ref_np_backward': W_ref_np_backward, 'W_np': W_np,
            'H_ref_np_backward': H_ref_np_backward,
            'y_proj_ref_np': y_proj_ref_np}


# %% Generic test function used later for each Updater type and projector

fwd_bwd = {'Siddon': _ref_fwd_bwd_cpu('Siddon'),
           'DD': _ref_fwd_bwd_cpu('DD')}


def _test_adjoint_generic(UpdaterType, projector_type, use_gpu):
    if use_gpu and not yrt.compiledWithCuda():
        pytest.skip('Code not compiled with cuda. Skipping...')
    prep_dict = prepare_data(UpdaterType)
    out_lm = prep_dict['lm']
    proj_params = prep_dict['proj_params']
    proj_params.setProjector(projector_type)
    img_params = prep_dict['img_params']
    HBasis_np = prep_dict['HBasis']
    rank = prep_dict['rank']
    bin_iter = prep_dict['bin_iter']

    oper = create_operator_projector(proj_params, bin_iter, use_gpu)
    count = out_lm.count()

    # --- sizes ---
    nt = img_params.nt
    nx_2d = img_params.nx
    ny_2d = img_params.ny
    nz_2d = img_params.nz

    # --- random x (image) and y (projection) ---
    x_np = np.abs(np.require(
        rng.standard_normal((nt, nz_2d, ny_2d, nx_2d), dtype=np.float32),
        requirements=['C']))
    y_np = np.abs(np.require(rng.standard_normal(count, dtype=np.float32),
                             requirements=['C']))

    # --- output holders (will be written by the ops) ---
    Ax_np = np.require(np.zeros_like(y_np), requirements=['C'])  # will hold A x
    AHy_np = np.require(np.zeros_like(x_np),
                        requirements=['C'])  # will hold A^H y

    # --- aliases (views into numpy memory) ---
    # Image aliases take ImageParams
    x_img = yrt.ImageAlias(img_params)
    x_img.bind(x_np)
    AHy_img = yrt.ImageAlias(img_params)
    AHy_img.bind(AHy_np)

    # Projection aliases take a ProjectionData to match layout
    y_proj = yrt.ProjectionListAlias(out_lm)
    y_proj.bind(y_np)
    Ax_proj = yrt.ProjectionListAlias(out_lm)
    Ax_proj.bind(Ax_np)

    # --- run A and A^H ---
    oper.applyA(x_img, Ax_proj)  # Ax_np is filled in-place
    oper.applyAH(y_proj, AHy_img)  # AHy_np is filled in-place

    # --- inner products ---
    lhs = float(np.dot(Ax_np, y_np))  # <A x, y>
    rhs = float(np.vdot(x_np.ravel(), AHy_np.ravel()))  # <x, A^H y>
    rel_err = abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))
    assert rel_err < 1e-3

    print(
        f'\n\n ................Adjoint test for {UpdaterType}................\n')
    print(f"<A x, y>   = {lhs:.6e}")
    print(f"<x, A^H y> = {rhs:.6e}")
    print(f"relative error = {rel_err:.3e}")


def _test_generic_fwd(UpdaterType, projector_type, use_gpu: bool):
    if use_gpu and not yrt.compiledWithCuda():
        pytest.skip('Code not compiled with cuda. Skipping...')
    prep_dict = prepare_data(UpdaterType)
    out_lm = prep_dict['lm']
    proj_params = prep_dict['proj_params']
    proj_params.setProjector(projector_type)
    img_params = prep_dict['img_params']
    HBasis_np = prep_dict['HBasis']
    rank = prep_dict['rank']
    T = prep_dict['T']
    ref_load = fwd_bwd[projector_type]
    bin_iter = prep_dict['bin_iter']

    oper = create_operator_projector(proj_params, bin_iter, use_gpu)
    count = out_lm.count()

    # Store A(W) * H computed from ProjectorParams.DEFAULT4D or LR
    y_proj_lr = yrt.ProjectionListAlias(out_lm)
    y_proj_lr_np = np.require(np.zeros(count), requirements=['C'],
                              dtype=np.float32)
    y_proj_lr.bind(y_proj_lr_np)

    x_lr_img = yrt.ImageAlias(img_params)
    x_np_test = ref_load['x_ref_np'] if 'DEFAULT4D' in UpdaterType else \
        ref_load['W_np']
    x_lr_img.bind(x_np_test)

    oper.applyA(x_lr_img, y_proj_lr)

    np.testing.assert_allclose(ref_load['y_proj_ref_np'], y_proj_lr_np,
                               atol=1e-5, rtol=1e-5)
    print(
        f'\n\n ................Forward test passed for {UpdaterType}!................\n')


def _test_generic_bwd(UpdaterType, projector_type, use_gpu: bool,
                      updateH=False):
    if use_gpu and not yrt.compiledWithCuda():
        pytest.skip('Code not compiled with cuda. Skipping...')
    prep_dict = prepare_data(UpdaterType)
    out_lm = prep_dict['lm']
    proj_params = prep_dict['proj_params']
    proj_params.setProjector(projector_type)
    img_params = prep_dict['img_params']
    HBasis_np = prep_dict['HBasis']
    rank = prep_dict['rank']
    T = prep_dict['T']
    ref_load = fwd_bwd[projector_type]
    bin_iter = prep_dict['bin_iter']

    oper = create_operator_projector(proj_params, bin_iter, use_gpu)

    x_np_test = ref_load['x_ref_np'] if 'DEFAULT4D' in UpdaterType else \
        ref_load['W_np']
    y_proj_ref = yrt.ProjectionListAlias(out_lm)
    y_proj_ref.bind(ref_load['y_proj_ref_np'])

    if not updateH:
        # Store A^T(y * H^T) computed from ProjectorParams.DEFAULT4D or LR
        W_lr_np_backward = np.require(
            np.zeros_like(x_np_test).astype(np.float32), requirements=['C'])
        W_lr_img_backward = yrt.ImageAlias(img_params)
        W_lr_img_backward.bind(W_lr_np_backward)
        oper.applyAH(y_proj_ref, W_lr_img_backward)

        backward_ref = (ref_load['x_ref_np_backward']
                        if 'DEFAULT4D' in UpdaterType else ref_load[
            'W_ref_np_backward'])

        np.testing.assert_allclose(backward_ref, W_lr_np_backward, atol=1e-5,
                                   rtol=2e-3)
        print(
            f'\n\n ................Backward test passed for {UpdaterType}!................\n')

    else:
        if 'LR' not in UpdaterType:
            raise ValueError(
                f"Cannot apply H-update backward on an opertor that is not LR (given {UpdaterType})")
        x_lr_img = yrt.ImageAlias(img_params)
        x_lr_img.bind(x_np_test)

        proj_params.updateH = updateH
        HBasisWrite_np = np.require(np.zeros_like(HBasis_np).astype(np.float32),
                                    requirements=['C'])
        oper_H = create_operator_projector(proj_params, bin_iter, use_gpu)
        oper_H.setUpdaterLRHBasisWrite(HBasisWrite_np)
        HBasis_copy = copy.deepcopy(oper_H.getUpdaterLRHBasis())
        x_np_test_copy = copy.deepcopy(x_np_test)
        assert oper_H.getUpdaterLRUpdateH() == proj_params.updateH

        oper_H.applyAH(y_proj_ref, x_lr_img)

        np.testing.assert_allclose(HBasis_copy, oper_H.getUpdaterLRHBasis())
        np.testing.assert_allclose(x_np_test_copy, x_np_test)
        H_lr_np_backward = oper_H.getUpdaterLRHBasisWrite()
        np.testing.assert_allclose(ref_load['H_ref_np_backward'],
                                   H_lr_np_backward, atol=1e-5, rtol=3e-3)
        print(
            f'\n\n ................Backward test passed for {UpdaterType} (H update)!................\n')


# %% --------------------------------- ADJOINT TESTS ---------------------------------
# %% CPU tests

# Siddon
def test_adjoint_default3d_siddon_cpu():
    _test_adjoint_generic(UpdaterType='DEFAULT3D', projector_type='Siddon',
                          use_gpu=False)


def test_adjoint_default4d_siddon_cpu():
    _test_adjoint_generic('DEFAULT4D', projector_type='Siddon', use_gpu=False)


def test_adjoint_lr_siddon_cpu():
    _test_adjoint_generic('LR', projector_type='Siddon', use_gpu=False)


# DD
def test_adjoint_default3d_dd_cpu():
    _test_adjoint_generic(UpdaterType='DEFAULT3D', projector_type='DD',
                          use_gpu=False)


def test_adjoint_default4d_dd_cpu():
    _test_adjoint_generic('DEFAULT4D', projector_type='DD', use_gpu=False)


def test_adjoint_lr_dd_cpu():
    _test_adjoint_generic('LR', projector_type='DD', use_gpu=False)


# %% GPU tests

# Siddon
def test_adjoint_default3d_siddon_gpu():
    _test_adjoint_generic(UpdaterType='DEFAULT3D', projector_type='Siddon',
                          use_gpu=True)


def test_adjoint_default4d_siddon_gpu():
    _test_adjoint_generic('DEFAULT4D', projector_type='Siddon', use_gpu=True)


def test_adjoint_lr_siddon_gpu():
    _test_adjoint_generic('LR', projector_type='Siddon', use_gpu=True)


# DD
def test_adjoint_default3d_dd_gpu():
    _test_adjoint_generic(UpdaterType='DEFAULT3D', projector_type='DD',
                          use_gpu=True)


def test_adjoint_default4d_dd_gpu():
    _test_adjoint_generic('DEFAULT4D', projector_type='DD', use_gpu=True)


def test_adjoint_lr_dd_gpu():
    _test_adjoint_generic('LR', projector_type='DD', use_gpu=True)


# %% --------------------------------- TEST FORWARD ---------------------------------
# %% CPU tests

# Siddon
def test_fwd_default4d_siddon_cpu():
    _test_generic_fwd('DEFAULT4D', projector_type='Siddon', use_gpu=False)


def test_fwd_lr_siddon_cpu():
    _test_generic_fwd('LR', projector_type='Siddon', use_gpu=False)


# DD
def test_fwd_default4d_dd_cpu():
    _test_generic_fwd('DEFAULT4D', projector_type='DD', use_gpu=False)


def test_fwd_lr_dd_cpu():
    _test_generic_fwd('LR', projector_type='DD', use_gpu=False)


# %% GPU tests

# Siddon
def test_fwd_default4d_siddon_gpu():
    _test_generic_fwd('DEFAULT4D', projector_type='Siddon', use_gpu=True)


def test_fwd_lr_siddon_gpu():
    _test_generic_fwd('LR', projector_type='Siddon', use_gpu=True)


# DD
def test_fwd_default4d_dd_gpu():
    _test_generic_fwd('DEFAULT4D', projector_type='DD', use_gpu=True)


def test_fwd_lr_dd_gpu():
    _test_generic_fwd('LR', projector_type='DD', use_gpu=True)


# %% --------------------------------- TEST BACKWARD ---------------------------------
# %% CPU tests

# Siddon
def test_bwd_default4d_siddon_cpu():
    _test_generic_bwd('DEFAULT4D', projector_type='Siddon', use_gpu=False)


def test_bwd_lr_siddon_cpu():
    _test_generic_bwd('LR', projector_type='Siddon', use_gpu=False)


def test_bwd_lr_hupdate_siddon_cpu():
    _test_generic_bwd('LR', projector_type='Siddon', use_gpu=False,
                      updateH=True)


# DD
def test_bwd_default4d_dd_cpu():
    _test_generic_bwd('DEFAULT4D', projector_type='DD', use_gpu=False)


def test_bwd_lr_dd_cpu():
    _test_generic_bwd('LR', projector_type='DD', use_gpu=False)


def test_bwd_lr_hupdate_dd_cpu():
    _test_generic_bwd('LR', projector_type='DD', use_gpu=False, updateH=True)


# %% GPU tests

# Siddon
def test_bwd_default4d_siddon_gpu():
    _test_generic_bwd('DEFAULT4D', projector_type='Siddon', use_gpu=True)


def test_bwd_lr_siddon_gpu():
    _test_generic_bwd('LR', projector_type='Siddon', use_gpu=True)


def test_bwd_lr_hupdate_siddon_gpu():
    _test_generic_bwd('LR', projector_type='Siddon', use_gpu=True, updateH=True)


# DD
def test_bwd_default4d_dd_gpu():
    _test_generic_bwd('DEFAULT4D', projector_type='DD', use_gpu=True)


def test_bwd_lr_dd_gpu():
    _test_generic_bwd('LR', projector_type='DD', use_gpu=True)


def test_bwd_lr_hupdate_dd_gpu():
    _test_generic_bwd('LR', projector_type='DD', use_gpu=True, updateH=True)
