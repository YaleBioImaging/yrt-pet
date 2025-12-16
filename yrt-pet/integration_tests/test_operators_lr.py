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

HOME_DIR = os.path.expanduser('~yd385')

fold_data = os.path.join(HOME_DIR, 'yrt-pet-integration_tests', 'test_data')
fold_ge2d_dyn = os.path.join(fold_data, "phantom_dynamic")
fold_ge2d = os.path.join(fold_data, "phantom_static")
scanner = yrt.Scanner(os.path.join(fold_ge2d_dyn, "UHR2D.json"))

rng = np.random.default_rng(0)

oper_func_gpu = lambda projector_type: getattr(yrt, f'OperatorProjector{projector_type}_GPU')
oper_func_cpu = lambda projector_type: getattr(yrt, f'OperatorProjector{projector_type}')


# %% Helper function to load data and prepare projector/list-mode

def prepare_data(UpdaterType):
    out_lm = yrt.ListModeLUTOwned(scanner, os.path.join(fold_ge2d_dyn, 'shepp_logan_dyn.lmDat'))
    img_params = yrt.ImageParams(os.path.join(fold_ge2d_dyn, "img_params_2d_dyn.json"))
    rank = 1 if UpdaterType == 'DEFAULT3D' else 5
    T = 1 if UpdaterType == 'DEFAULT3D' else 20
    img_params.nt = rank if 'LR' in UpdaterType else T

    if UpdaterType in ['DEFAULT4D', 'LR']:
        HBasis_np = pickle.load(open(os.path.join(fold_ge2d_dyn,
                                                  f'HBasis_r{rank}.pik'), 'rb'))['HBasis'].astype(np.float32)
        HBasis_np = np.require(HBasis_np, requirements=['C'])

        max_timestamp = np.max(out_lm.getTimestampArray())
        dynamic_framing_array = np.require(np.linspace(0, max_timestamp + 1,
                                                       num=T + 1,
                                                       dtype=np.uint32), requirements=['C'])

        df = yrt.DynamicFraming(dynamic_framing_array)
        out_lm.addDynamicFraming(df)
    else:
        df = None
        HBasis_np = None

    bin_iter = out_lm.getBinIter(1, 0)

    count = out_lm.count()
    assert count == bin_iter.size()

    proj_params = yrt.OperatorProjectorParams(scanner)
    proj_params.binIter = bin_iter
    proj_params.projectorUpdaterType = getattr(yrt.OperatorProjectorParams, UpdaterType)

    if UpdaterType == 'LR':
        proj_params.setHBasisFromNumpy(HBasis_np)

    return {'lm': out_lm, 'proj_params': proj_params, 'df': df, 'bin_iter': bin_iter,
            'rank': rank, 'T': T, 'img_params': img_params, 'HBasis': HBasis_np}


def _ref_fwd_bwd_CPU(projector_type):
    # Prepare reference CPU forward and backward
    prep_dict_ref = prepare_data('DEFAULT4D')
    out_lm_ref = prep_dict_ref['lm']
    proj_params_ref = prep_dict_ref['proj_params']
    img_params_ref = prep_dict_ref['img_params']
    HBasis_np_ref = prep_dict_ref['HBasis']
    rank_ref = prep_dict_ref['rank']
    T_ref = prep_dict_ref['T']

    oper_ref_cpu = oper_func_cpu(projector_type)(proj_params_ref)
    count = out_lm_ref.count()

    # Prepare variables for projections
    y_proj_ref_np = np.require(np.zeros(count), requirements=['C'], dtype=np.float32)
    W_np = np.abs(np.require(rng.standard_normal((rank_ref, img_params_ref.nz,
                                                  img_params_ref.ny, img_params_ref.nx),
                                                 dtype=np.float32), requirements=['C']))

    y_proj_ref = yrt.ProjectionListAlias(out_lm_ref)
    y_proj_ref.bind(y_proj_ref_np)
    x_ref_np = np.tensordot(HBasis_np_ref.T, W_np, (1, 0))
    x_ref_img = yrt.ImageAlias(img_params_ref)
    x_ref_img.bind(x_ref_np)

    x_ref_np_backward = np.require(np.zeros_like(x_ref_np).astype(np.float32), requirements=['C'])
    x_ref_img_backward = yrt.ImageAlias(img_params_ref)
    x_ref_img_backward.bind(x_ref_np_backward)

    # Compute and store A(W * H) in y_proj_ref
    oper_ref_cpu.applyA(x_ref_img, y_proj_ref)
    # Compute and store A^T(y) * H^T in x_ref_img_backward
    oper_ref_cpu.applyAH(y_proj_ref, x_ref_img_backward)

    W_ref_np_backward = np.tensordot(HBasis_np_ref, x_ref_np_backward, (1, 0))
    H_ref_np_backward = np.tensordot(x_ref_np_backward, W_np.T, ((-1, -2, -3), (0, 1, 2))).T

    return {'x_ref_np_backward': x_ref_np_backward, 'x_ref_np': x_ref_np,
            'W_ref_np_backward': W_ref_np_backward, 'W_np': W_np,
            'H_ref_np_backward': H_ref_np_backward, 'y_proj_ref_np': y_proj_ref_np}


# %% Generic test function used later for each Updater type and projector

fwd_bwd = {'Siddon': _ref_fwd_bwd_CPU('Siddon'),
           'DD': _ref_fwd_bwd_CPU('DD')}


def _test_adjoint_generic(UpdaterType, projector_type, useGPU):
    if useGPU and not yrt.compiledWithCuda():
        pytest.skip('Code not compiled with cuda. Skipping...')
    prep_dict = prepare_data(UpdaterType)
    out_lm = prep_dict['lm']
    proj_params = prep_dict['proj_params']
    img_params = prep_dict['img_params']
    HBasis_np = prep_dict['HBasis']
    rank = prep_dict['rank']

    oper_test = oper_func_gpu if useGPU else oper_func_cpu
    oper = oper_test(projector_type)(proj_params)
    count = out_lm.count()

    # --- sizes ---
    nt = img_params.nt
    nx_2d = img_params.nx
    ny_2d = img_params.ny
    nz_2d = img_params.nz

    # --- random x (image) and y (projection) ---
    x_np = np.abs(np.require(rng.standard_normal((nt, nz_2d, ny_2d, nx_2d), dtype=np.float32),
                             requirements=['C']))
    y_np = np.abs(np.require(rng.standard_normal(count, dtype=np.float32),
                             requirements=['C']))

    # --- output holders (will be written by the ops) ---
    Ax_np = np.require(np.zeros_like(y_np), requirements=['C'])  # will hold A x
    AHy_np = np.require(np.zeros_like(x_np), requirements=['C'])  # will hold A^H y

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

    print(f'\n\n ................Adjoint test for {UpdaterType}................\n')
    print(f"<A x, y>   = {lhs:.6e}")
    print(f"<x, A^H y> = {rhs:.6e}")
    print(f"relative error = {rel_err:.3e}")


def _test_generic_fwd(UpdaterType, projector_type, useGPU):
    if useGPU and not yrt.compiledWithCuda():
        pytest.skip('Code not compiled with cuda. Skipping...')
    prep_dict = prepare_data(UpdaterType)
    out_lm = prep_dict['lm']
    proj_params = prep_dict['proj_params']
    img_params = prep_dict['img_params']
    HBasis_np = prep_dict['HBasis']
    rank = prep_dict['rank']
    T = prep_dict['T']
    ref_load = fwd_bwd[projector_type]

    oper_test = oper_func_gpu if useGPU else oper_func_cpu
    oper = oper_test(projector_type)(proj_params)
    count = out_lm.count()

    # Store A(W) * H computed from OperatorProjectorParams.DEFAULT4D or LR
    y_proj_lr = yrt.ProjectionListAlias(out_lm)
    y_proj_lr_np = np.require(np.zeros(count), requirements=['C'], dtype=np.float32)
    y_proj_lr.bind(y_proj_lr_np)

    x_lr_img = yrt.ImageAlias(img_params)
    x_np_test = ref_load['x_ref_np'] if 'DEFAULT4D' in UpdaterType else ref_load['W_np']
    x_lr_img.bind(x_np_test)

    oper.applyA(x_lr_img, y_proj_lr)

    np.testing.assert_allclose(ref_load['y_proj_ref_np'], y_proj_lr_np, atol=1e-5, rtol=1e-5)
    print(f'\n\n ................Forward test passed for {UpdaterType}!................\n')


def _test_generic_bwd(UpdaterType, projector_type, useGPU, updateH=False):
    if useGPU and not yrt.compiledWithCuda():
        pytest.skip('Code not compiled with cuda. Skipping...')
    prep_dict = prepare_data(UpdaterType)
    out_lm = prep_dict['lm']
    proj_params = prep_dict['proj_params']
    img_params = prep_dict['img_params']
    HBasis_np = prep_dict['HBasis']
    rank = prep_dict['rank']
    T = prep_dict['T']
    ref_load = fwd_bwd[projector_type]

    oper_test = oper_func_gpu if useGPU else oper_func_cpu
    oper = oper_test(projector_type)(proj_params)

    x_np_test = ref_load['x_ref_np'] if 'DEFAULT4D' in UpdaterType else ref_load['W_np']
    y_proj_ref = yrt.ProjectionListAlias(out_lm)
    y_proj_ref.bind(ref_load['y_proj_ref_np'])

    if not updateH:
        # Store A^T(y * H^T) computed from OperatorProjectorParams.DEFAULT4D or LR
        W_lr_np_backward = np.require(np.zeros_like(x_np_test).astype(np.float32), requirements=['C'])
        W_lr_img_backward = yrt.ImageAlias(img_params)
        W_lr_img_backward.bind(W_lr_np_backward)
        oper.applyAH(y_proj_ref, W_lr_img_backward)

        backward_ref = (ref_load['x_ref_np_backward']
                        if 'DEFAULT4D' in UpdaterType else ref_load['W_ref_np_backward'])

        np.testing.assert_allclose(backward_ref, W_lr_np_backward, atol=1e-5, rtol=2e-3)
        print(f'\n\n ................Backward test passed for {UpdaterType}!................\n')

    else:
        if 'LR' not in UpdaterType:
            raise ValueError(f"Cannot apply H-update backward on an opertor that is not LR (given {UpdaterType})")
        x_lr_img = yrt.ImageAlias(img_params)
        x_lr_img.bind(x_np_test)

        proj_params.updateH = updateH
        HBasisWrite_np = np.require(np.zeros_like(HBasis_np).astype(np.float32), requirements=['C'])
        oper_H = oper_test(projector_type)(proj_params)
        oper_H.setUpdaterLRHBasisWrite(HBasisWrite_np)
        HBasis_copy = copy.deepcopy(oper_H.getUpdaterLRHBasis())
        x_np_test_copy = copy.deepcopy(x_np_test)
        assert oper_H.getUpdaterLRUpdateH() == proj_params.updateH

        oper_H.applyAH(y_proj_ref, x_lr_img)

        np.testing.assert_allclose(HBasis_copy, oper_H.getUpdaterLRHBasis())
        np.testing.assert_allclose(x_np_test_copy, x_np_test)
        H_lr_np_backward = oper_H.getUpdaterLRHBasisWrite()
        np.testing.assert_allclose(ref_load['H_ref_np_backward'], H_lr_np_backward, atol=1e-5, rtol=3e-3)
        print(f'\n\n ................Backward test passed for {UpdaterType} (H update)!................\n')


# %% --------------------------------- ADJOINT TESTS ---------------------------------
# %% CPU tests

# Siddon
def test_adjoint_DEFAULT3D_Siddon_CPU():
    _test_adjoint_generic(UpdaterType='DEFAULT3D', projector_type='Siddon', useGPU=False)


def test_adjoint_DEFAULT4D_Siddon_CPU():
    _test_adjoint_generic('DEFAULT4D', projector_type='Siddon', useGPU=False)


def test_adjoint_LR_Siddon_CPU():
    _test_adjoint_generic('LR', projector_type='Siddon', useGPU=False)


# DD
def test_adjoint_DEFAULT3D_DD_CPU():
    _test_adjoint_generic(UpdaterType='DEFAULT3D', projector_type='DD', useGPU=False)


def test_adjoint_DEFAULT4D_DD_CPU():
    _test_adjoint_generic('DEFAULT4D', projector_type='DD', useGPU=False)


def test_adjoint_LR_DD_CPU():
    _test_adjoint_generic('LR', projector_type='DD', useGPU=False)


# %% GPU tests

# Siddon
def test_adjoint_DEFAULT3D_Siddon_GPU():
    _test_adjoint_generic(UpdaterType='DEFAULT3D', projector_type='Siddon', useGPU=True)


def test_adjoint_DEFAULT4D_Siddon_GPU():
    _test_adjoint_generic('DEFAULT4D', projector_type='Siddon', useGPU=True)


def test_adjoint_LR_Siddon_GPU():
    _test_adjoint_generic('LR', projector_type='Siddon', useGPU=True)


# DD
def test_adjoint_DEFAULT3D_DD_GPU():
    _test_adjoint_generic(UpdaterType='DEFAULT3D', projector_type='DD', useGPU=True)


def test_adjoint_DEFAULT4D_DD_GPU():
    _test_adjoint_generic('DEFAULT4D', projector_type='DD', useGPU=True)


def test_adjoint_LR_DD_GPU():
    _test_adjoint_generic('LR', projector_type='DD', useGPU=True)


# %% --------------------------------- TEST FORWARD ---------------------------------
# %% CPU tests

# Siddon
def test_fwd_DEFAULT4D_Siddon_CPU():
    _test_generic_fwd('DEFAULT4D', projector_type='Siddon', useGPU=False)


def test_fwd_LR_Siddon_CPU():
    _test_generic_fwd('LR', projector_type='Siddon', useGPU=False)


# DD
def test_fwd_DEFAULT4D_DD_CPU():
    _test_generic_fwd('DEFAULT4D', projector_type='DD', useGPU=False)


def test_fwd_LR_DD_CPU():
    _test_generic_fwd('LR', projector_type='DD', useGPU=False)


# %% GPU tests

# Siddon
def test_fwd_DEFAULT4D_Siddon_GPU():
    _test_generic_fwd('DEFAULT4D', projector_type='Siddon', useGPU=True)


def test_fwd_LR_Siddon_GPU():
    _test_generic_fwd('LR', projector_type='Siddon', useGPU=True)


# DD
def test_fwd_DEFAULT4D_DD_GPU():
    _test_generic_fwd('DEFAULT4D', projector_type='DD', useGPU=True)


def test_fwd_LR_DD_GPU():
    _test_generic_fwd('LR', projector_type='DD', useGPU=True)


# %% --------------------------------- TEST BACKWARD ---------------------------------
# %% CPU tests

# Siddon
def test_bwd_DEFAULT4D_Siddon_CPU():
    _test_generic_bwd('DEFAULT4D', projector_type='Siddon', useGPU=False)


def test_bwd_LR_Siddon_CPU():
    _test_generic_bwd('LR', projector_type='Siddon', useGPU=False)


def test_bwd_LR_Hupdate_Siddon_CPU():
    _test_generic_bwd('LR', projector_type='Siddon', useGPU=False, updateH=True)


# DD
def test_bwd_DEFAULT4D_DD_CPU():
    _test_generic_bwd('DEFAULT4D', projector_type='DD', useGPU=False)


def test_bwd_LR_DD_CPU():
    _test_generic_bwd('LR', projector_type='DD', useGPU=False)


def test_bwd_LR_Hupdate_DD_CPU():
    _test_generic_bwd('LR', projector_type='DD', useGPU=False, updateH=True)


# %% GPU tests

# Siddon
def test_bwd_DEFAULT4D_Siddon_GPU():
    _test_generic_bwd('DEFAULT4D', projector_type='Siddon', useGPU=True)


def test_bwd_LR_Siddon_GPU():
    _test_generic_bwd('LR', projector_type='Siddon', useGPU=True)


def test_bwd_LR_Hupdate_Siddon_GPU():
    _test_generic_bwd('LR', projector_type='Siddon', useGPU=True, updateH=True)


# DD
def test_bwd_DEFAULT4D_DD_GPU():
    _test_generic_bwd('DEFAULT4D', projector_type='DD', useGPU=True)


def test_bwd_LR_DD_GPU():
    _test_generic_bwd('LR', projector_type='DD', useGPU=True)


def test_bwd_LR_Hupdate_DD_GPU():
    _test_generic_bwd('LR', projector_type='DD', useGPU=True, updateH=True)
