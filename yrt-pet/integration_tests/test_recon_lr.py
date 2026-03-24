import sys
import os

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(fold_py))

import pyyrtpet as yrt
import os
import numpy as np
import socket
import json
import scipy
import importlib
import pickle
import tqdm
import re
import glob
import copy
import gc
import datetime
from collections import OrderedDict
import itertools
import time

import helper as _helper

fold_data = _helper.fold_data
fold_out = _helper.fold_out
fold_bin = _helper.fold_bin

fold_uhr2d = os.path.join(fold_data, 'uhr2d')
scanner_path = os.path.join(fold_uhr2d, 'UHR2D.json')
fold_uhr2d_hbasis = os.path.join(fold_uhr2d, 'hbasis')

## Get paths

HOME_DIR = os.path.expanduser("~")


def prepare_data(UpdaterType):
    out_lm = yrt.ListModeLUTOwned(
        scanner, os.path.join(fold_ge2d_dyn, "shepp_logan_dyn.lmDat")
    )
    img_params = yrt.ImageParams(os.path.join(fold_ge2d_dyn, "img_params_2d_dyn.json"))
    rank = 1 if UpdaterType == "DEFAULT3D" else 5
    T = 1 if UpdaterType == "DEFAULT3D" else 20
    img_params.num_frames = rank if "LR" in UpdaterType else T

    if UpdaterType in ["DEFAULT4D", "LR"]:
        HBasis_np = pickle.load(
            open(os.path.join(fold_ge2d_dyn, f"HBasis_r{rank}.pik"), "rb")
        )["HBasis"].astype(np.float32)
        HBasis_np = np.require(HBasis_np, requirements=["C"])

        max_timestamp = np.max(out_lm.getTimestampArray())
        dynamic_framing_array = np.require(
            np.linspace(0, max_timestamp + 1, num=T + 1, dtype=np.uint32),
            requirements=["C"],
        )

        df = yrt.DynamicFraming(dynamic_framing_array)
        out_lm.addDynamicFraming(df)
    else:
        df = None
        HBasis_np = None

    bin_iter = out_lm.getBinIter(num_subsets, idx_subset)

    count = out_lm.count()
    assert count == bin_iter.size()

    proj_params = yrt.OperatorProjectorParams(scanner)
    proj_params.binIter = bin_iter
    proj_params.projectorUpdaterType = getattr(yrt.OperatorProjectorParams, UpdaterType)

    if UpdaterType == "LR":
        proj_params.setHBasisFromNumpy(HBasis_np)

    return {
        "lm": out_lm,
        "proj_params": proj_params,
        "df": df,
        "bin_iter": bin_iter,
        "rank": rank,
        "T": T,
        "img_params": img_params,
        "HBasis": HBasis_np,
    }


##

fold_data_old = os.path.expanduser("~/yrt-pet-integration_tests/test_data")
fold_ge2d_dyn = os.path.join(fold_data_old, "phantom_dynamic")
fold_ge2d = os.path.join(fold_data_old, "phantom_static")
scanner = yrt.Scanner(os.path.join(fold_ge2d_dyn, "UHR2D.json"))

UpdaterType_list = ["DEFAULT3D", "DEFAULT4D", "LR"]

UpdaterTypeDynamic_list = list(UpdaterType_list)
UpdaterTypeDynamic_list.remove("DEFAULT3D")

num_subsets = 1
idx_subset = 0

rng = np.random.default_rng(0)

run_adjoint_tests = False
run_fwd_bwd_tests = False
run_3d_osem_tests = False
run_4d_osem_tests = False
run_lr_osem_tests = False
run_lr_H_osem_tests = True
run_lr_it_osem_tests = False

# %% ADJOINT TESTS

if run_adjoint_tests:
    for UpdaterType in UpdaterType_list:
        prep_dict = prepare_data(UpdaterType)
        out_lm = prep_dict["lm"]
        proj_params = prep_dict["proj_params"]
        img_params = prep_dict["img_params"]
        HBasis_np = prep_dict["HBasis"]
        rank = prep_dict["rank"]

        oper = yrt.OperatorProjectorSiddon_GPU(proj_params)
        count = out_lm.count()

        # --- sizes ---
        nt = img_params.num_frames
        nx_2d = img_params.nx
        ny_2d = img_params.ny
        nz_2d = img_params.nz

        # --- random x (image) and y (projection) ---
        x_np = np.abs(
            np.require(
                rng.standard_normal((nt, nz_2d, ny_2d, nx_2d), dtype=np.float32),
                requirements=["C"],
            )
        )
        y_np = np.abs(
            np.require(rng.standard_normal(count, dtype=np.float32), requirements=["C"])
        )

        # --- output holders (will be written by the ops) ---
        Ax_np = np.require(np.zeros_like(y_np), requirements=["C"])  # will hold A x
        AHy_np = np.require(np.zeros_like(x_np), requirements=["C"])  # will hold A^H y

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

        print(f"\n\n ................Adjoint test for {UpdaterType}................\n")
        print(f"<A x, y>   = {lhs:.6e}")
        print(f"<x, A^H y> = {rhs:.6e}")
        print(f"relative error = {rel_err:.3e}")

    print(
        "\n\n----------------------------------All Adjoint tests passed!----------------------------------\n"
    )

# %% TEST FORWARD AND BACKWARDS

if run_fwd_bwd_tests:
    # Prepare reference CPU forward and backward
    prep_dict_ref = prepare_data("DEFAULT4D")
    out_lm_ref = prep_dict_ref["lm"]
    proj_params_ref = prep_dict_ref["proj_params"]
    img_params_ref = prep_dict_ref["img_params"]
    HBasis_np_ref = prep_dict_ref["HBasis"]
    rank_ref = prep_dict_ref["rank"]
    T_ref = prep_dict_ref["T"]

    oper_ref_cpu = yrt.OperatorProjectorSiddon(proj_params_ref)
    count = out_lm_ref.count()

    # Prepare variables for projections
    y_proj_ref_np = np.require(np.zeros(count), requirements=["C"], dtype=np.float32)
    x_np = np.abs(
        np.require(
            rng.standard_normal(
                (rank_ref, img_params_ref.nz, img_params_ref.ny, img_params_ref.nx),
                dtype=np.float32,
            ),
            requirements=["C"],
        )
    )

    y_proj_ref = yrt.ProjectionListAlias(out_lm_ref)
    y_proj_ref.bind(y_proj_ref_np)
    x_ref_np = np.tensordot(HBasis_np_ref.T, x_np, (1, 0))
    x_ref_img = yrt.ImageAlias(img_params_ref)
    x_ref_img.bind(x_ref_np)

    x_ref_np_backward = np.require(
        np.zeros_like(x_ref_np).astype(np.float32), requirements=["C"]
    )
    x_ref_img_backward = yrt.ImageAlias(img_params_ref)
    x_ref_img_backward.bind(x_ref_np_backward)

    # Compute and store A(W * H) in y_proj_ref
    oper_ref_cpu.applyA(x_ref_img, y_proj_ref)
    # Compute and store A^T(y) * H^T in x_ref_img_backward
    oper_ref_cpu.applyAH(y_proj_ref, x_ref_img_backward)

    W_ref_np_backward = np.tensordot(HBasis_np_ref, x_ref_np_backward, (1, 0))
    H_ref_np_backward = np.tensordot(HBasis_np_ref, x_ref_np_backward, (1, 0))

    for UpdaterType in UpdaterTypeDynamic_list:
        prep_dict = prepare_data(UpdaterType)
        out_lm = prep_dict["lm"]
        proj_params = prep_dict["proj_params"]
        img_params = prep_dict["img_params"]
        HBasis_np = prep_dict["HBasis"]
        rank = prep_dict["rank"]
        T = prep_dict["T"]

        oper = yrt.OperatorProjectorSiddon_GPU(proj_params)
        count = out_lm.count()

        # Store A(W) * H computed from OperatorProjectorParams.DEFAULT4D or LR
        y_proj_lr = yrt.ProjectionListAlias(out_lm)
        y_proj_lr_np = np.require(np.zeros(count), requirements=["C"], dtype=np.float32)
        y_proj_lr.bind(y_proj_lr_np)

        x_lr_img = yrt.ImageAlias(img_params)
        x_np_test = x_ref_np if "DEFAULT4D" in UpdaterType else x_np
        x_lr_img.bind(x_np_test)

        oper.applyA(x_lr_img, y_proj_lr)

        np.testing.assert_allclose(y_proj_ref_np, y_proj_lr_np, atol=1e-5, rtol=1e-5)
        print(
            f"\n\n ................Forward test passed for {UpdaterType}!................\n"
        )

        # Store A^T(y * H^T) computed from OperatorProjectorParams.DEFAULT4D or LR
        W_lr_np_backward = np.require(
            np.zeros_like(x_np_test).astype(np.float32), requirements=["C"]
        )
        W_lr_img_backward = yrt.ImageAlias(img_params)
        W_lr_img_backward.bind(W_lr_np_backward)
        oper.applyAH(y_proj_lr, W_lr_img_backward)

        backward_ref = (
            x_ref_np_backward if "DEFAULT4D" in UpdaterType else W_ref_np_backward
        )

        np.testing.assert_allclose(backward_ref, W_lr_np_backward, atol=1e-5, rtol=2e-3)
        print(
            f"\n\n ................Backward test passed for {UpdaterType}!................\n"
        )

    print(
        "\n\n----------------------------------All FWD and BWD tests passed!----------------------------------\n"
    )

## ## List mode test static (DEFAULT3D)

if run_3d_osem_tests:
    dataset_load = yrt.ListModeLUTOwned(
        scanner, os.path.join(fold_ge2d, "shepp_logan.lmDat")
    )
    img_params = yrt.ImageParams(os.path.join(fold_ge2d, "img_params_2d.json"))

    ref_img = yrt.ImageOwned(
        img_params, os.path.join(fold_ge2d, "shepp_logan_mlem_his_400.nii")
    )
    np_ref_img = np.array(ref_img, copy=False)

    osem = yrt.createOSEM(scanner, useGPU=True)
    osem.setImageParams(img_params)
    osem.setProjectorUpdaterType(yrt.OperatorProjectorParams.DEFAULT3D)
    osem.setNumRays(1)
    osem.num_MLEM_iterations = 30
    osem.num_OSEM_subsets = 1
    osem.setDataInput(dataset_load)
    sens_img = osem.generateSensitivityImages()
    osem.setSensitivityImage(sens_img[0])
    time_static = time.time()
    out_img = osem.reconstruct()
    elapsed_static = time.time() - time_static
    print(f"\nStatic list mode reconstruction: {elapsed_static:.1f} seconds\n")

    np_out_img_static = np.array(out_img, copy=True)
    # TODO NOW CHECK IF IMAGES ARE THE SAME ?

## ## List mode test dynamic (DEFAULT4D)

num_MLEM_iterations = 60
num_OSEM_subsets = 1

if run_4d_osem_tests:
    load_ref_dict = pickle.load(
        open(
            os.path.join(
                fold_ge2d_dyn,
                f"shepp_logan_mlem_lm_it{num_MLEM_iterations}_"
                f"sub{num_OSEM_subsets}.pik",
            ),
            "rb",
        )
    )
    np_ref_img = load_ref_dict["x"]
    dynamic_framing_array = load_ref_dict["dynamic_framing"]

    dataset_load = yrt.ListModeLUTOwned(
        scanner, os.path.join(fold_ge2d_dyn, "shepp_logan_dyn.lmDat")
    )
    img_params = yrt.ImageParams(os.path.join(fold_ge2d_dyn, "img_params_2d_dyn.json"))
    df = yrt.DynamicFraming(dynamic_framing_array)
    dataset_load.addDynamicFraming(df)

    osem = yrt.createOSEM(scanner, useGPU=True)
    osem.setImageParams(img_params)
    osem.setProjectorUpdaterType(yrt.OperatorProjectorParams.DEFAULT4D)
    osem.setNumRays(1)
    osem.num_MLEM_iterations = num_MLEM_iterations
    osem.num_OSEM_subsets = num_OSEM_subsets
    osem.setDataInput(dataset_load)
    sens_img = osem.generateSensitivityImages()
    osem.setSensitivityImage(sens_img[0])
    time_static = time.time()
    out_img = osem.reconstruct()
    elapsed_static = time.time() - time_static
    print(
        f"\n\nDEFAULT4D list mode reconstruction: {elapsed_static:.1f} seconds\n"
        f'GPU {(load_ref_dict["elapsed_time"] / elapsed_static):.2f}x time faster than CPU \n'
    )

    np_out_img = np.array(out_img, copy=True)
    np.testing.assert_allclose(np_out_img, np_ref_img, atol=1e-5, rtol=1e-3)
    print(
        "----------------------------------DEFAULT4D: Assert tests passed!----------------------------------\n"
    )

    # TODO NOW CHECK IF IMAGES ARE THE SAME ?

## ## List mode test dynamic (LR)

num_MLEM_iterations = 30
num_OSEM_subsets = 1
rank = 5

if run_lr_osem_tests:
    load_ref_dict = pickle.load(
        open(
            os.path.join(
                fold_ge2d_dyn,
                f"shepp_logan_mlem_lm_lr_it{num_MLEM_iterations}_"
                f"sub{num_OSEM_subsets}_r{rank}.pik",
            ),
            "rb",
        )
    )
    np_ref_img = load_ref_dict["x"]
    np_ref_W = load_ref_dict["W"]
    HBasis_np = load_ref_dict["H_orig"]
    dynamic_framing_array = load_ref_dict["dynamic_framing"]

    dataset_load = yrt.ListModeLUTOwned(
        scanner, os.path.join(fold_ge2d_dyn, "shepp_logan_dyn.lmDat")
    )
    img_params = yrt.ImageParams(os.path.join(fold_ge2d_dyn, "img_params_2d_dyn.json"))
    img_params.num_frames = rank
    df = yrt.DynamicFraming(dynamic_framing_array)
    dataset_load.addDynamicFraming(df)

    osem = yrt.createOSEM(scanner, useGPU=True)
    osem.setImageParams(img_params)
    osem.setProjectorUpdaterType(yrt.OperatorProjectorParams.LR)
    osem.setHBasisFromNumpy(HBasis_np)

    # quick check that H is correctly stored
    Hbasis_osem = osem.getHBasisNumpy()
    assert np.allclose(Hbasis_osem, HBasis_np)

    osem.setNumRays(1)
    osem.num_MLEM_iterations = num_MLEM_iterations
    osem.num_OSEM_subsets = num_OSEM_subsets
    osem.setDataInput(dataset_load)
    sens_img = osem.generateSensitivityImages()
    osem.setSensitivityImage(sens_img[0])
    time_static = time.time()
    out_W = osem.reconstruct()
    elapsed_static = time.time() - time_static
    print(
        f"\n\nLR list mode reconstruction: {elapsed_static:.1f} seconds\n"
        f'GPU {(load_ref_dict["elapsed_time"] / elapsed_static):.2f}x time faster than CPU \n'
    )

    np_out_W = np.array(out_W, copy=True)
    np_out_img = np.tensordot(HBasis_np.T, np_out_W, (1, 0))

    # check W is the same
    # np.testing.assert_allclose(np_out_W, np_ref_W, atol=0, rtol=2e-2)
    # check image is also the same (should be if H is the same)
    np.testing.assert_allclose(np_out_img, np_ref_img, atol=0, rtol=2e-2)

    print(
        "----------------------------------LR: Assert tests passed!----------------------------------\n"
    )

    # TODO NOW CHECK IF IMAGES ARE THE SAME ?

## ## List mode test dynamic (LR with JUST H update)

osem_W_init = {"num_MLEM_iterations": 1, "num_OSEM_subsets": 1}
num_MLEM_iterations = 1
num_OSEM_subsets = 1
rank = 5
lr_H_recon_fname = (
    f"shepp_logan_mlem_lm_lr_H_it{num_MLEM_iterations}_"
    f"sub{num_OSEM_subsets}_r{rank}"
)

if run_lr_H_osem_tests:
    lr_recon_fname = (
        f'shepp_logan_mlem_lm_lr_it{osem_W_init["num_MLEM_iterations"]}_'
        f'sub{osem_W_init["num_OSEM_subsets"]}_r{rank}'
    )
    load_ref_dict = pickle.load(
        open(os.path.join(fold_ge2d_dyn, lr_recon_fname + ".pik"), "rb")
    )
    np_ref_img = load_ref_dict["x"]
    np_ref_W = load_ref_dict["W"]
    HBasis_np = load_ref_dict["H_orig"]
    HBasis_np_orig = np.copy(HBasis_np)
    dynamic_framing_array = load_ref_dict["dynamic_framing"]

    dataset_load = yrt.ListModeLUTOwned(
        scanner, os.path.join(fold_ge2d_dyn, "shepp_logan_dyn.lmDat")
    )
    img_params = yrt.ImageParams(os.path.join(fold_ge2d_dyn, "img_params_2d_dyn.json"))
    img_params.num_frames = rank
    df = yrt.DynamicFraming(dynamic_framing_array)
    dataset_load.addDynamicFraming(df)

    W_init = yrt.ImageOwned(
        img_params, os.path.join(fold_ge2d_dyn, lr_recon_fname + ".nii")
    )
    np_W_init = np.array(W_init, copy=True)
    load_H_ref_dict = pickle.load(
        open(
            os.path.join(fold_ge2d_dyn, lr_recon_fname, lr_H_recon_fname + ".pik"), "rb"
        )
    )
    assert np.allclose(HBasis_np_orig, load_H_ref_dict["H_orig"])

    osem_H = yrt.createOSEM(scanner, useGPU=False)
    osem_H.setImageParams(img_params)
    osem_H.setProjectorUpdaterType(yrt.OperatorProjectorParams.LR)
    osem_H.setHBasisFromNumpy(HBasis_np)
    osem_H.setUpdateH(True)
    osem_H.initialEstimate = W_init

    osem_H.setNumRays(1)
    osem_H.num_MLEM_iterations = num_MLEM_iterations
    osem_H.num_OSEM_subsets = num_OSEM_subsets
    osem_H.setDataInput(dataset_load)
    sens_img = osem_H.generateSensitivityImages()
    osem_H.setSensitivityImage(sens_img[0])
    time_static = time.time()
    out_H = osem_H.reconstruct()
    elapsed_static = time.time() - time_static
    print(
        f"\n\nLR list mode reconstruction: {elapsed_static:.1f} seconds\n"
        f'GPU {(load_H_ref_dict["elapsed_time"] / elapsed_static):.2f}x time faster than CPU \n'
    )

    # Check that W did not change during osem.reconstruct()
    np.testing.assert_allclose(np_W_init, np.array(out_H, copy=False), rtol=1e-4)
    # Check that H changed compared to input HBasis
    assert not np.allclose(HBasis_np, HBasis_np_orig)

    # Load CPU recon and compare with GPU update of H
    np.testing.assert_allclose(HBasis_np, load_H_ref_dict["H"], rtol=1e-4)

    print(
        "----------------------------------LR (H update): Assert tests passed!----------------------------------\n"
    )

    np_out_img = np.tensordot(HBasis_np.T, np_W_init, (1, 0))
    np_img = np.tensordot(HBasis_np_orig.T, np_W_init, (1, 0))

    # TODO NOW CHECK IF IMAGES ARE THE SAME ?


## ## List mode test dynamic (LR with H update)

num_MLEM_iterations = 1
num_OSEM_subsets = 1
rank = 5

num_iter_W = 1
num_iter_H = 1

if run_lr_it_osem_tests:
    load_ref_dict = pickle.load(
        open(
            os.path.join(
                fold_ge2d_dyn,
                f"shepp_logan_mlem_lm_lr_it{num_MLEM_iterations}_"
                f"sub{num_OSEM_subsets}_r{rank}.pik",
            ),
            "rb",
        )
    )
    np_ref_img = load_ref_dict["x"]
    np_ref_W = load_ref_dict["W"]
    HBasis_np = load_ref_dict["H_orig"]
    HBasis_np_orig = np.copy(HBasis_np)
    dynamic_framing_array = load_ref_dict["dynamic_framing"]

    dataset_load = yrt.ListModeLUTOwned(
        scanner, os.path.join(fold_ge2d_dyn, "shepp_logan_dyn.lmDat")
    )
    img_params = yrt.ImageParams(os.path.join(fold_ge2d_dyn, "img_params_2d_dyn.json"))
    img_params.num_frames = rank
    df = yrt.DynamicFraming(dynamic_framing_array)
    dataset_load.addDynamicFraming(df)

    for n_iter in range(num_MLEM_iterations):
        for _ in range(num_iter_W):
            osem = yrt.createOSEM(scanner, useGPU=True)
            osem.setImageParams(img_params)
            osem.setProjectorUpdaterType(yrt.OperatorProjectorParams.LR)
            osem.setHBasisFromNumpy(HBasis_np)
            osem.setUpdateH(False)

            # quick check that H is correctly stored
            Hbasis_osem = osem.getHBasisNumpy()
            assert np.allclose(Hbasis_osem, HBasis_np)

            if n_iter > 0:
                osem.initialEstimate = out_H

            osem.setNumRays(1)
            osem.num_MLEM_iterations = 1
            osem.num_OSEM_subsets = num_OSEM_subsets
            osem.setDataInput(dataset_load)

            if "sens_img" not in locals():
                sens_img = osem.generateSensitivityImages()

            osem.setSensitivityImage(sens_img[0])
            time_static = time.time()
            out_W = osem.reconstruct()
            elapsed_static = time.time() - time_static

        for _ in range(num_iter_H):
            osem_H = yrt.createOSEM(scanner, useGPU=True)
            osem_H.setImageParams(img_params)
            osem_H.setProjectorUpdaterType(yrt.OperatorProjectorParams.LR)
            osem_H.setHBasisFromNumpy(HBasis_np)
            osem_H.setUpdateH(True)

            osem_H.initialEstimate = out_W

            osem_H.setNumRays(1)
            osem_H.num_MLEM_iterations = 1
            osem_H.num_OSEM_subsets = num_OSEM_subsets
            osem_H.setDataInput(dataset_load)
            osem_H.setSensitivityImage(sens_img[0])
            time_static = time.time()
            out_H = osem_H.reconstruct()
            elapsed_static = time.time() - time_static
            print(
                f"\n\nLR list mode reconstruction: {elapsed_static:.1f} seconds\n"
                f'GPU {(load_ref_dict["elapsed_time"] / elapsed_static):.2f}x time faster than CPU \n'
            )

    np_out_H = np.array(out_H, copy=True)
    np_out_img = np.tensordot(HBasis_np.T, np_out_H, (1, 0))

    print(
        "----------------------------------LR: Assert tests passed!----------------------------------\n"
    )

    # TODO NOW CHECK IF IMAGES ARE THE SAME ?
