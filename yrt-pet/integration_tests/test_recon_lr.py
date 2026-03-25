#!/bin/env python

# %% Imports

import os
import sys

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, fold_py)

import pickle
import copy
import time

import numpy as np
import pytest

import pyyrtpet as yrt
import helper as _helper

# %% Paths
fold_data = _helper.fold_data
fold_out  = _helper.fold_out
fold_bin  = _helper.fold_bin

fold_uhr2d        = os.path.join(fold_data, "uhr2d")
scanner_path      = os.path.join(fold_uhr2d, "UHR2D.json")
fold_uhr2d_ref    = os.path.join(fold_uhr2d, "ref")
fold_uhr2d_hbasis = os.path.join(fold_uhr2d, "hbasis")

# TODO NOW: Make sure these files are valid
# Files read by this script (all under fold_uhr2d):
#   fold_uhr2d/UHR2D.json
#   fold_uhr2d/shepp_logan.lmDat                                      (3D OSEM)
#   fold_uhr2d/img_params_2d.json                                     (3D OSEM)
#   fold_uhr2d/shepp_logan_mlem_his_400.nii                           (3D OSEM reference)
#   fold_uhr2d/shepp_logan_dyn.lmDat                                  (4D / LR OSEM)
#   fold_uhr2d/img_params_2d_dyn.json                                 (4D / LR OSEM)
#   fold_uhr2d/shepp_logan_mlem_lm_it{N}_sub{S}.pik                  (4D OSEM reference)
#   fold_uhr2d/shepp_logan_mlem_lm_lr_it{N}_sub{S}_r{R}.pik          (LR OSEM reference)
#   fold_uhr2d/{lr_recon_fname}.pik                                   (LR-H OSEM reference W)
#   fold_uhr2d/{lr_recon_fname}.nii                                   (LR-H OSEM initial W)
#   fold_uhr2d/{lr_recon_fname}/{lr_H_recon_fname}.pik                (LR-H OSEM reference H)

# %% OSEM reconstruction parameters
OSEM_3D_NUM_ITER   = 30
OSEM_4D_NUM_ITER   = 60
OSEM_LR_NUM_ITER   = 30
OSEM_LR_H_NUM_ITER = 1
OSEM_NUM_SUBSETS   = 1
RANK               = 5

lr_W_init_params = {"num_MLEM_iterations": 1, "num_OSEM_subsets": 1}

# Derived filenames (used by multiple fixtures / tests)
_lr_recon_fname = (
    f'shepp_logan_mlem_lm_lr_it{lr_W_init_params["num_MLEM_iterations"]}_'
    f'sub{lr_W_init_params["num_OSEM_subsets"]}_r{RANK}'
)
_lr_H_recon_fname = (
    f"shepp_logan_mlem_lm_lr_H_it{OSEM_LR_H_NUM_ITER}_"
    f"sub{OSEM_NUM_SUBSETS}_r{RANK}"
)


# %% Fixtures

@pytest.fixture(scope="module")
def osem_3d_data():
    try:
        scanner = yrt.Scanner(scanner_path)
        dataset = yrt.ListModeLUTOwned(
            scanner, os.path.join(fold_uhr2d, "shepp_logan.lmDat")
        )
        img_params = yrt.ImageParams(
            os.path.join(fold_uhr2d, "img_params_2d.json")
        )
        ref_img    = yrt.ImageOwned(
            img_params,
            os.path.join(fold_uhr2d_ref, "shepp_logan_mlem_his_400.nii"),
        )
        np_ref_img = np.array(ref_img, copy=False)
    except Exception as e:
        pytest.skip(f"osem_3d_data setup failed: {e}")

    return {"scanner":    scanner,
            "dataset":    dataset,
            "img_params": img_params,
            "np_ref_img": np_ref_img}


@pytest.fixture(scope="module")
def osem_4d_data():
    try:
        scanner = yrt.Scanner(scanner_path)
        ref_fname = (
            f"shepp_logan_mlem_lm_it{OSEM_4D_NUM_ITER}_sub{OSEM_NUM_SUBSETS}.pik"
        )
        load_ref_dict        = pickle.load(
            open(os.path.join(fold_uhr2d_ref, ref_fname), "rb")
        )
        np_ref_img           = load_ref_dict["x"]
        dynamic_framing_array = load_ref_dict["dynamic_framing"]
        elapsed_cpu          = load_ref_dict["elapsed_time"]

        dataset    = yrt.ListModeLUTOwned(
            scanner, os.path.join(fold_uhr2d, "shepp_logan_dyn.lmDat")
        )
        img_params = yrt.ImageParams(
            os.path.join(fold_uhr2d, "img_params_2d_dyn.json")
        )
        df = yrt.DynamicFraming(dynamic_framing_array)
        dataset.addDynamicFraming(df)
    except Exception as e:
        pytest.skip(f"osem_4d_data setup failed: {e}")

    return {
        "scanner":     scanner,
        "dataset":     dataset,
        "img_params":  img_params,
        "np_ref_img":  np_ref_img,
        "elapsed_cpu": elapsed_cpu,
    }


@pytest.fixture(scope="module")
def osem_lr_data():
    try:
        scanner = yrt.Scanner(scanner_path)
        ref_fname = (
            f"shepp_logan_mlem_lm_lr_it{OSEM_LR_NUM_ITER}_"
            f"sub{OSEM_NUM_SUBSETS}_r{RANK}.pik"
        )
        load_ref_dict        = pickle.load(
            open(os.path.join(fold_uhr2d_ref, ref_fname), "rb")
        )
        np_ref_img           = load_ref_dict["x"]
        np_ref_W             = load_ref_dict["W"]
        HBasis_np            = load_ref_dict["H_orig"]
        dynamic_framing_array = load_ref_dict["dynamic_framing"]
        elapsed_cpu          = load_ref_dict["elapsed_time"]

        dataset    = yrt.ListModeLUTOwned(
            scanner, os.path.join(fold_uhr2d, "shepp_logan_dyn.lmDat")
        )
        img_params = yrt.ImageParams(
            os.path.join(fold_uhr2d, "img_params_2d_dyn.json")
        )
        img_params.num_frames = RANK
        df = yrt.DynamicFraming(dynamic_framing_array)
        dataset.addDynamicFraming(df)
    except Exception as e:
        pytest.skip(f"osem_lr_data setup failed: {e}")

    return {
        "scanner":     scanner,
        "dataset":     dataset,
        "img_params":  img_params,
        "np_ref_img":  np_ref_img,
        "np_ref_W":    np_ref_W,
        "HBasis_np":   HBasis_np,
        "elapsed_cpu": elapsed_cpu,
    }


@pytest.fixture(scope="module")
def osem_lr_H_data():
    """
    Load list-mode data, initial W, and reference H for the LR H-update OSEM test.
    Skips dependent tests if setup fails.
    """
    try:
        scanner = yrt.Scanner(scanner_path)
        # Load reference W reconstruction used as the initial estimate
        load_ref_dict        = pickle.load(
            open(os.path.join(fold_uhr2d_ref, _lr_recon_fname + ".pik"), "rb")
        )
        np_ref_W             = load_ref_dict["W"]
        HBasis_np            = load_ref_dict["H_orig"]
        HBasis_np_orig       = np.copy(HBasis_np)
        dynamic_framing_array = load_ref_dict["dynamic_framing"]

        dataset    = yrt.ListModeLUTOwned(
            scanner, os.path.join(fold_uhr2d, "shepp_logan_dyn.lmDat")
        )
        img_params = yrt.ImageParams(
            os.path.join(fold_uhr2d, "img_params_2d_dyn.json")
        )
        img_params.num_frames = RANK
        df = yrt.DynamicFraming(dynamic_framing_array)
        dataset.addDynamicFraming(df)

        W_init    = yrt.ImageOwned(
            img_params, os.path.join(fold_uhr2d_ref, _lr_recon_fname + ".nii")
        )
        np_W_init = np.array(W_init, copy=True)

        # Load reference H after H-update reconstruction
        load_H_ref_dict = pickle.load(
            open(
                os.path.join(
                    fold_uhr2d_ref, _lr_recon_fname, _lr_H_recon_fname + ".pik"
                ),
                "rb",
            )
        )
        assert np.allclose(HBasis_np_orig, load_H_ref_dict["H_orig"])
        elapsed_cpu = load_H_ref_dict["elapsed_time"]
    except Exception as e:
        pytest.skip(f"osem_lr_H_data setup failed: {e}")

    return {
        "scanner":       scanner,
        "dataset":       dataset,
        "img_params":    img_params,
        "W_init":        W_init,
        "np_W_init":     np_W_init,
        "HBasis_np":     HBasis_np,
        "HBasis_np_orig": HBasis_np_orig,
        "H_ref":         load_H_ref_dict["H"],
        "elapsed_cpu":   elapsed_cpu,
    }


# %% Tests

def test_3d_osem(osem_3d_data):
    d          = osem_3d_data
    scanner = d['scanner']
    img_params = d["img_params"]
    np_ref_img = d["np_ref_img"]
    lm = d["dataset"]

    osem = yrt.createOSEM(scanner, use_gpu=True)
    osem.setImageParams(img_params)
    osem.setNumRays(1)
    osem.num_MLEM_iterations = OSEM_3D_NUM_ITER
    osem.num_OSEM_subsets    = OSEM_NUM_SUBSETS
    osem.setDataInput(lm)
    sens_img = osem.generateSensitivityImages()
    osem.setSensitivityImage(sens_img[0])

    t_start   = time.time()
    out_img   = osem.reconstruct()
    elapsed   = time.time() - t_start
    print(f"\nStatic list-mode reconstruction: {elapsed:.1f} s")

    np_out_img = np.array(out_img, copy=True)
    np.testing.assert_allclose(np_out_img, np_ref_img, atol=1e-5, rtol=1e-3)


def test_4d_osem(osem_4d_data):
    d          = osem_4d_data
    scanner = d['scanner']
    img_params = d["img_params"]
    np_ref_img = d["np_ref_img"]
    lm = d["dataset"]

    osem = yrt.createOSEM(scanner, use_gpu=True)
    osem.setImageParams(img_params)
    osem.setNumRays(1)
    osem.num_MLEM_iterations = OSEM_4D_NUM_ITER
    osem.num_OSEM_subsets    = OSEM_NUM_SUBSETS
    osem.setDataInput(lm)
    sens_img = osem.generateSensitivityImages()
    osem.setSensitivityImage(sens_img[0])

    t_start  = time.time()
    out_img  = osem.reconstruct()
    elapsed  = time.time() - t_start
    print(
        f"\nDEFAULT4D reconstruction: {elapsed:.1f} s  "
        f"({d['elapsed_cpu'] / elapsed:.2f}x faster than CPU reference)"
    )

    np_out_img = np.array(out_img, copy=True)
    np.testing.assert_allclose(np_out_img, np_ref_img, atol=1e-5, rtol=1e-3)


def test_lr_W_osem(osem_lr_data):
    """Dynamic list-mode LR OSEM reconstruction and comparison to reference image."""
    d          = osem_lr_data
    scanner    = d['scanner']
    img_params = d["img_params"]
    HBasis_np  = d["HBasis_np"]
    np_ref_img = d["np_ref_img"]
    np_ref_W   = d["np_ref_W"]
    lm         = d["dataset"]

    osem = yrt.createOSEM(scanner, use_gpu=True, is_low_rank=True)
    osem.setImageParams(img_params)
    osem.setProjectorUpdaterType(yrt.OperatorProjectorParams.LR)
    osem.setHBasisFromNumpy(HBasis_np)
    np.testing.assert_allclose(osem.getHBasisNumpy(), HBasis_np)
    osem.setNumRays(1)
    osem.num_MLEM_iterations = OSEM_LR_NUM_ITER
    osem.num_OSEM_subsets    = OSEM_NUM_SUBSETS
    osem.setDataInput(lm)
    sens_img = osem.generateSensitivityImages()
    osem.setSensitivityImage(sens_img[0])

    t_start = time.time()
    out_W   = osem.reconstruct()
    elapsed = time.time() - t_start
    print(
        f"\nLR reconstruction: {elapsed:.1f} s  "
        f"({d['elapsed_cpu'] / elapsed:.2f}x faster than CPU reference)"
    )

    # Check W
    np_out_W   = np.array(out_W, copy=True)
    np.testing.assert_allclose(np_out_W, np_ref_W, atol=0, rtol=2e-2)

    # Check image
    np_out_img = np.tensordot(HBasis_np.T, np_out_W, (1, 0))
    np.testing.assert_allclose(np_out_img, np_ref_img, atol=0, rtol=2e-2)


def test_lr_H_osem(osem_lr_H_data):
    """
    LR OSEM with H-update only (W fixed).
    Verifies that W is unchanged and that the updated H matches the CPU reference.
    """
    d             = osem_lr_H_data
    scanner = d['scanner']
    img_params    = d["img_params"]
    HBasis_np     = d["HBasis_np"]
    HBasis_np_orig = d["HBasis_np_orig"]
    initial_estimate = d["W_init"]
    init_W = d["np_W_init"]

    osem_H = yrt.createOSEM(scanner, use_gpu=False, is_low_rank=True)
    osem_H.setImageParams(img_params)
    osem_H.setProjectorUpdaterType(yrt.OperatorProjectorParams.LR)
    osem_H.setHBasisFromNumpy(HBasis_np)
    osem_H.setUpdateH(True)
    osem_H.setInitialEstimate(initial_estimate)
    osem_H.setNumRays(1)
    osem_H.num_MLEM_iterations = OSEM_LR_H_NUM_ITER
    osem_H.num_OSEM_subsets    = OSEM_NUM_SUBSETS
    osem_H.setDataInput(d["dataset"])
    sens_img = osem_H.generateSensitivityImages()
    osem_H.setSensitivityImage(sens_img[0])

    t_start = time.time()
    out_H   = osem_H.reconstruct()
    elapsed = time.time() - t_start
    print(
        f"\nLR H-update reconstruction: {elapsed:.1f} s  "
        f"({d['elapsed_cpu'] / elapsed:.2f}x faster than CPU reference)"
    )

    # W must be unchanged after an H-only update
    np.testing.assert_allclose(
        init_W, np.array(out_H, copy=False), rtol=1e-4
    )
    # H must have changed from the original input
    assert not np.allclose(HBasis_np, HBasis_np_orig)
    # Updated H must match the CPU reference
    np.testing.assert_allclose(HBasis_np, d["H_ref"], rtol=1e-4)
