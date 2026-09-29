#!/bin/env python

# %% Imports

import os
import sys

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, fold_py)

import numpy as np
import pytest

import pyyrtpet as yrt
import helper as _helper

# %% Paths
fold_data = _helper.fold_data
fold_out = _helper.fold_out
fold_bin = _helper.fold_bin

fold_uhr2d = os.path.join(fold_data, "uhr2d")
scanner_path = os.path.join(fold_uhr2d, "UHR2D.json")
fold_uhr2d_ref = os.path.join(fold_uhr2d, "ref")
fold_uhr2d_hbasis = os.path.join(fold_uhr2d, "hbasis")

# %% OSEM reconstruction parameters
OSEM_3D_NUM_ITER = 400
OSEM_4D_NUM_ITER = 60
OSEM_LR_NUM_ITER = 30
OSEM_NUM_SUBSETS = 1
RANK = 5
OSEM_LR_H_NUM_ITER = 3
OSEM_NUM_SUBSETS_H = 10
RANK_UPDATEH = 6


# %% Fixtures


@pytest.fixture(scope="module")
def osem_3d_data():
    scanner = yrt.Scanner(scanner_path)
    dataset = yrt.ListModeLUTOwned(
        scanner, os.path.join(fold_uhr2d, "shepp_logan.lmDat")
    )
    img_params = yrt.ImageParams(os.path.join(fold_uhr2d,
                                              "img_params_2d.json"))
    ref_img = yrt.ImageOwned(
        img_params,
        os.path.join(fold_uhr2d_ref,
                     f"shepp_logan_mlem_lm_{OSEM_3D_NUM_ITER}.nii"))
    ref_img_np = np.array(ref_img, copy=False)

    return {"scanner": scanner,
            "dataset": dataset,
            "img_params": img_params,
            "ref_img_np": ref_img_np}


@pytest.fixture(scope="module")
def osem_4d_data():
    scanner = yrt.Scanner(scanner_path)
    ref_img_fname = (f"shepp_logan_mlem_lm_it{OSEM_4D_NUM_ITER}_" +
                 f"sub{OSEM_NUM_SUBSETS}.nii")
    ref_img_path = os.path.join(fold_uhr2d_ref, ref_img_fname)
    ref_img_y = yrt.ImageOwned(ref_img_path)
    ref_img_np = np.array(ref_img_y)
    df = yrt.DynamicFraming(os.path.join(fold_uhr2d,
                                         "dynamic_framing_shepp_logan.dyn"))

    dataset = yrt.ListModeLUTOwned(
        scanner, os.path.join(fold_uhr2d, "shepp_logan_dyn.lmDat")
    )
    img_params = yrt.ImageParams(os.path.join(fold_uhr2d,
                                              "img_params_2d_dyn.json"))
    dataset.addDynamicFraming(df)

    return {"scanner": scanner,
            "dataset": dataset,
            "img_params": img_params,
            "ref_img_np": ref_img_np}


@pytest.fixture(scope="module")
def osem_lr_W_data():
    scanner = yrt.Scanner(scanner_path)
    ref_fname_prefix = (
        f"shepp_logan_mlem_lm_lrW_it{OSEM_LR_NUM_ITER}_"
        f"sub{OSEM_NUM_SUBSETS}_r{RANK}"
    )
    ref_img = yrt.ImageOwned(
        os.path.join(fold_uhr2d_ref, f"{ref_fname_prefix}_recon_image.nii"))
    ref_img_np = np.array(ref_img, copy=True)
    Wref = yrt.ImageOwned(
        os.path.join(fold_uhr2d_ref, f"{ref_fname_prefix}_Wref.nii"))
    Wref_np = np.array(Wref, copy=True)
    Hinit_np = np.genfromtxt(os.path.join(fold_uhr2d_ref,
                                           f"{ref_fname_prefix}_Hinit.csv"),
                              delimiter=",")
    df = yrt.DynamicFraming(os.path.join(fold_uhr2d,
                                         "dynamic_framing_shepp_logan.dyn"))

    dataset = yrt.ListModeLUTOwned(
        scanner, os.path.join(fold_uhr2d, "shepp_logan_dyn.lmDat")
    )
    img_params = yrt.ImageParams(os.path.join(fold_uhr2d,
                                              "img_params_2d_dyn.json"))
    img_params.nt = RANK

    dataset.addDynamicFraming(df)

    return {
        "scanner": scanner,
        "dataset": dataset,
        "img_params": img_params,
        "ref_img_np": ref_img_np,
        "Wref_np": Wref_np,
        "Hinit_np": Hinit_np,
    }


@pytest.fixture(scope="module")
def osem_lr_H_data():
    scanner = yrt.Scanner(scanner_path)

    lr_recon_fname_prefix = (
        f"shepp_logan_mlem_lm_lrH_it{OSEM_LR_H_NUM_ITER}_"
        f"sub{OSEM_NUM_SUBSETS_H}_r{RANK_UPDATEH}"
    )

    # Load reference W reconstruction used as the initial estimate
    Hinit_np = np.genfromtxt(os.path.join(
        fold_uhr2d_ref, f"{lr_recon_fname_prefix}_Hinit.csv"))
    Href_np = np.genfromtxt(os.path.join(
        fold_uhr2d_ref, f"{lr_recon_fname_prefix}_Href.csv"))

    dataset = yrt.ListModeLUTOwned(
        scanner, os.path.join(fold_uhr2d, "shepp_logan_dyn.lmDat")
    )
    img_params = yrt.ImageParams(os.path.join(fold_uhr2d,
                                              "img_params_2d_dyn.json"))
    img_params.nt = RANK_UPDATEH
    df = yrt.DynamicFraming(os.path.join(fold_uhr2d,
                                         "dynamic_framing_shepp_logan.dyn"))
    dataset.addDynamicFraming(df)

    Winit = yrt.ImageOwned(
        img_params, os.path.join(fold_uhr2d_ref,
                                 f"{lr_recon_fname_prefix}_Winit.nii")
    )
    Winit_np = np.array(Winit, copy=True)

    return {
        "scanner": scanner,
        "dataset": dataset,
        "img_params": img_params,
        "Winit": Winit,
        "Winit_np": Winit_np,
        "Hinit_np": Hinit_np,
        "H_ref": Href_np
    }


# %% Tests


def test_uhr2d_shepp_logan_osem3d(osem_3d_data):
    d = osem_3d_data
    scanner = d["scanner"]
    img_params = d["img_params"]
    ref_img_np = d["ref_img_np"]
    lm = d["dataset"]

    osem = yrt.createOSEM(scanner, use_gpu=True)
    osem.setImageParams(img_params)
    osem.setNumRays(1)
    osem.num_MLEM_iterations = OSEM_3D_NUM_ITER
    osem.num_OSEM_subsets = OSEM_NUM_SUBSETS
    osem.setDataInput(lm)
    [sens_img] = osem.generateSensitivityImages()
    sens_img.writeToFile(os.path.join(
        fold_out, "test_uhr2d_shepp_logan_osem3d_sens_image.nii.gz"))
    osem.setSensitivityImage(sens_img)

    out_img = osem.reconstruct()
    out_img.writeToFile(os.path.join(
        fold_out, "test_uhr2d_shepp_logan_osem3d_recon_image.nii.gz"))

    out_img_np = np.array(out_img, copy=True)
    _helper.assert_allclose_with_threshold(
        out_img_np, ref_img_np, atol=0, rtol=0.01, threshold=1e-5)


def test_uhr2d_shepp_logan_osem4d(osem_4d_data):
    d = osem_4d_data
    scanner = d["scanner"]
    img_params = d["img_params"]
    ref_img_np = d["ref_img_np"]
    lm = d["dataset"]

    osem = yrt.createOSEM(scanner, use_gpu=True)
    osem.setImageParams(img_params)
    osem.setNumRays(1)
    osem.num_MLEM_iterations = OSEM_4D_NUM_ITER
    osem.num_OSEM_subsets = OSEM_NUM_SUBSETS
    osem.setDataInput(lm)
    [sens_img] = osem.generateSensitivityImages()
    sens_img.writeToFile(os.path.join(
        fold_out, "test_uhr2d_shepp_logan_osem4d_sens_image.nii.gz"))
    osem.setSensitivityImage(sens_img)

    out_img = osem.reconstruct()
    out_img.writeToFile(os.path.join(
        fold_out, "test_uhr2d_shepp_logan_osem4d_recon_image.nii.gz"))

    out_img_np = np.array(out_img, copy=True)
    _helper.assert_allclose_with_threshold(
        out_img_np, ref_img_np, atol=0, rtol=0.01, threshold=1e-5
    )


def test_uhr2d_shepp_logan_lrem_updatew(osem_lr_W_data):
    d = osem_lr_W_data
    scanner = d["scanner"]
    img_params = d["img_params"]
    Hinit_np = d["Hinit_np"]
    ref_img_np = d["ref_img_np"]
    Wref_np = d["Wref_np"]
    lm = d["dataset"]

    osem = yrt.createOSEM(scanner, use_gpu=True, is_low_rank=True)
    osem.setImageParams(img_params)
    osem.setProjectorUpdaterType(yrt.UpdaterType.LR)
    osem.setHBasisFromNumpy(Hinit_np)
    np.testing.assert_allclose(osem.getHBasisNumpy(), Hinit_np)
    osem.setNumRays(1)
    osem.num_MLEM_iterations = OSEM_LR_NUM_ITER
    osem.num_OSEM_subsets = OSEM_NUM_SUBSETS
    osem.setDataInput(lm)
    [sens_img] = osem.generateSensitivityImages()
    sens_img.writeToFile(os.path.join(
        fold_out, "test_uhr2d_shepp_logan_lrem_updatew_sens_image.nii.gz"))
    osem.setSensitivityImage(sens_img)

    out_W = osem.reconstruct()
    out_W.writeToFile(os.path.join(
        fold_out, "test_uhr2d_shepp_logan_lrem_updatew_W_image.nii.gz"))

    # Check W
    out_W_np = np.array(out_W, copy=True)
    _helper.assert_allclose_with_threshold(
        out_W_np, Wref_np, atol=0, rtol=2e-2, threshold=1e-6)

    # Check the resulting image
    out_img_np = np.tensordot(Hinit_np.T, out_W_np, (1, 0))

    # Save image
    img_params_full = yrt.ImageParams(img_params)
    img_params_full.nt = Hinit_np.shape[1]
    out_img = yrt.ImageAlias(img_params_full)
    out_img.bind(out_img_np)
    out_img.writeToFile(os.path.join(
        fold_out, "test_uhr2d_shepp_logan_lrem_updatew_recon_image.nii.gz"))

    # Check allclose
    _helper.assert_allclose_with_threshold(
        out_img_np, ref_img_np, atol=0, rtol=2e-2, threshold=1e-5)


def test_uhr2d_shepp_logan_lrem_updateh(osem_lr_H_data):
    """
    LR OSEM with H-update only (W fixed).
    Verifies that W is unchanged and that the updated H matches the CPU
    reference.
    """
    d = osem_lr_H_data
    scanner = d["scanner"]
    img_params = d["img_params"]
    HBasis_np = d["Hinit_np"]
    HBasis_np_orig = np.copy(HBasis_np)
    Winit = d["Winit"]
    Winit_np = d["Winit_np"]
    lm = d["dataset"]

    osem_H = yrt.createOSEM(scanner, use_gpu=False, is_low_rank=True)
    osem_H.setImageParams(img_params)
    osem_H.setProjectorUpdaterType(yrt.UpdaterType.LR)
    osem_H.setHBasisFromNumpy(HBasis_np)
    osem_H.setUpdateH(True)
    osem_H.setInitialEstimate(Winit)
    osem_H.setNumRays(1)
    osem_H.num_MLEM_iterations = OSEM_LR_H_NUM_ITER
    osem_H.num_OSEM_subsets = OSEM_NUM_SUBSETS_H
    osem_H.setDataInput(lm)
    sens_img = osem_H.generateSensitivityImages()
    osem_H.setSensitivityImage(sens_img[0])

    out_W = osem_H.reconstruct()

    # W must be unchanged after an H-only update
    np.testing.assert_allclose(Winit_np, np.array(out_W, copy=False), rtol=1e-4)
    # H must have changed from the original input
    assert not np.allclose(HBasis_np, HBasis_np_orig)
    # Updated H must match the CPU reference
    np.testing.assert_allclose(HBasis_np, d["H_ref"], rtol=1e-4)

    # Save the resulting image
    out_img_np = np.tensordot(HBasis_np.T, Winit_np, (1, 0))
    img_params_full = yrt.ImageParams(img_params)
    img_params_full.nt = HBasis_np.shape[1]
    out_img = yrt.ImageAlias(img_params_full)
    out_img.bind(out_img_np)
    out_img.writeToFile(os.path.join(
        fold_out, "test_uhr2d_shepp_logan_lrem_updateh_recon_image.nii.gz"))
