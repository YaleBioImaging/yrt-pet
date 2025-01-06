#!/bin/env python
"""Integration tests for YRT-PET"""

# %% Imports

import os
import sys
import json
import tempfile

import numpy as np

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(fold_py))
import pyyrtpet as yrt

# %% Helper functions

def make_scanner():
    # Create scanner
    scanner = yrt.Scanner(scannerName='test_scanner', axialFOV=25,
                          crystalSize_z=2.0, crystalSize_trans=2.0,
                          crystalDepth=10.0, scannerRadius=30.0,
                          detsPerRing=512, numRings=8, numDOI=1, maxRingDiff=7,
                          minAngDiff=0, detsPerBlock=32)
    det_coords = yrt.DetRegular(scanner)
    det_coords.generateLUT()
    scanner.setDetectorSetup(det_coords)
    return scanner


# %% List-mode


def test_list_mode():
    scanner = make_scanner()
    ts = np.arange(10)
    d1 = np.random.randint(0, 100, ts.size)
    d2 = np.random.randint(0, 100, ts.size)
    tof = np.random.randn(ts.size)
    data = {'ts': ts, 'd1': d1, 'd2': d2, 'tof': tof}
    for flag_tof in [False, True]:
        dtype_list = [('ts', np.uint32), ('d1', np.uint32), ('d2', np.uint32)]
        if flag_tof:
            dtype_list.append(('tof', np.float32))
        dtype = np.dtype(dtype_list)
        lm_np = np.zeros(ts.size, dtype=dtype)
        for dt in dtype_list:
            lm_np[dt[0]] = np.require(data[dt[0]], dtype=dt[1],
                                   requirements=['C_CONTIGUOUS'])
        with tempfile.TemporaryDirectory() as tmp_dir:
            lm_fname = os.path.join(tmp_dir, 'lm.dat')
            lm_np.tofile(lm_fname)
            lm_yrt = yrt.ListModeLUTOwned(scanner, lm_fname, flag_tof=flag_tof)
            lm_ts = lm_yrt.getTimestampArray()
            lm_d1 = lm_yrt.getDetector1Array()
            lm_d2 = lm_yrt.getDetector2Array()
            if flag_tof:
                lm_tof = lm_yrt.getTOFArray()
                np.testing.assert_allclose(tof, lm_tof)
            else:
                np.testing.assert_raises(RuntimeError, lm_yrt.getTOFArray)
            np.testing.assert_allclose(ts, lm_ts)
            np.testing.assert_allclose(d1, lm_d1)
            np.testing.assert_allclose(d2, lm_d2)


# %% Image transformation


def test_image_transform():

    min_val, max_val = 1, 10
    def rescale(sample): return (max_val - min_val) * sample + min_val

    # Simple translation
    x = rescale(np.random.random([12, 13, 14])).astype(np.float32)
    img_params = yrt.ImageParams(14, 13, 12, 28.0, 26.0, 24.0, 1, 2, 3)
    img = yrt.ImageAlias(img_params)
    img.bind(x)
    v_rot = yrt.Vector3D(0.0, 0.0, 0.0)
    v_tr = yrt.Vector3D(2.0, 0.0, 0.0)
    img_t = img.transformImage(v_rot, v_tr)
    x_t = np.array(img_t, copy=False)
    np.testing.assert_allclose(x[..., :-1], x_t[..., 1:], rtol=9e-6)
    # Simple rotation
    x = rescale(np.random.random([14, 12, 12])).astype(np.float32)
    img_params = yrt.ImageParams(12, 12, 14, 26.0, 26.0, 28.0)
    img = yrt.ImageAlias(img_params)
    img.bind(x)
    v_rot = yrt.Vector3D(0.0, 0.0, np.pi / 2)
    v_tr = yrt.Vector3D(0.0, 0.0, 0.0)
    img_t = img.transformImage(v_rot, v_tr)
    x_t = np.array(img_t, copy=False)
    np.testing.assert_allclose(np.moveaxis(x, 1, 2)[..., ::-1], x_t, rtol=9e-6)
