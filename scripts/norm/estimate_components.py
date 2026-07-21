#!/usr/bin/env python3
"""Generic component-based PET normalization estimator using pyyrtpet.

Estimates all components of the model:

    N(z,phi,r) = G x B(d1%bs,d2%bs) x d(d1%bs,d2%bs,r)
                 x A(z) x eps(d1) x eps(d2) x scale

for any scanner from a reference norm histogram (.his format).

Usage:
    python estimate_components.py scanner.json ref_norm.his output_dir
        [--bs 16] [--fan-sum-iters 5] [--min-count 0]
        [--n-planes-b all]
"""

import os, sys, json, time, struct
import numpy as np
import pyyrtpet as yrt

HISTO_HEADER_SIZE = 32
MAGIC_NUMBER = 732174000


# ── RWD format I/O ─────────────────────────────────────────────

def save_rwd(fname, data):
    with open(fname, 'wb') as f:
        np.array([MAGIC_NUMBER, data.ndim], dtype=np.int32).tofile(f)
        np.array(data.shape, dtype=np.uint64).tofile(f)
        data.astype(np.float32).tofile(f)


# ── .his plane reader ──────────────────────────────────────────

def read_his_plane(f, nphi, nr):
    raw = f.read(nphi * nr * 4)
    if len(raw) < nphi * nr * 4:
        return None
    return np.frombuffer(raw, dtype=np.float32).reshape(nphi, nr)


# ── Histogram mapping helpers (pure Python, mirrors Histogram3D) ──

def get_det_pair_in_same_ring(scanner, r_ring, phi):
    n = scanner.detsPerRing
    m_rCut = scanner.minAngDiff // 2
    if scanner.minAngDiff % 2 != 0 and n % 4 == 0:
        m_rCut += 1
    d01 = 0
    d02 = n // 2
    if phi % 2 != 0:
        d02 = n // 2 + 1
    dr1 = d01 + (r_ring - n // 4 + m_rCut)
    dr2 = d02 - (r_ring - n // 4 + m_rCut)
    if dr1 < 0:
        dr1 += n
    if dr2 < 0:
        dr2 += n
    d1 = (dr1 + phi // 2) % n
    d2 = (dr2 + phi // 2) % n
    if d1 > d2:
        d1, d2 = d2, d1
    return d1, d2


# ── Scanner properties (derived from raw params) ───────────────

def scanner_num_doi_poss(scanner):
    return scanner.numDOI * scanner.numDOI

def scanner_num_r_ring(scanner):
    return scanner.detsPerRing // 2 + 1 - scanner.minAngDiff

def scanner_num_r(scanner):
    return scanner_num_doi_poss(scanner) * scanner_num_r_ring(scanner)

def scanner_num_phi(scanner):
    return scanner.detsPerRing

def scanner_num_z_bin(scanner):
    dz_max = scanner.maxRingDiff
    nz = (dz_max + 1) * scanner.numRings \
         - (dz_max * (dz_max + 1)) // 2
    nz_diff = nz - scanner.numRings
    return nz + nz_diff


# ── Geometric factor G from LUT ────────────────────────────────

def compute_geo(scanner, lut):
    """Compute G = |n1*d|*|n2*d|/|d|^3 using real detector positions.

    Parameters
    ----------
    scanner : yrt.Scanner
        Scanner object.
    lut : ndarray, shape (num_dets, 6)
        LUT array from scanner.createLUT(), columns = [px,py,pz, nx,ny,nz].

    Returns
    -------
    geo_lut : ndarray, shape (maxRingDiff+1, nR_ring, 2, nDOI, nDOI)
    """
    pos = lut[:, :3]
    normv = lut[:, 3:]

    nRings = scanner.numRings
    nDets = scanner.detsPerRing
    nDOI = scanner.numDOI
    maxRingDiff = scanner.maxRingDiff
    minAngDiff = scanner.minAngDiff
    nr_ring = scanner_num_r_ring(scanner)
    doi_stride = nDets * nRings

    m_rCut = minAngDiff // 2
    if minAngDiff % 2 != 0 and nDets % 4 == 0:
        m_rCut += 1

    geo = np.zeros(
        (maxRingDiff + 1, nr_ring, 2, nDOI, nDOI), dtype=np.float64)

    for dz in range(maxRingDiff + 1):
        for rr in range(nr_ring):
            for parity in [0, 1]:
                for doi1 in range(nDOI):
                    for doi2 in range(nDOI):
                        d01 = 0
                        d02 = nDets // 2
                        if parity != 0:
                            d02 = nDets // 2 + 1
                        dr1 = d01 + (rr - nDets // 4 + m_rCut)
                        dr2 = d02 - (rr - nDets // 4 + m_rCut)
                        if dr1 < 0:
                            dr1 += nDets
                        if dr2 < 0:
                            dr2 += nDets
                        d1_ring = (dr1 + parity // 2) % nDets
                        d2_ring = (dr2 + parity // 2) % nDets
                        if d1_ring > d2_ring:
                            d1_ring, d2_ring = d2_ring, d1_ring

                        d1 = d1_ring + 0 * nDets + doi1 * doi_stride
                        d2 = d2_ring + dz * nDets + doi2 * doi_stride

                        p1 = pos[d1]
                        p2 = pos[d2]
                        n1 = normv[d1]
                        n2 = normv[d2]

                        d_vec = p2 - p1
                        dist_sq = np.dot(d_vec, d_vec)
                        if dist_sq < 1e-12:
                            continue
                        dist = np.sqrt(dist_sq)
                        cos1 = abs(np.dot(n1, d_vec))
                        cos2 = abs(np.dot(n2, d_vec))
                        geo[dz, rr, parity, doi1, doi2] = (
                            cos1 * cos2 / (dist_sq * dist))
    return geo


# ── Main estimation ────────────────────────────────────────────

NRING_AVG = 'all'

def estimate_components(
    scanner_json,
    ref_norm_path,
    output_dir,
    block_size=None,
    fan_sum_iters=5,
    fan_sum_stride=1,
    min_count=0,
    n_planes_b=NRING_AVG,
    smooth_b_sigma=0.0,
    verbose=True,
):
    """Estimate all normalization components from a reference norm.

    Parameters
    ----------
    scanner_json : str
        Path to scanner config .json.
    ref_norm_path : str
        Path to reference normalization .his.
    output_dir : str
        Directory to save component files.
    block_size : int or None
        Detectors per block. If None, uses scanner config.
    fan_sum_iters : int
        Number of fan-sum iterations for epsilon (default 5).
    min_count : int
        Exclude LORs with ref < min_count (for noisy data).
    n_planes_b : int or 'all'
        Number of direct planes to average for B estimation.
        'all' uses all direct planes.
    smooth_b_sigma : float
        If > 0, apply Gaussian smoothing to B with this sigma.
    verbose : bool
        Print progress.

    Returns
    -------
    dict of component arrays.
    Also saves all components to output_dir/.
    """
    t_start = time.time()

    s = yrt.Scanner(scanner_json)
    histo = yrt.Histogram3DAlias(s)

    nRings = s.numRings
    nDets = s.detsPerRing
    nR = scanner_num_r(s)
    nPhi = scanner_num_phi(s)
    nZ = scanner_num_z_bin(s)
    nDOI = s.numDOI
    nDOIPoss = scanner_num_doi_poss(s)
    bs = block_size or s.detsPerBlock
    nr_ring = scanner_num_r_ring(s)
    num_det = nRings * nDets * nDOI
    doi_stride = nDets * nRings
    plane_bytes = nPhi * nR * 4

    if verbose:
        print('=' * 60)
        print(f'Estimating norm components')
        print('=' * 60)
        print(f'  Rings={nRings}, Dets/ring={nDets}, DOI={nDOI}')
        print(f'  Block size={bs}, maxRingDiff={s.maxRingDiff}')
        print(f'  nZ={nZ}, nPhi={nPhi}, nR={nR}, nDet={num_det}')
        print(f'  Fan-sum iters={fan_sum_iters}, stride={fan_sum_stride},'
              f' min_count={min_count}')

    # ── G: from real detector positions (LUT) ──
    if verbose:
        print(f'\n[1/6] Computing G from LUT (createLUT)...')
    lut = s.createLUT()
    geo = compute_geo(s, lut)

    # ── Precompute d1/d2 maps (used by all estimation steps) ──
    if verbose:
        print(f'[1] Precomputing detector maps...')
    d1_ring_map = np.zeros((nr_ring, nPhi), dtype=np.int32)
    d2_ring_map = np.zeros((nr_ring, nPhi), dtype=np.int32)
    for rr in range(nr_ring):
        for phi in range(nPhi):
            d1, d2 = get_det_pair_in_same_ring(s, rr, phi)
            d1_ring_map[rr, phi] = d1
            d2_ring_map[rr, phi] = d2

    r_ring_of_r = np.arange(nR, dtype=np.int32) // nDOIPoss
    doi1_of_r = np.arange(nR, dtype=np.int32) % nDOI
    doi2_of_r = (np.arange(nR, dtype=np.int32) // nDOI) % nDOI

    d1_base = d1_ring_map[r_ring_of_r, :].T.astype(np.int64)
    d2_base = d2_ring_map[r_ring_of_r, :].T.astype(np.int64)

    doi_off1 = (doi1_of_r * doi_stride).astype(np.int64)
    doi_off2 = (doi2_of_r * doi_stride).astype(np.int64)

    # z1/z2/dz for all z-bins
    z1_map = np.zeros(nZ, dtype=np.int32)
    z2_map = np.zeros(nZ, dtype=np.int32)
    dz_map = np.zeros(nZ, dtype=np.int32)
    for z_bin in range(nZ):
        z1, z2 = histo.getZ1Z2(z_bin)
        z1_map[z_bin] = z1
        z2_map[z_bin] = z2
        dz_map[z_bin] = abs(z2 - z1)

    phi_parity = np.arange(nPhi, dtype=np.int64) % 2
    max_det = num_det - 1
    max_dz = s.maxRingDiff
    r_idx = np.arange(nR, dtype=np.int64)[np.newaxis, :]

    # geo_for_r[parity, dz, r]
    geo_for_r = np.zeros((2, max_dz + 1, nR), dtype=np.float64)
    for parity in [0, 1]:
        for dz in range(max_dz + 1):
            geo_for_r[parity, dz, :] = geo[dz, r_ring_of_r, parity,
                                             doi1_of_r, doi2_of_r]

    # ── 2. B: block profile ──
    if verbose:
        print(f'\n[2/6] Estimating block profile B...')

    if n_planes_b == 'all' or n_planes_b > nRings:
        n_planes_b = nRings

    n_blocks = bs * bs
    B_accum = np.zeros(n_blocks, dtype=np.float64)
    B_count = np.zeros(n_blocks, dtype=np.int64)

    with open(ref_norm_path, 'rb') as f:
        f.seek(HISTO_HEADER_SIZE)
        for z_bin in range(n_planes_b):
            ref = read_his_plane(f, nPhi, nR)
            if ref is None:
                break
            z1 = z1_map[z_bin]
            z2 = z2_map[z_bin]
            dz = dz_map[z_bin]

            d1 = d1_base + z1 * nDets + doi_off1[np.newaxis, :]
            d2 = d2_base + z2 * nDets + doi_off2[np.newaxis, :]
            np.clip(d1, 0, max_det, out=d1)
            np.clip(d2, 0, max_det, out=d2)

            geo_2d = geo_for_r[phi_parity, dz, :]
            ratio = ref / np.maximum(geo_2d, 1e-30)
            valid = (ref > min_count) & (geo_2d > 0)

            b1 = d1 % bs
            b2 = d2 % bs
            flat_b = (b1 * bs + b2).ravel()
            v = valid.ravel()
            B_accum += np.bincount(flat_b[v], weights=ratio.ravel()[v],
                                   minlength=n_blocks).astype(np.float64)
            B_count += np.bincount(flat_b[v],
                                   minlength=n_blocks).astype(np.int64)

    mask = B_count > 0
    B_flat = np.ones(n_blocks, dtype=np.float64)
    B_flat[mask] = B_accum[mask] / B_count[mask]
    B_flat /= B_flat[mask].mean()
    B = B_flat.reshape(bs, bs)

    if smooth_b_sigma > 0:
        from scipy.ndimage import gaussian_filter
        B_smooth = gaussian_filter(B, sigma=smooth_b_sigma, mode='reflect')
        B_smooth /= B_smooth.ravel()[mask].mean()
        B.ravel()[mask] = B_smooth.ravel()[mask]

    if verbose:
        n_used = n_planes_b if n_planes_b != 'all' else min(nRings, nZ)
        print(f'  B from {n_used} direct plane(s)')
        print(f'  B shape={B.shape}, range=[{B.min():.4f},{B.max():.4f}]')

    # ── 3. d: crystal interference ──
    if verbose:
        print(f'\n[3/6] Estimating crystal interference d...')
    n_blocks = bs * bs
    n_bins_d = nR * n_blocks
    d_accum = np.zeros(n_bins_d, dtype=np.float64)
    d_count = np.zeros(n_bins_d, dtype=np.int64)

    r_idx_2d = np.arange(nR, dtype=np.int64)[np.newaxis, :] * n_blocks

    with open(ref_norm_path, 'rb') as f:
        f.seek(HISTO_HEADER_SIZE)
        for z_bin in range(nRings):
            ref = read_his_plane(f, nPhi, nR)
            if ref is None:
                break
            z1 = z1_map[z_bin]
            z2 = z2_map[z_bin]
            dz = dz_map[z_bin]

            d1 = d1_base + z1 * nDets + doi_off1[np.newaxis, :]
            d2 = d2_base + z2 * nDets + doi_off2[np.newaxis, :]
            np.clip(d1, 0, max_det, out=d1)
            np.clip(d2, 0, max_det, out=d2)

            geo_2d = geo_for_r[phi_parity, dz, :]
            b1 = d1 % bs
            b2 = d2 % bs
            ratio = ref / np.maximum(geo_2d * B[b1, b2], 1e-30)
            valid = (ref > min_count) & (geo_2d > 0) & (B[b1, b2] > 0)

            flat_idx = r_idx_2d + b1 * bs + b2
            v = valid.ravel()
            fi = flat_idx.ravel()[v]
            rv = ratio.ravel()[v]
            d_accum += np.bincount(fi, weights=rv,
                                   minlength=n_bins_d).astype(np.float64)
            d_count += np.bincount(fi,
                                   minlength=n_bins_d).astype(np.int64)
            if z_bin % 50 == 0 and verbose:
                print(f'  d: z_bin {z_bin}/{nRings}', end='\r')

    mask_d = d_count > 0
    d_flat = np.ones(n_bins_d, dtype=np.float64)
    d_flat[mask_d] = d_accum[mask_d] / d_count[mask_d]
    d = d_flat.reshape(nR, bs, bs).transpose(1, 2, 0).copy()
    d_mean = d_flat[mask_d].mean()
    d /= d_mean
    if verbose:
        print(f'  d shape={d.shape}, range=[{d.min():.4f},{d.max():.4f}]')

    # ── 4. A(z): plane efficiency ──
    if verbose:
        print(f'\n[4/6] Estimating plane efficiency A(z)...')
    A_accum = np.zeros(nZ, dtype=np.float64)
    A_count = np.zeros(nZ, dtype=np.int64)

    with open(ref_norm_path, 'rb') as f:
        f.seek(HISTO_HEADER_SIZE)
        for z_bin in range(nZ):
            ref = read_his_plane(f, nPhi, nR)
            if ref is None:
                break
            z1 = z1_map[z_bin]
            z2 = z2_map[z_bin]
            dz = dz_map[z_bin]

            d1 = d1_base + z1 * nDets + doi_off1[np.newaxis, :]
            d2 = d2_base + z2 * nDets + doi_off2[np.newaxis, :]
            np.clip(d1, 0, max_det, out=d1)
            np.clip(d2, 0, max_det, out=d2)

            geo_2d = geo_for_r[phi_parity, dz, :]
            b1 = d1 % bs
            b2 = d2 % bs
            gbd = geo_2d * B[b1, b2] * d[b1, b2, r_idx]
            valid = (ref > min_count) & (gbd > 0)
            if valid.sum() > 0:
                A_accum[z_bin] = np.mean(ref[valid] / gbd[valid])
                A_count[z_bin] = valid.sum()
            if z_bin % 500 == 0 and verbose:
                print(f'  A: z_bin {z_bin}/{nZ}', end='\r')

    if verbose:
        print(f'  A: z_bin {nZ-1}/{nZ}')
    valid_z = A_count > 0
    A = np.ones(nZ, dtype=np.float64)
    A[valid_z] = A_accum[valid_z]
    A /= A[valid_z].mean()
    if verbose:
        print(f'  A shape={A.shape}, range=[{A[valid_z].min():.4f},'
              f'{A[valid_z].max():.4f}]')

    # ── 5. eps: crystal efficiency via fan-sum ──
    if verbose:
        nz_used = (nZ + fan_sum_stride - 1) // fan_sum_stride
        print(f'\n[5/6] Estimating epsilon via fan-sum '
              f'({fan_sum_iters} iterations x {nz_used} z-planes,'
              f' stride={fan_sum_stride})...')
    log_eps = np.zeros(num_det, dtype=np.float64)

    for iteration in range(fan_sum_iters):
        accum = np.zeros(num_det, dtype=np.float64)
        cnt = np.zeros(num_det, dtype=np.int64)

        nz_used = (nZ + fan_sum_stride - 1) // fan_sum_stride
        with open(ref_norm_path, 'rb') as f:
            for idx, z_bin in enumerate(range(0, nZ, fan_sum_stride)):
                f.seek(HISTO_HEADER_SIZE + z_bin * plane_bytes)
                ref = read_his_plane(f, nPhi, nR)
                if ref is None:
                    break
                z1 = z1_map[z_bin]
                z2 = z2_map[z_bin]
                dz = dz_map[z_bin]

                d1 = d1_base + z1 * nDets + doi_off1[np.newaxis, :]
                d2 = d2_base + z2 * nDets + doi_off2[np.newaxis, :]
                np.clip(d1, 0, max_det, out=d1)
                np.clip(d2, 0, max_det, out=d2)

                geo_2d = geo_for_r[phi_parity, dz, :]
                b1 = d1 % bs
                b2 = d2 % bs
                gbd = geo_2d * B[b1, b2] * d[b1, b2, r_idx]

                log_res = (np.log(np.maximum(ref, 1e-30))
                           - np.log(np.maximum(gbd, 1e-30))
                           - np.log(np.maximum(A[z_bin], 1e-30)))
                valid = (ref > min_count) & (gbd > 0) & (A[z_bin] > 0)

                d1_v = d1[valid]
                d2_v = d2[valid]
                log_res_v = log_res[valid]

                cnt += np.bincount(d1_v, minlength=num_det).astype(np.int64)
                cnt += np.bincount(d2_v, minlength=num_det).astype(np.int64)
                accum += np.bincount(
                    d1_v, weights=log_res_v - log_eps[d2_v],
                    minlength=num_det).astype(np.float64)
                accum += np.bincount(
                    d2_v, weights=log_res_v - log_eps[d1_v],
                    minlength=num_det).astype(np.float64)

                if idx % 100 == 0 and verbose:
                    print(f'  iter {iteration+1}/{fan_sum_iters}, '
                          f'z_bin {z_bin}/{nZ} ({idx}/{nz_used})', end='\r')

        mask_e = cnt > 0
        log_eps[mask_e] = accum[mask_e] / cnt[mask_e]
        eps = np.exp(log_eps)
        eps_mean = eps[mask_e].mean()
        log_eps -= np.log(eps_mean)
        eps = np.exp(log_eps)
        if verbose:
            print(f'  iter {iteration+1}: eps mean={eps[mask_e].mean():.4f}, '
                  f'std={eps[mask_e].std():.4f}, '
                  f'cnt_avg={cnt[mask_e].mean():.0f}')

    eps = np.exp(log_eps)
    if verbose:
        print(f'  Final eps: mean={eps.mean():.4f}, '
              f'std={eps.std():.4f}')

    # ── 6. Scale ──
    if verbose:
        print(f'\n[6/6] Computing global scale...')
    sum_ref = 0.0
    sum_pred = 0.0
    with open(ref_norm_path, 'rb') as f:
        f.seek(HISTO_HEADER_SIZE)
        ref0 = read_his_plane(f, nPhi, nR)
        z1 = z1_map[0]
        z2 = z2_map[0]
        dz = dz_map[0]
        d1 = d1_base + z1 * nDets + doi_off1[np.newaxis, :]
        d2 = d2_base + z2 * nDets + doi_off2[np.newaxis, :]
        np.clip(d1, 0, max_det, out=d1)
        np.clip(d2, 0, max_det, out=d2)
        geo_2d = geo_for_r[phi_parity, dz, :]
        b1 = d1 % bs
        b2 = d2 % bs
        pred = (A[0] * geo_2d * B[b1, b2] * d[b1, b2, r_idx]
                * eps[d1] * eps[d2])
        valid = (ref0 > min_count) & (pred > 0)
        if valid.sum() > 0:
            scale = float(ref0[valid].sum()) / float(pred[valid].sum())
        else:
            scale = 1.0

    # Fallback: scale from all direct planes
    if scale <= 0 or not np.isfinite(scale):
        if verbose:
            print('  Single-plane scale failed, using all direct planes...')
        sum_r = 0.0
        sum_p = 0.0
        with open(ref_norm_path, 'rb') as f:
            f.seek(HISTO_HEADER_SIZE)
            for z_bin in range(nRings):
                ref = read_his_plane(f, nPhi, nR)
                if ref is None:
                    break
                z1 = z1_map[z_bin]
                z2 = z2_map[z_bin]
                dz = dz_map[z_bin]
                d1 = d1_base + z1 * nDets + doi_off1[np.newaxis, :]
                d2 = d2_base + z2 * nDets + doi_off2[np.newaxis, :]
                np.clip(d1, 0, max_det, out=d1)
                np.clip(d2, 0, max_det, out=d2)
                geo_2d = geo_for_r[phi_parity, dz, :]
                b1 = d1 % bs
                b2 = d2 % bs
                pred = (A[z_bin] * geo_2d * B[b1, b2] * d[b1, b2, r_idx]
                        * eps[d1] * eps[d2])
                valid = (ref > min_count) & (pred > 0)
                sum_r += ref[valid].sum()
                sum_p += pred[valid].sum()
        scale = float(sum_r) / float(sum_p) if sum_p > 0 else 1.0
    if verbose:
        print(f'  Scale = {scale:.6f}')

    # ── Package components ──
    comps = {
        'geo_lut': geo,
        'block': B,
        'd_pattern': d,
        'A': A,
        'eps': eps,
        'scale': np.float64(scale),
    }

    # ── Save (RWD format for arrays, plain text for scale) ──
    os.makedirs(output_dir, exist_ok=True)
    save_rwd(os.path.join(output_dir, 'geo_lut.rwd'),
             geo.astype(np.float32))
    save_rwd(os.path.join(output_dir, 'block_profile.rwd'),
             B.astype(np.float32))
    # Transpose to (nR, bs, bs) matching C++ Array::writeToFile convention
    save_rwd(os.path.join(output_dir, 'd_pattern.rwd'),
             d.transpose(2, 0, 1).astype(np.float32))
    save_rwd(os.path.join(output_dir, 'plane_eff_A.rwd'),
             A.astype(np.float32))
    save_rwd(os.path.join(output_dir, 'eps.rwd'),
             eps.astype(np.float32))
    with open(os.path.join(output_dir, 'scale.txt'), 'w') as f:
        f.write(f'{scale:.6f}\n')
    elapsed = time.time() - t_start
    if verbose:
        print(f'\n{"=" * 60}')
        print(f'Done in {elapsed:.1f}s')
        print(f'Components saved to {output_dir}/')
        print(f'  geo_lut.rwd        {geo.shape}')
        print(f'  block_profile.rwd  {B.shape}')
        nR_save = d.shape[2]
        print(f'  d_pattern.rwd      ({nR_save}, {d.shape[0]}, {d.shape[1]})')
        print(f'  plane_eff_A.rwd    {A.shape}')
        print(f'  eps.rwd            {eps.shape}')
        print(f'  scale.txt          scalar = {scale:.6f}')
        print(f'{"=" * 60}')

    return comps


def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Estimate PET norm components from reference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('scanner_json', help='Scanner config .json')
    p.add_argument('ref_norm', help='Reference norm .his')
    p.add_argument('output_dir', help='Output directory for components')
    p.add_argument('--block-size', type=int, default=None,
                   help='Detectors per block (default: from scanner)')
    p.add_argument('--fan-sum-iters', type=int, default=5,
                   help='Fan-sum iterations for epsilon')
    p.add_argument('--fan-sum-stride', type=int, default=1,
                   help='Subsample z-bins for fan-sum (1=all, 4=every 4th)')
    p.add_argument('--min-count', type=float, default=0,
                   help='Min ref count to include LOR (for noisy data)')
    p.add_argument('--n-planes-b', default='all',
                   help='Direct planes to avg for B (int or "all")')
    p.add_argument('--smooth-b', type=float, default=0.0,
                   help='Gaussian sigma for B smoothing (0=off)')
    args = p.parse_args()

    if args.n_planes_b != 'all':
        n_planes_b = int(args.n_planes_b)
    else:
        n_planes_b = NRING_AVG

    estimate_components(
        args.scanner_json,
        args.ref_norm,
        args.output_dir,
        block_size=args.block_size,
        fan_sum_iters=args.fan_sum_iters,
        fan_sum_stride=args.fan_sum_stride,
        min_count=args.min_count,
        n_planes_b=n_planes_b,
        smooth_b_sigma=args.smooth_b,
    )


if __name__ == '__main__':
    main()
