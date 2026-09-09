#!/usr/bin/env python3
"""Recompose full norm Histogram3D from components using pyyrtpet.

Usage:
    python recompose.py <output.his> --scanner <scanner.json>
        [--ref <ref.his>] [--comp-dir <dir>]

    python recompose.py <output.his> --scanner <scanner.json>
        --divide-by-eps <measured.his> [--comp-dir <dir>]

    python recompose.py --scanner <scanner.json> --diagnose <measured.his>
        [--comp-dir <dir>] [--singles <singles.npy>]

Loads components from a directory, recomposes the full 3D norm,
and writes to the given path.  With --divide-by-eps, instead divides
a measured norm by eps(d1)*eps(d2) to remove detector efficiency.
With --diagnose, runs diagnostic analysis of the norm components.
"""

import os, sys, json, struct, time
import numpy as np
import pyyrtpet as yrt

HISTO_HEADER_SIZE = 32
HISTO_MAGIC = 732174000


# ── .his I/O helpers ──────────────────────────────────────────

def write_his_header(f, numZ, numPhi, numR, version=3):
    hdr = struct.pack("<QQQQ",
                      HISTO_MAGIC | (version << 32),
                      numZ, numPhi, numR)
    f.write(hdr)


def read_his_plane(path, plane_idx, nphi, nr):
    offset = HISTO_HEADER_SIZE + plane_idx * nphi * nr * 4
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(nphi * nr * 4)
    return np.frombuffer(raw, dtype=np.float32).reshape(nphi, nr)


# ── RWD format I/O ────────────────────────────────────────────

def read_rwd(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('<i', f.read(4))[0]
        ndim = struct.unpack('<i', f.read(4))[0]
        shape = struct.unpack(f'<{ndim}Q', f.read(ndim * 8))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(shape)
    return data.copy()


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


# ── Scanner properties (derived) ───────────────────────────────

def scanner_num_doi_poss(scanner):
    return scanner.numDOI * scanner.numDOI

def scanner_num_r_ring(scanner):
    return scanner.detsPerRing // 2 + 1 - scanner.minAngDiff

def scanner_num_r(scanner):
    return scanner_num_doi_poss(scanner) * scanner_num_r_ring(scanner)

def scanner_num_z_bin(scanner):
    dz_max = scanner.maxRingDiff
    nz = (dz_max + 1) * scanner.numRings \
         - (dz_max * (dz_max + 1)) // 2
    nz_diff = nz - scanner.numRings
    return nz + nz_diff


# ── Load components ───────────────────────────────────────────

def load_components(comp_dir, scanner):
    comps = {}
    comps['scanner'] = scanner
    histo = yrt.Histogram3DAlias(scanner)

    nRings = scanner.numRings
    nDets = scanner.detsPerRing
    nDOI = scanner.numDOI
    nDOIPoss = scanner_num_doi_poss(scanner)
    nR_ring = scanner_num_r_ring(scanner)
    nR = scanner_num_r(scanner)
    nPhi = scanner.detsPerRing
    nZ = scanner_num_z_bin(scanner)

    comps['nRings'] = nRings
    comps['nDets'] = nDets
    comps['nDOI'] = nDOI
    comps['nDOIPoss'] = nDOIPoss
    comps['nR_ring'] = nR_ring
    comps['nR'] = nR
    comps['nPhi'] = nPhi
    comps['nZ'] = nZ
    comps['maxRingDiff'] = scanner.maxRingDiff
    comps['minAngDiff'] = scanner.minAngDiff
    comps['detsPerBlock'] = scanner.detsPerBlock

    # RWD format components
    comps['block'] = read_rwd(os.path.join(comp_dir, 'block_profile.rwd'))
    comps['d_pattern'] = np.transpose(
        read_rwd(os.path.join(comp_dir, 'd_pattern.rwd')),
        (1, 2, 0))
    comps['A'] = read_rwd(os.path.join(comp_dir, 'plane_eff_A.rwd')).ravel()
    comps['eps'] = read_rwd(os.path.join(comp_dir, 'eps.rwd')).ravel()
    try:
        with open(os.path.join(comp_dir, 'scale.txt')) as f:
            comps['scale'] = float(f.read().strip())
    except FileNotFoundError:
        comps['scale'] = 1.0

    # Geo LUT
    geo_rwd = os.path.join(comp_dir, 'geo_lut.rwd')
    if os.path.isfile(geo_rwd):
        comps['geo_lut'] = read_rwd(geo_rwd)
    else:
        comps['geo_lut'] = None

    return comps


# ── Build detector maps ────────────────────────────────────────

def build_maps(comps):
    scanner = comps['scanner']
    nRings = comps['nRings']
    nDets = comps['nDets']
    nPhi = comps['nPhi']
    nR = comps['nR']
    nR_ring = comps['nR_ring']
    nDOI = comps['nDOI']
    nDOIPoss = comps['nDOIPoss']
    doi_stride = nDets * nRings
    histo = yrt.Histogram3DAlias(scanner)

    # d1/d2 base indices: (nR_ring, nPhi)
    d1_ring_map = np.zeros((nR_ring, nPhi), dtype=np.int32)
    d2_ring_map = np.zeros((nR_ring, nPhi), dtype=np.int32)
    for rr in range(nR_ring):
        for phi in range(nPhi):
            d1, d2 = get_det_pair_in_same_ring(scanner, rr, phi)
            d1_ring_map[rr, phi] = d1
            d2_ring_map[rr, phi] = d2

    r_ring_of_r = np.arange(nR, dtype=np.int32) // nDOIPoss
    doi1_of_r = np.arange(nR, dtype=np.int32) % nDOI
    doi2_of_r = (np.arange(nR, dtype=np.int32) // nDOI) % nDOI

    d1_base_idx = d1_ring_map[r_ring_of_r, :].T.astype(np.int64)
    d2_base_idx = d2_ring_map[r_ring_of_r, :].T.astype(np.int64)

    # z1/z2/dz for all z_bins
    nZ = comps['nZ']
    z1_map = np.zeros(nZ, dtype=np.int32)
    z2_map = np.zeros(nZ, dtype=np.int32)
    dz_map = np.zeros(nZ, dtype=np.int32)
    for z_bin in range(nZ):
        z1, z2 = histo.getZ1Z2(z_bin)
        z1_map[z_bin] = z1
        z2_map[z_bin] = z2
        dz_map[z_bin] = abs(z2 - z1)

    doi_off1 = (doi1_of_r * doi_stride).astype(np.int64)
    doi_off2 = (doi2_of_r * doi_stride).astype(np.int64)

    return {
        'd1_base_idx': d1_base_idx,
        'd2_base_idx': d2_base_idx,
        'z1_map': z1_map,
        'z2_map': z2_map,
        'dz_map': dz_map,
        'doi_stride': doi_stride,
        'doi_off1': doi_off1,
        'doi_off2': doi_off2,
        'r_ring_of_r': r_ring_of_r,
        'doi1_of_r': doi1_of_r,
        'doi2_of_r': doi2_of_r,
        'nDOI': nDOI,
        'nDOIPoss': nDOIPoss,
    }


# ── Geometry factor from LUT ────────────────────────────────────

def compute_geo_lut(scanner):
    """Compute G from real detector positions using scanner.createLUT()."""
    lut = scanner.createLUT()
    pos = lut[:, :3]
    normv = lut[:, 3:]

    nDets = scanner.detsPerRing
    nRings = scanner.numRings
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
            for parity in range(2):
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


# ── Recomposition ──────────────────────────────────────────────

def recompose(comps, maps, output_path, ref_path=None, is_norm=False,
              chunk_z=32):
    s = comps['scanner']
    nZ = comps['nZ']
    nPhi = comps['nPhi']
    nR = comps['nR']
    nRings = comps['nRings']
    nDets = comps['nDets']
    nDOI = comps['nDOI']
    nDOIPoss = comps['nDOIPoss']
    bs = comps['detsPerBlock']
    maxRingDiff = comps['maxRingDiff']

    block = comps['block']
    d_pattern = comps['d_pattern']
    eps = comps['eps']
    A = comps['A']
    scale = comps['scale']

    d1_base = maps['d1_base_idx']
    d2_base = maps['d2_base_idx']
    z1_map = maps['z1_map']
    z2_map = maps['z2_map']
    dz_map = maps['dz_map']
    doi_off1 = maps['doi_off1']
    doi_off2 = maps['doi_off2']
    r_ring_of_r = maps['r_ring_of_r']
    doi1_of_r = maps['doi1_of_r']
    doi2_of_r = maps['doi2_of_r']

    # Precompute geo_for_r[parity, dz, r]
    geo_lut = comps['geo_lut']
    if geo_lut is None:
        print("Computing G LUT from scanner geometry...")
        geo_lut = compute_geo_lut(s)
    geo_for_r = np.zeros((2, maxRingDiff + 1, nR), dtype=np.float64)
    for parity in [0, 1]:
        for dz in range(maxRingDiff + 1):
            geo_for_r[parity, dz, :] = geo_lut[
                dz, r_ring_of_r, parity, doi1_of_r, doi2_of_r]

    phi_parity = np.arange(nPhi, dtype=np.int64) % 2
    max_idx = nDets * nRings * nDOI - 1

    # Scale from reference if requested
    if ref_path is not None and os.path.isfile(ref_path):
        print("Computing scale from direct plane (z_bin=0)...")
        ref = read_his_plane(ref_path, 0, nPhi, nR)
        dz = dz_map[0]
        z1 = z1_map[0]
        z2 = z2_map[0]
        d1 = d1_base + z1 * nDets + doi_off1[np.newaxis, :]
        d2 = d2_base + z2 * nDets + doi_off2[np.newaxis, :]
        np.clip(d1, 0, max_idx, out=d1)
        np.clip(d2, 0, max_idx, out=d2)
        geo = geo_for_r[phi_parity, dz, :]
        b1 = d1 % bs
        b2 = d2 % bs
        r_idx = np.arange(nR, dtype=np.int64)[np.newaxis, :]
        pred = (A[0] * geo * block[b1, b2] * d_pattern[b1, b2, r_idx]
                * eps[d1] * eps[d2])
        valid = (ref > 0) & (pred > 0)
        if valid.sum() > 0 and pred[valid].sum() > 0:
            if is_norm:
                scale = float((1.0 / ref[valid]).sum()) / float(pred[valid].sum())
            else:
                scale = float(ref[valid].sum()) / float(pred[valid].sum())
            print(f"  Computed scale = {scale:.6f}")
        else:
            print("  WARNING: no valid LORs, using stored scale")
    else:
        print(f"  Using stored scale = {scale:.6f}")

    print(f"Recomposing to {output_path}")
    print(f"  nz={nZ}, nphi={nPhi}, nr={nR}, scale={scale:.6f}")
    print(f"  Output is a sensitivity (multiply data by it)")

    t0 = time.time()
    with open(output_path, "wb") as f:
        write_his_header(f, nZ, nPhi, nR)
        for z_bin in range(0, nZ, chunk_z):
            z_end = min(z_bin + chunk_z, nZ)
            if z_bin % 2000 == 0:
                print(f"  z_bin {z_bin}/{nZ}")

            out_chunk = np.zeros(
                (z_end - z_bin, nPhi, nR), dtype=np.float32)

            for iz, z_idx in enumerate(range(z_bin, z_end)):
                z1 = z1_map[z_idx]
                z2 = z2_map[z_idx]
                dz = dz_map[z_idx]

                d1 = d1_base + z1 * nDets + doi_off1[np.newaxis, :]
                d2 = d2_base + z2 * nDets + doi_off2[np.newaxis, :]
                np.clip(d1, 0, max_idx, out=d1)
                np.clip(d2, 0, max_idx, out=d2)

                geo = geo_for_r[phi_parity, dz, :]
                b1 = d1 % bs
                b2 = d2 % bs
                r_idx = np.arange(nR, dtype=np.int64)[np.newaxis, :]

                norm_val = (A[z_idx] * geo * block[b1, b2]
                            * d_pattern[b1, b2, r_idx]
                            * eps[d1] * eps[d2])
                out_chunk[iz] = np.where(
                    norm_val > 0,
                    np.float32(scale * norm_val), 0.0)

            f.write(out_chunk.tobytes())

    elapsed = time.time() - t0
    file_size = os.path.getsize(output_path) / 1e9
    print(f"Done. {file_size:.1f} GB in {elapsed:.1f}s")
    return scale


# ── Divide measured norm by eps ───────────────────────────────

def divide_by_eps(measured_path, output_path, eps, maps, scanner,
                  chunk_z=32):
    """Divide a measured histogram3D by eps(d1)*eps(d2).

    Removes detector efficiency to retrieve geometry-only norm.
    Useful for diagnosing whether mismatches between measured and
    recomposed norms are explained by eps alone.
    """
    nRings = scanner.numRings
    nDets = scanner.detsPerRing
    nDOI = scanner.numDOI
    max_idx = nDets * nRings * nDOI - 1

    d1_base = maps['d1_base_idx']
    d2_base = maps['d2_base_idx']
    z1_map = maps['z1_map']
    z2_map = maps['z2_map']
    doi_off1 = maps['doi_off1']
    doi_off2 = maps['doi_off2']

    with open(measured_path, 'rb') as f:
        raw_hdr = f.read(HISTO_HEADER_SIZE)
        nZ_file = struct.unpack('<Q', raw_hdr[8:16])[0]
        nPhi_file = struct.unpack('<Q', raw_hdr[16:24])[0]
        nR_file = struct.unpack('<Q', raw_hdr[24:32])[0]

    nPhi = nPhi_file
    nR = nR_file

    print(f"Dividing measured norm by eps: {measured_path}")
    print(f"  nz={nZ_file}, nphi={nPhi_file}, nr={nR_file}")

    t0 = time.time()
    with open(output_path, 'wb') as f_out:
        write_his_header(f_out, nZ_file, nPhi_file, nR_file)
        with open(measured_path, 'rb') as f_in:
            f_in.seek(HISTO_HEADER_SIZE)
            for z_bin in range(0, nZ_file, chunk_z):
                z_end = min(z_bin + chunk_z, nZ_file)
                if z_bin % 2000 == 0:
                    print(f"  z_bin {z_bin}/{nZ_file}")

                chunk = np.zeros(
                    (z_end - z_bin, nPhi, nR), dtype=np.float32)
                for iz, z_idx in enumerate(range(z_bin, z_end)):
                    raw = f_in.read(nPhi * nR * 4)
                    if len(raw) < nPhi * nR * 4:
                        break
                    plane = np.frombuffer(
                        raw, dtype=np.float32).reshape(nPhi, nR)

                    z1 = z1_map[z_idx]
                    z2 = z2_map[z_idx]
                    d1 = d1_base + z1 * nDets + doi_off1[np.newaxis, :]
                    d2 = d2_base + z2 * nDets + doi_off2[np.newaxis, :]
                    np.clip(d1, 0, max_idx, out=d1)
                    np.clip(d2, 0, max_idx, out=d2)

                    eps_factor = eps[d1] * eps[d2]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        divided = np.where(
                            eps_factor > 0,
                            plane / eps_factor, 0.0)
                    chunk[iz] = divided.astype(np.float32)

                f_out.write(chunk.tobytes())

    elapsed = time.time() - t0
    file_size = os.path.getsize(output_path) / 1e6
    print(f"Done. {file_size:.1f} MB in {elapsed:.1f}s")


# ── Diagnostics ───────────────────────────────────────────────

def compute_pre_eps_residual(measured_path, comps, maps, chunk_z=32):
    """Compute R = measured / (G*B*d*A*scale) per LOR, then per-detector
    statistics of R.

    If the model is correct, R(d1,d2) = eps(d1)*eps(d2), so the mean
    of R over all LORs involving detector d is an independent estimate
    of eps(d) (up to a global constant).  Comparing this to the
    pipeline eps reveals whether eps is capturing real efficiency or
    compensating for model errors.

    Returns dict with:
      eps_independent : per-detector mean of R  (num_detectors,)
      cv              : coefficient of variation of R per detector
      count           : number of LORs contributing to each detector
      eps_pipeline    : the eps from the component file (for comparison)
    """
    scanner = comps['scanner']
    nRings = comps['nRings']
    nDets = comps['nDets']
    nDOI = comps['nDOI']
    nDOIPoss = comps['nDOIPoss']
    bs = comps['detsPerBlock']
    maxRingDiff = comps['maxRingDiff']
    num_det = nDets * nRings * nDOI
    max_idx = num_det - 1

    block = comps['block']
    d_pattern = comps['d_pattern']
    eps = comps['eps']
    A = comps['A']
    scale = comps['scale']

    d1_base = maps['d1_base_idx']
    d2_base = maps['d2_base_idx']
    z1_map = maps['z1_map']
    z2_map = maps['z2_map']
    doi_off1 = maps['doi_off1']
    doi_off2 = maps['doi_off2']
    r_ring_of_r = maps['r_ring_of_r']
    doi1_of_r = maps['doi1_of_r']
    doi2_of_r = maps['doi2_of_r']

    geo_lut = comps['geo_lut']
    if geo_lut is None:
        print("Computing G LUT from scanner geometry...")
        geo_lut = compute_geo_lut(scanner)
    geo_for_r = np.zeros((2, maxRingDiff + 1, len(r_ring_of_r)),
                         dtype=np.float64)
    for parity in [0, 1]:
        for dz in range(maxRingDiff + 1):
            geo_for_r[parity, dz, :] = geo_lut[
                dz, r_ring_of_r, parity, doi1_of_r, doi2_of_r]

    phi_parity = np.arange(maps['d1_base_idx'].shape[0],
                           dtype=np.int64) % 2
    nPhi = maps['d1_base_idx'].shape[0]
    nR = maps['d1_base_idx'].shape[1]

    with open(measured_path, 'rb') as f:
        raw_hdr = f.read(HISTO_HEADER_SIZE)
        nZ_file = struct.unpack('<Q', raw_hdr[8:16])[0]

    sum_R = np.zeros(num_det, dtype=np.float64)
    sum_R2 = np.zeros(num_det, dtype=np.float64)
    count = np.zeros(num_det, dtype=np.int64)

    print(f"Computing pre-eps residual: {measured_path}")
    t0 = time.time()
    with open(measured_path, 'rb') as f_in:
        f_in.seek(HISTO_HEADER_SIZE)
        for z_bin in range(0, nZ_file, chunk_z):
            z_end = min(z_bin + chunk_z, nZ_file)
            if z_bin % 2000 == 0:
                print(f"  z_bin {z_bin}/{nZ_file}")

            for z_idx in range(z_bin, z_end):
                raw = f_in.read(nPhi * nR * 4)
                if len(raw) < nPhi * nR * 4:
                    break
                measured = np.frombuffer(
                    raw, dtype=np.float32).reshape(nPhi, nR)

                z1 = z1_map[z_idx]
                z2 = z2_map[z_idx]
                dz = dz_map[z_idx]

                d1 = d1_base + z1 * nDets + doi_off1[np.newaxis, :]
                d2 = d2_base + z2 * nDets + doi_off2[np.newaxis, :]
                np.clip(d1, 0, max_idx, out=d1)
                np.clip(d2, 0, max_idx, out=d2)

                geo = geo_for_r[phi_parity, dz, :]
                b1 = d1 % bs
                b2 = d2 % bs
                r_idx = np.arange(nR, dtype=np.int64)[np.newaxis, :]

                model = (A[z_idx] * geo * block[b1, b2]
                         * d_pattern[b1, b2, r_idx]
                         * scale)
                valid = (measured > 0) & (model > 0)
                if not valid.any():
                    continue

                R = np.zeros_like(measured)
                R[valid] = measured[valid] / model[valid]

                d1_flat = d1.ravel()
                d2_flat = d2.ravel()
                R_flat = R.ravel()
                v = valid.ravel()

                sum_R += np.bincount(
                    d1_flat[v], weights=R_flat[v],
                    minlength=num_det).astype(np.float64)
                sum_R2 += np.bincount(
                    d1_flat[v], weights=(R_flat[v] ** 2),
                    minlength=num_det).astype(np.float64)
                count += np.bincount(
                    d1_flat[v], minlength=num_det).astype(np.int64)

                sum_R += np.bincount(
                    d2_flat[v], weights=R_flat[v],
                    minlength=num_det).astype(np.float64)
                sum_R2 += np.bincount(
                    d2_flat[v], weights=(R_flat[v] ** 2),
                    minlength=num_det).astype(np.float64)
                count += np.bincount(
                    d2_flat[v], minlength=num_det).astype(np.int64)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    has_data = count > 0
    eps_indep = np.zeros(num_det, dtype=np.float64)
    cv = np.full(num_det, np.nan, dtype=np.float64)
    eps_indep[has_data] = sum_R[has_data] / count[has_data]
    mean = eps_indep[has_data]
    var = sum_R2[has_data] / count[has_data] - mean ** 2
    cv[has_data] = np.sqrt(np.maximum(var, 0)) / np.maximum(mean, 1e-30)

    return {
        'eps_independent': eps_indep,
        'cv': cv,
        'count': count,
        'eps_pipeline': eps,
    }


def analyze_eps_structure(eps, scanner):
    """Examine the structure of the estimated eps.

    Returns dict with per-ring, per-DOI, per-block means and a list
    of dead detectors.
    """
    nRings = scanner.numRings
    nDets = scanner.detsPerRing
    nDOI = scanner.numDOI
    bs = scanner.detsPerBlock

    eps_3d = eps.reshape(nDOI, nRings, nDets)

    per_ring = eps_3d.mean(axis=(0, 2))
    per_doi = eps_3d.mean(axis=(1, 2))
    per_block = eps.reshape(nDOI, nRings // bs, bs,
                            nDets // bs, bs)
    per_block = per_block.mean(axis=(0, 2, 4))

    dead = np.where(eps < 1e-10)[0]
    low = np.where((eps > 0) & (eps < 0.1 * eps[eps > 0].mean()))[0]

    return {
        'per_ring': per_ring,
        'per_doi': per_doi,
        'per_block': per_block,
        'dead_detectors': dead,
        'low_detectors': low,
    }


def compare_eps_to_singles(eps, singles, scanner):
    """Compare estimated eps to independently measured singles rates.

    Both are normalized to mean=1 before comparison.
    singles should be shape (num_detectors,) with one value per
    detector (DOI-resolved: each DOI layer is a separate detector).

    Returns dict with correlation metrics.
    """
    from scipy.stats import pearsonr

    nRings = scanner.numRings
    nDets = scanner.detsPerRing
    nDOI = scanner.numDOI

    mask = (eps > 0) & (singles > 0)
    if mask.sum() < 10:
        print("WARNING: too few valid detectors for comparison")
        return {'correlation': 0, 'per_ring': None, 'per_doi': None}

    eps_norm = eps[mask] / eps[mask].mean()
    singles_norm = singles[mask] / singles[mask].mean()

    r, p = pearsonr(eps_norm, singles_norm)

    eps_3d = eps.reshape(nDOI, nRings, nDets)
    singles_3d = singles.reshape(nDOI, nRings, nDets)

    per_ring = np.zeros(nRings, dtype=np.float64)
    for ring in range(nRings):
        m = mask.reshape(nDOI, nRings, nDets)[:, ring, :].ravel()
        if m.sum() < 2:
            per_ring[ring] = np.nan
            continue
        e = eps_3d[:, ring, :].ravel()[m]
        s = singles_3d[:, ring, :].ravel()[m]
        per_ring[ring] = pearsonr(e / e.mean(), s / s.mean())[0]

    per_doi = np.zeros(nDOI, dtype=np.float64)
    for doi in range(nDOI):
        m = mask.reshape(nDOI, nRings, nDets)[doi, :, :].ravel()
        if m.sum() < 2:
            per_doi[doi] = np.nan
            continue
        e = eps_3d[doi, :, :].ravel()[m]
        s = singles_3d[doi, :, :].ravel()[m]
        per_doi[doi] = pearsonr(e / e.mean(), s / s.mean())[0]

    return {
        'correlation': r,
        'p_value': p,
        'per_ring': per_ring,
        'per_doi': per_doi,
    }


# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Recompose norm from components')
    p.add_argument('output', nargs='?', default=None,
                   help='Output .his file path')
    p.add_argument('--ref', default=None,
                   help='Reference .his for scale computation'
                        ' (overrides stored scale)')
    p.add_argument('--comp-dir', default=os.path.dirname(
        os.path.abspath(__file__)),
        help='Component directory')
    p.add_argument('--scanner', required=True,
                   help='Scanner config .json')
    p.add_argument('--divide-by-eps', default=None, metavar='MEASURED',
                   help='Divide measured .his by eps(d1)*eps(d2)'
                        ' instead of recomposing')
    p.add_argument('--diagnose', default=None, metavar='MEASURED',
                   help='Run diagnostic analysis on measured .his'
                        ' (requires output path for residual .his)')
    p.add_argument('--analyze-eps', action='store_true',
                   help='Analyze eps structure and print summary')
    p.add_argument('--singles', default=None, metavar='FILE',
                   help='Singles rates .npy file (one value per'
                        ' detector) for comparison with eps')
    p.add_argument('--is-norm', action='store_true',
                   help='Reference is a norm (1/sensitivity);'
                        ' invert when computing scale from --ref')
    args = p.parse_args()

    print("Loading components...")
    scanner = yrt.Scanner(args.scanner)
    comps = load_components(args.comp_dir, scanner)
    maps = build_maps(comps)

    if args.diagnose is not None:
        if args.output is None:
            p.error("--diagnose requires an output path for"
                    " the residual .his file")
        res = compute_pre_eps_residual(
            args.diagnose, comps, maps)
        eps_indep = res['eps_independent']
        eps_pipe = res['eps_pipeline']
        cv = res['cv']
        count = res['count']

        has = count > 0
        print(f"\nPre-eps residual diagnostic:")
        print(f"  Detectors with data: {has.sum()}/{len(count)}")
        print(f"  eps_pipeline:  mean={eps_pipe[has].mean():.4f}"
              f"  std={eps_pipe[has].std():.4f}"
              f"  range=[{eps_pipe[has].min():.4f}"
              f", {eps_pipe[has].max():.4f}]")
        print(f"  eps_independent: mean={eps_indep[has].mean():.4f}"
              f"  std={eps_indep[has].std():.4f}"
              f"  range=[{eps_indep[has].min():.4f}"
              f", {eps_indep[has].max():.4f}]")
        valid_cv = cv[has & ~np.isnan(cv)]
        print(f"  CV of R:  mean={valid_cv.mean():.4f}"
              f"  std={valid_cv.std():.4f}"
              f"  range=[{valid_cv.min():.4f}"
              f", {valid_cv.max():.4f}]")
        ratio = eps_indep[has] / np.maximum(eps_pipe[has], 1e-30)
        print(f"  eps_indep/eps_pipeline:"
              f" mean={ratio.mean():.4f}"
              f" std={ratio.std():.4f}")

        if args.singles is not None:
            singles = np.load(args.singles).ravel()
            comp = compare_eps_to_singles(
                eps_pipe, singles, scanner)
            print(f"\n  Singles correlation (pipeline eps):"
                  f" r={comp['correlation']:.4f}"
                  f"  p={comp['p_value']:.2e}")
            comp2 = compare_eps_to_singles(
                eps_indep, singles, scanner)
            print(f"  Singles correlation (independent):"
                  f" r={comp2['correlation']:.4f}"
                  f"  p={comp2['p_value']:.2e}")

    if args.analyze_eps:
        struct = analyze_eps_structure(comps['eps'], scanner)
        nDOI = scanner.numDOI
        nRings = scanner.numRings
        print(f"\nEps structure:")
        print(f"  Dead detectors: {len(struct['dead_detectors'])}")
        print(f"  Low detectors (<10% mean):"
              f" {len(struct['low_detectors'])}")
        print(f"  Per-DOI mean:")
        for doi in range(nDOI):
            print(f"    DOI {doi}: {struct['per_doi'][doi]:.4f}")
        print(f"  Per-ring mean (first 10):")
        for ring in range(min(10, nRings)):
            print(f"    Ring {ring}: {struct['per_ring'][ring]:.4f}")

    if args.divide_by_eps is not None:
        if args.output is None:
            p.error("--divide-by-eps requires an output path")
        divide_by_eps(args.divide_by_eps, args.output,
                      comps['eps'], maps, scanner)
    elif args.diagnose is None and not args.analyze_eps:
        if args.output is None:
            p.error("output path is required")
        scale = recompose(comps, maps, args.output, ref_path=args.ref,
                          is_norm=args.is_norm)
