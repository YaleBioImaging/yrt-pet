#!/usr/bin/env python3
"""Recompose full norm Histogram3D from components using pyyrtpet.

Usage:
    python recompose.py <output.his> [--ref <ref.his>] [--comp-dir <dir>]

Loads components from a directory, recomposes the full 3D norm,
and writes to the given path.
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

def recompose(comps, maps, output_path, ref_path=None, chunk_z=32):
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
            scale = float(ref[valid].sum()) / float(pred[valid].sum())
            print(f"  Computed scale = {scale:.6f}")
        else:
            print("  WARNING: no valid LORs, using stored scale")
    else:
        print(f"  Using stored scale = {scale:.6f}")

    print(f"Recomposing to {output_path}")
    print(f"  nz={nZ}, nphi={nPhi}, nr={nR}, scale={scale:.6f}")

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


# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Recompose norm from components')
    p.add_argument('output', help='Output .his file path')
    p.add_argument('--ref', default=None,
                   help='Reference .his for scale computation'
                        ' (overrides stored scale)')
    p.add_argument('--comp-dir', default=os.path.dirname(
        os.path.abspath(__file__)),
        help='Component directory')
    p.add_argument('--scanner', required=True,
                   help='Scanner config .json')
    args = p.parse_args()

    print("Loading components...")
    scanner = yrt.Scanner(args.scanner)
    comps = load_components(args.comp_dir, scanner)
    maps = build_maps(comps)

    scale = recompose(comps, maps, args.output, ref_path=args.ref)
