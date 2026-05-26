#!/usr/bin/env python3
"""Real-data GE CPU-vs-Metal OSEM driver for the experimental projector path."""

import argparse
import copy
import csv
import os
import time

import numpy as np
import pyyrtpet as yrt


def zero_outside_largest_fitting_circle(image, radius_reduction=0):
    x_size = image.shape[-1]
    y_size = image.shape[-2]
    radius = max(0.0, min(y_size, x_size) / 2.0 - radius_reduction)
    cx = x_size / 2.0
    cy = y_size / 2.0
    y_indices = np.arange(y_size, dtype=float)
    x_indices = np.arange(x_size, dtype=float)
    y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing="ij")
    mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= radius**2
    return np.where(mask[None, ...], image, 0.0)


def create_psf_kernel_image_space(
    img_params, save_path, fwhm_x=3.6, fwhm_y=3.6, fwhm_z=4.0, kernel_size=11
):
    sig_x = fwhm_x / 2.35482
    sig_y = fwhm_y / 2.35482
    sig_z = fwhm_z / 2.35482
    sig_x_vox = sig_x / (img_params.length_x / img_params.nx)
    sig_y_vox = sig_y / (img_params.length_y / img_params.ny)
    sig_z_vox = sig_z / (img_params.length_z / img_params.nz)

    half = kernel_size // 2
    coords = np.arange(-half, half + 1)
    hx = np.exp(-(coords**2) / (2.0 * sig_x_vox**2))
    hy = np.exp(-(coords**2) / (2.0 * sig_y_vox**2))
    hz = np.exp(-(coords**2) / (2.0 * sig_z_vox**2))
    hx /= hx.sum()
    hy /= hy.sum()
    hz /= hz.sum()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savetxt(save_path, np.vstack([hx, hy, hz]), delimiter=",", fmt="%.16f")
    with open(save_path, mode="ab") as handle:
        np.savetxt(
            handle,
            np.array([[kernel_size, kernel_size, kernel_size]]),
            delimiter=",",
            fmt="%d",
        )


def convert_ge_histo_to_histogram3d(scanner, histo):
    histo3d = yrt.Histogram3DOwned(scanner)
    histo3d.allocate()
    histo3d.clearProjections(0.0)
    yrt.convertToHistogram3D(histo, histo3d)
    return histo3d


def make_ge_histo(scanner, config_dir, data_dir, filename):
    return yrt.GE.GEHistoOwned(
        scanner,
        os.path.join(config_dir, "phi_u_to_dethigh.txt"),
        os.path.join(config_dir, "phi_u_to_detlow.txt"),
        os.path.join(config_dir, "theta_to_ring_pair.txt"),
        os.path.join(data_dir, filename),
    )


def load_histos(scanner, config_dir, data_dir):
    histo_acf = convert_ge_histo_to_histogram3d(
        scanner, make_ge_histo(scanner, config_dir, data_dir, "acf_272_415_1981_YRT.dat")
    )
    histo_norm = convert_ge_histo_to_histogram3d(
        scanner, make_ge_histo(scanner, config_dir, data_dir, "norm_272_415_1981_YRT.dat")
    )

    acf_np = np.array(histo_acf, copy=False)
    norm_np = np.array(histo_norm, copy=False)
    corr_np = np.zeros_like(acf_np, dtype=np.float32)
    np.divide(acf_np, norm_np, out=corr_np, where=norm_np > 0)
    corr_np = np.require(corr_np, dtype=np.float32, requirements=["C_CONTIGUOUS"])

    histo_corr = yrt.Histogram3DAlias(scanner)
    histo_corr.bind(corr_np)

    histo_scatter = convert_ge_histo_to_histogram3d(
        scanner,
        make_ge_histo(scanner, config_dir, data_dir, "scatter_272_415_1981_YRT.dat"),
    )
    histo_randoms = convert_ge_histo_to_histogram3d(
        scanner,
        make_ge_histo(scanner, config_dir, data_dir, "randoms_272_415_1981_YRT.dat"),
    )
    return histo_corr, histo_scatter, histo_randoms, corr_np, histo_acf, histo_norm


def scale_and_bind_histogram3d(scanner, source, scale):
    scaled = np.array(source, dtype=np.float32, copy=True) * np.float32(scale)
    scaled = np.require(scaled, dtype=np.float32, requirements=["C_CONTIGUOUS"])
    alias = yrt.Histogram3DAlias(scanner)
    alias.bind(scaled)
    return alias, scaled


def load_listmode_lut_alias(scanner, listmode_path, event_count):
    fields_per_event = 3
    raw = np.fromfile(
        listmode_path,
        dtype=np.uint32,
        count=event_count * fields_per_event,
    )
    if raw.size != event_count * fields_per_event:
        raise RuntimeError(
            f"Expected {event_count} events from {listmode_path}, got "
            f"{raw.size // fields_per_event}"
        )
    events = raw.reshape(-1, fields_per_event)
    timestamps = np.ascontiguousarray(events[:, 0])
    detector1 = np.ascontiguousarray(events[:, 1])
    detector2 = np.ascontiguousarray(events[:, 2])

    dataset = yrt.ListModeLUTAlias(scanner)
    dataset.bind(timestamps, detector1, detector2)
    return dataset, timestamps, detector1, detector2


def resolve_event_count(listmode_path, pct, max_events):
    fields_per_event = 3
    total_events = os.path.getsize(listmode_path) // (4 * fields_per_event)
    used_events = total_events if pct >= 100.0 else int(round(total_events * pct / 100.0))
    if max_events > 0:
        used_events = min(used_events, max_events)
    if used_events <= 0:
        raise RuntimeError("No events selected")
    return total_events, used_events


def image_stats(image):
    values = np.array(image, copy=False)
    return {
        "sum": float(np.sum(values, dtype=np.float64)),
        "max": float(np.max(values)),
        "nonzero": int(np.count_nonzero(values)),
    }


def index_label(index):
    return "x".join(str(int(value)) for value in index)


def compare_images(cpu_image, metal_image, abs_tol, rel_tol, top_k):
    cpu = np.asarray(cpu_image, dtype=np.float64)
    metal = np.asarray(metal_image, dtype=np.float64)
    diff = metal - cpu
    abs_diff = np.abs(diff)
    scale = np.maximum(1.0, np.abs(cpu))
    mismatches = np.sum(abs_diff > (abs_tol + rel_tol * scale))
    total = int(cpu.size)
    flat_abs_diff = abs_diff.reshape(-1)
    flat_cpu = cpu.reshape(-1)
    flat_metal = metal.reshape(-1)
    flat_diff = diff.reshape(-1)
    max_abs_flat_index = int(np.argmax(flat_abs_diff)) if total > 0 else 0
    max_abs_index = np.unravel_index(max_abs_flat_index, cpu.shape)
    rmse = float(np.sqrt(np.mean(diff * diff))) if total > 0 else 0.0
    ref_rms = float(np.sqrt(np.mean(cpu * cpu))) if total > 0 else 0.0
    diff_norm = float(np.linalg.norm(flat_diff))
    ref_norm = float(np.linalg.norm(flat_cpu))
    eps = 1.0e-12
    top = []
    if total > 0 and top_k > 0:
        selected = min(int(top_k), total)
        if selected == total:
            top_indices = np.argsort(flat_abs_diff)[::-1]
        else:
            top_indices = np.argpartition(flat_abs_diff, -selected)[-selected:]
            top_indices = top_indices[np.argsort(flat_abs_diff[top_indices])[::-1]]
        flat_scale = scale.reshape(-1)
        for rank, flat_index in enumerate(top_indices, start=1):
            index = np.unravel_index(int(flat_index), cpu.shape)
            top.append(
                {
                    "rank": rank,
                    "index": index_label(index),
                    "cpu": float(flat_cpu[flat_index]),
                    "metal": float(flat_metal[flat_index]),
                    "diff": float(flat_diff[flat_index]),
                    "abs_diff": float(flat_abs_diff[flat_index]),
                    "rel_diff": float(flat_abs_diff[flat_index] / flat_scale[flat_index]),
                }
            )
    return {
        "max_abs_diff": float(np.max(abs_diff)) if total > 0 else 0.0,
        "max_rel_diff": float(np.max(abs_diff / scale)) if total > 0 else 0.0,
        "mismatches": int(mismatches),
        "mismatch_fraction": float(mismatches) / float(total) if total > 0 else 0.0,
        "rmse": rmse,
        "nrmse": rmse / max(ref_rms, eps),
        "mae": float(np.mean(abs_diff)) if total > 0 else 0.0,
        "rel_l2": diff_norm / max(ref_norm, eps),
        "sum_diff": float(np.sum(metal, dtype=np.float64) - np.sum(cpu, dtype=np.float64)),
        "p50_abs_diff": float(np.percentile(abs_diff, 50.0)) if total > 0 else 0.0,
        "p95_abs_diff": float(np.percentile(abs_diff, 95.0)) if total > 0 else 0.0,
        "p99_abs_diff": float(np.percentile(abs_diff, 99.0)) if total > 0 else 0.0,
        "p999_abs_diff": float(np.percentile(abs_diff, 99.9)) if total > 0 else 0.0,
        "max_abs_index": index_label(max_abs_index),
        "max_abs_cpu": float(flat_cpu[max_abs_flat_index]) if total > 0 else 0.0,
        "max_abs_metal": float(flat_metal[max_abs_flat_index]) if total > 0 else 0.0,
        "max_abs_signed_diff": float(flat_diff[max_abs_flat_index]) if total > 0 else 0.0,
        "top_diffs": top,
    }


def print_diff_diagnostics(diff_stats):
    print(
        "rmse,nrmse,rel_l2,mae,p95_abs_diff,p99_abs_diff,p999_abs_diff,"
        "sum_diff,mismatch_fraction,max_abs_index,max_abs_cpu,max_abs_metal,"
        "max_abs_signed_diff"
    )
    print(
        f"{diff_stats['rmse']:.9g},{diff_stats['nrmse']:.9g},"
        f"{diff_stats['rel_l2']:.9g},{diff_stats['mae']:.9g},"
        f"{diff_stats['p95_abs_diff']:.9g},{diff_stats['p99_abs_diff']:.9g},"
        f"{diff_stats['p999_abs_diff']:.9g},{diff_stats['sum_diff']:.9g},"
        f"{diff_stats['mismatch_fraction']:.9g},"
        f"{diff_stats['max_abs_index']},{diff_stats['max_abs_cpu']:.9g},"
        f"{diff_stats['max_abs_metal']:.9g},"
        f"{diff_stats['max_abs_signed_diff']:.9g}"
    )
    if not diff_stats["top_diffs"]:
        return
    print("top_abs_diff_rank,index,cpu,metal,diff,abs_diff,rel_diff")
    for entry in diff_stats["top_diffs"]:
        print(
            f"{entry['rank']},{entry['index']},{entry['cpu']:.9g},"
            f"{entry['metal']:.9g},{entry['diff']:.9g},"
            f"{entry['abs_diff']:.9g},{entry['rel_diff']:.9g}"
        )


def parse_positive_int_list(value, option_name):
    if not value:
        return []
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = int(item)
        except ValueError as exc:
            raise SystemExit(f"{option_name} expects comma-separated integers") from exc
        if number <= 0:
            raise SystemExit(f"{option_name} values must be positive")
        parsed.append(number)
    if not parsed:
        raise SystemExit(f"{option_name} did not contain any values")
    return parsed


def format_field(value):
    if isinstance(value, float):
        return f"{value:.9g}"
    if value is None:
        return ""
    return str(value)


def print_sweep_summary(rows):
    fields = [
        "used_events",
        "iterations",
        "subsets",
        "setup_s",
        "cpu_recon_s",
        "metal_recon_s",
        "metal_over_cpu",
        "max_abs_diff",
        "max_rel_diff",
        "rmse",
        "nrmse",
        "rel_l2",
        "mismatches",
        "mismatch_fraction",
        "metal_projector_ran",
    ]
    print("sweep_summary")
    print(",".join(fields))
    for row in rows:
        print(",".join(format_field(row.get(field)) for field in fields))


def write_summary_csv(path, rows):
    if not rows:
        return
    preferred_fields = [
        "case",
        "total_events",
        "used_events",
        "iterations",
        "subsets",
        "setup_s",
        "cpu_recon_s",
        "metal_recon_s",
        "metal_over_cpu",
        "cpu_image_sum",
        "cpu_image_max",
        "cpu_image_nonzero",
        "metal_image_sum",
        "metal_image_max",
        "metal_image_nonzero",
        "max_abs_diff",
        "max_rel_diff",
        "rmse",
        "nrmse",
        "mae",
        "rel_l2",
        "sum_diff",
        "p50_abs_diff",
        "p95_abs_diff",
        "p99_abs_diff",
        "p999_abs_diff",
        "mismatches",
        "mismatch_fraction",
        "max_abs_index",
        "max_abs_cpu",
        "max_abs_metal",
        "max_abs_signed_diff",
        "metal_projector_ran",
    ]
    field_set = set()
    for row in rows:
        field_set.update(row.keys())
    fields = [field for field in preferred_fields if field in field_set]
    fields.extend(sorted(field_set.difference(fields)))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_sensitivity(
    scanner,
    img_params,
    histo_corr,
    dataset,
    lor_motion,
    psf_csv_path,
    use_motion,
    use_psf,
    global_scale_factor,
):
    sens_img = yrt.ImageOwned(img_params)
    sens_img.allocate()

    bin_iter = histo_corr.getBinIter(1, 0)
    proj_params = yrt.ProjectorParams(scanner)
    proj_params.setProjector("Siddon")
    operator_siddon = yrt.createOperatorProjector(proj_params, bin_iter)
    print("Computing sensitivity map with Siddon...")
    operator_siddon.applyAH(histo_corr, sens_img)

    if use_psf:
        print("Applying image PSF to sensitivity map...")
        yrt.OperatorPsf(psf_csv_path).applyA(sens_img, sens_img)

    if use_motion:
        print("Moving sensitivity image...")
        sens_img = yrt.timeAverageMoveImage(
            lor_motion,
            sens_img,
            int(dataset.getTimestamp(0)),
            int(dataset.getTimestamp(dataset.count() - 1)),
        )

    sens_np = np.array(sens_img, copy=False)
    sens_np = zero_outside_largest_fitting_circle(sens_np)
    sens_np = np.require(sens_np, dtype=np.float32, requirements=["C_CONTIGUOUS"])
    sens_alias = yrt.ImageAlias(img_params)
    sens_alias.bind(sens_np)
    sens_alias.multWithScalar(global_scale_factor)
    return sens_alias, sens_np


def run_osem(
    scanner,
    dataset,
    img_params,
    sensitivity,
    randoms,
    scatter,
    psf_csv_path,
    args,
    use_metal,
):
    recon = yrt.createOSEM(scanner, False)
    recon.setRandomsHistogram(randoms)
    recon.setScatterHistogram(scatter)
    recon.setProjector("S")
    recon.num_MLEM_iterations = args.iterations
    recon.num_OSEM_subsets = args.subsets
    recon.setDataInput(dataset)
    recon.setImageParams(img_params)
    if args.psf:
        recon.addImagePSF(psf_csv_path)
    recon.setSensitivityImages([sensitivity])
    recon.setGlobalScalingFactor(args.global_scale_factor)
    if use_metal:
        if not hasattr(recon, "setExperimentalMetalProjectorEnabled"):
            raise RuntimeError("This pyyrtpet build does not expose the Metal OSEM opt-in")
        recon.setExperimentalMetalProjectorEnabled(True)

    start = time.perf_counter()
    image = recon.reconstruct()
    elapsed = time.perf_counter() - start
    return recon, image, elapsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-data mini-hot-spot OSEM, optionally CPU-vs-Metal."
    )
    parser.add_argument("--base", default="/Users/yanischemli/Desktop/mini_hot_spot")
    parser.add_argument("--image-params", default="img_param_0.8mm.json")
    parser.add_argument("--pct", type=float, default=10.0)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--subsets", type=int, default=17)
    parser.add_argument("--global-scale-factor", type=float, default=1.0 / 2.3e5)
    parser.add_argument("--motion", dest="motion", action="store_true", default=True)
    parser.add_argument("--no-motion", dest="motion", action="store_false")
    parser.add_argument("--psf", dest="psf", action="store_true", default=True)
    parser.add_argument("--no-psf", dest="psf", action="store_false")
    parser.add_argument("--compare-metal", action="store_true")
    parser.add_argument(
        "--metal-event-limit",
        type=int,
        default=4096,
        help=(
            "Safety limit for --compare-metal. The current Metal backprojector "
            "is experimental and should be scaled gradually."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-metal",
        action="store_true",
        help="Bypass Metal safety limits for intentional repo-side stress tests.",
    )
    parser.add_argument("--atol", type=float, default=1.0e-2)
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=5,
        help="Print this many largest voxel differences for CPU-vs-Metal runs.",
    )
    parser.add_argument(
        "--sweep-events",
        default="",
        help="Comma-separated --max-events values to run in one process.",
    )
    parser.add_argument(
        "--sweep-iterations",
        default="",
        help="Comma-separated --iterations values to run in one process.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional CSV path for single-run or sweep summary rows.",
    )
    parser.add_argument(
        "--no-write-images",
        action="store_true",
        help="Skip writing sensitivity/CPU/Metal NIfTI outputs.",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default="")
    return parser.parse_args()


def enforce_metal_safety(args, used_events):
    if not args.compare_metal:
        return
    if args.psf:
        raise SystemExit(
            "The current experimental Metal OSEM projector path cannot run with "
            "image PSF enabled. Re-run with --compare-metal --no-psf."
        )
    if args.allow_unsafe_metal:
        print(
            "WARNING: --allow-unsafe-metal bypasses safeguards for the "
            "experimental Metal backprojector. Scale this path gradually."
        )
        return
    if args.iterations != 1 or args.subsets != 1:
        raise SystemExit(
            "Refusing --compare-metal with more than one iteration/subset. The "
            "current Metal adjoint is experimental and should be scaled from "
            "single-iteration/single-subset repo-side tests first. Use "
            "--iterations 1 --subsets 1 for Metal smoke comparisons."
        )
    if used_events > args.metal_event_limit:
        raise SystemExit(
            f"Refusing --compare-metal for {used_events} events. The default "
            f"safety limit is {args.metal_event_limit}. Try --max-events "
            f"{args.metal_event_limit} --iterations 1 --subsets 1, or pass "
            "--allow-unsafe-metal only for intentional repo-side stress tests."
        )


def default_output_dir(args):
    return os.path.join(
        args.base,
        f"out_mini_hot_spot_{args.motion}_DD_False_PSF_{args.psf}",
    )


def run_case(args, out_dir, write_images, case_label="", emit_pass=True):
    config_dir = os.path.join(args.base, "GE_config")
    image_params_path = (
        args.image_params
        if os.path.isabs(args.image_params)
        else os.path.join(args.base, args.image_params)
    )
    os.makedirs(out_dir, exist_ok=True)

    scanner = yrt.Scanner(os.path.join(config_dir, "GE.json"))
    img_params = yrt.ImageParams(image_params_path)
    listmode_path = os.path.join(args.base, "PseudoListMode.yrt")
    total_events, used_events = resolve_event_count(
        listmode_path, args.pct, args.max_events
    )
    enforce_metal_safety(args, used_events)

    voxel_size = np.round(img_params.length_x / img_params.nx, 1)
    psf_csv_path = os.path.join(
        out_dir,
        f"psfKernelImageSpace_{voxel_size}mm_x_{img_params.nx}_y_"
        f"{img_params.ny}_z_{img_params.nz}.csv",
    )
    if args.psf:
        create_psf_kernel_image_space(img_params, psf_csv_path)

    if case_label:
        print(f"case={case_label}")
    print(f"total_events={total_events}")
    print(f"used_events={used_events}")
    print(f"iterations={args.iterations}")
    print(f"subsets={args.subsets}")
    print(f"motion={args.motion}")
    print(f"psf={args.psf}")
    print(f"compare_metal={args.compare_metal}")

    setup_start = time.perf_counter()
    histo_corr, histo_scatter, histo_randoms, corr_np, histo_acf, histo_norm = (
        load_histos(scanner, config_dir, args.base)
    )
    fraction = float(used_events) / float(total_events)
    randoms, randoms_np = scale_and_bind_histogram3d(scanner, histo_randoms, fraction)
    scatter, scatter_np = scale_and_bind_histogram3d(scanner, histo_scatter, fraction)
    dataset, timestamps, detector1, detector2 = load_listmode_lut_alias(
        scanner, listmode_path, used_events
    )

    lor_motion = None
    if args.motion:
        lor_motion = yrt.LORMotion(os.path.join(args.base, "Motion.yrt"))
        dataset.addLORMotion(lor_motion)

    sensitivity, sensitivity_np = build_sensitivity(
        scanner,
        img_params,
        histo_corr,
        dataset,
        lor_motion,
        psf_csv_path,
        args.motion,
        args.psf,
        args.global_scale_factor,
    )
    if write_images:
        sensitivity.writeToFile(os.path.join(out_dir, "sens_img.nii.gz"))
    setup_elapsed = time.perf_counter() - setup_start

    print("Starting CPU reconstruction...")
    cpu_recon, cpu_image, cpu_elapsed = run_osem(
        scanner,
        dataset,
        img_params,
        sensitivity,
        randoms,
        scatter,
        psf_csv_path,
        args,
        use_metal=False,
    )
    if cpu_recon.didLastExperimentalMetalProjectorRun():
        raise RuntimeError("CPU reference unexpectedly ran the Metal projector")
    if write_images:
        cpu_image.writeToFile(os.path.join(out_dir, "cpu_osem.nii.gz"))
    cpu_stats = image_stats(cpu_image)
    row = {
        "case": case_label,
        "total_events": total_events,
        "used_events": used_events,
        "iterations": args.iterations,
        "subsets": args.subsets,
        "setup_s": setup_elapsed,
        "cpu_recon_s": cpu_elapsed,
        "cpu_image_sum": cpu_stats["sum"],
        "cpu_image_max": cpu_stats["max"],
        "cpu_image_nonzero": cpu_stats["nonzero"],
    }

    print(
        "mode,setup_s,recon_s,metal_projector_ran,image_sum,image_max,"
        "image_nonzero"
    )
    print(
        f"cpu,{setup_elapsed:.6f},{cpu_elapsed:.6f},"
        f"{cpu_recon.didLastExperimentalMetalProjectorRun()},"
        f"{cpu_stats['sum']:.9g},{cpu_stats['max']:.9g},"
        f"{cpu_stats['nonzero']}"
    )

    keep_alive = (
        corr_np,
        randoms_np,
        scatter_np,
        timestamps,
        detector1,
        detector2,
        sensitivity_np,
        histo_acf,
        histo_norm,
        lor_motion,
    )
    _ = keep_alive

    if not args.compare_metal:
        if emit_pass:
            print("PASS")
        return row

    print("Starting experimental Metal reconstruction...")
    metal_recon, metal_image, metal_elapsed = run_osem(
        scanner,
        dataset,
        img_params,
        sensitivity,
        randoms,
        scatter,
        psf_csv_path,
        args,
        use_metal=True,
    )
    if write_images:
        metal_image.writeToFile(os.path.join(out_dir, "metal_osem.nii.gz"))
    metal_stats = image_stats(metal_image)
    diff_stats = compare_images(
        cpu_image,
        metal_image,
        args.atol,
        args.rtol,
        max(0, args.diagnostic_top_k),
    )
    metal_over_cpu = metal_elapsed / cpu_elapsed if cpu_elapsed > 0.0 else float("inf")
    row.update(
        {
            "metal_recon_s": metal_elapsed,
            "metal_over_cpu": metal_over_cpu,
            "metal_projector_ran": metal_recon.didLastExperimentalMetalProjectorRun(),
            "metal_image_sum": metal_stats["sum"],
            "metal_image_max": metal_stats["max"],
            "metal_image_nonzero": metal_stats["nonzero"],
        }
    )
    row.update(
        {
            key: value
            for key, value in diff_stats.items()
            if key != "top_diffs"
        }
    )

    print(
        f"metal,{setup_elapsed:.6f},{metal_elapsed:.6f},"
        f"{metal_recon.didLastExperimentalMetalProjectorRun()},"
        f"{metal_stats['sum']:.9g},{metal_stats['max']:.9g},"
        f"{metal_stats['nonzero']}"
    )
    print(
        "max_abs_diff,max_rel_diff,mismatches,metal_over_cpu\n"
        f"{diff_stats['max_abs_diff']:.9g},{diff_stats['max_rel_diff']:.9g},"
        f"{diff_stats['mismatches']},"
        f"{metal_over_cpu:.3f}"
    )
    print_diff_diagnostics(diff_stats)

    if not metal_recon.didLastExperimentalMetalProjectorRun():
        raise SystemExit("Metal projector path did not run")
    if diff_stats["mismatches"] != 0:
        print(
            "WARNING: CPU and Metal differ beyond the requested tolerance. "
            "Use --fail-on-mismatch to make this fatal."
        )
        if args.fail_on_mismatch:
            raise SystemExit(1)
    if emit_pass:
        print("PASS")
    return row


def main():
    args = parse_args()
    if args.threads > 0:
        yrt.setNumThreads(args.threads)

    sweep_events = parse_positive_int_list(args.sweep_events, "--sweep-events")
    sweep_iterations = parse_positive_int_list(
        args.sweep_iterations, "--sweep-iterations"
    )
    is_sweep = bool(sweep_events or sweep_iterations)
    event_values = sweep_events or [args.max_events]
    iteration_values = sweep_iterations or [args.iterations]
    rows = []

    if is_sweep:
        base_out_dir = args.out_dir or os.path.join(args.base, "out_metal_ge_osem_sweep")
        print(
            "sweep_plan="
            f"events:{','.join(str(value) for value in event_values)};"
            f"iterations:{','.join(str(value) for value in iteration_values)}"
        )
        for max_events in event_values:
            for iterations in iteration_values:
                case_args = copy.copy(args)
                case_args.max_events = max_events
                case_args.iterations = iterations
                case_label = (
                    f"events_{max_events}_iters_{iterations}_subsets_{args.subsets}"
                )
                case_out_dir = os.path.join(base_out_dir, case_label)
                rows.append(
                    run_case(
                        case_args,
                        case_out_dir,
                        write_images=not args.no_write_images,
                        case_label=case_label,
                        emit_pass=False,
                    )
                )
        print_sweep_summary(rows)
        if args.summary_csv:
            write_summary_csv(args.summary_csv, rows)
            print(f"summary_csv={args.summary_csv}")
        print("PASS")
        return

    out_dir = args.out_dir or default_output_dir(args)
    rows.append(
        run_case(
            args,
            out_dir,
            write_images=not args.no_write_images,
            emit_pass=True,
        )
    )
    if args.summary_csv:
        write_summary_csv(args.summary_csv, rows)
        print(f"summary_csv={args.summary_csv}")


if __name__ == "__main__":
    main()
