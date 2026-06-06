#!/usr/bin/env python3
"""Minimal GE mini-hot-spot reconstruction through the experimental Metal path."""

import argparse
import os
import subprocess
import time

import numpy as np
import pyyrtpet as yrt

from metal_ge_osem_smoke import (
    apply_metal_environment,
    create_psf_kernel_image_space,
    load_histos,
    scale_and_bind_histogram3d,
    zero_outside_largest_fitting_circle,
)
from metal_options import MetalRecipeOptions, apply_projector_runtime_options


FIELDS_PER_EVENT = 3
BYTES_PER_EVENT = FIELDS_PER_EVENT * 4
OWNED_READ_BUFFER_BYTES = (1 << 30) * 4
DEFAULT_BASE = "/Users/yanischemli/Desktop/mini_hot_spot"
DEFAULT_GLOBAL_SCALE_FACTOR = 1.0 / 2.3e5


def bytes_to_gib(value):
    return float(value) / (1024.0 * 1024.0 * 1024.0)


def system_memory_gib():
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and page_count > 0:
            return bytes_to_gib(page_size * page_count)
    except (AttributeError, OSError, ValueError):
        pass

    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bytes_to_gib(int(out.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def resolve_event_count(listmode_path, pct, max_events):
    total_events = os.path.getsize(listmode_path) // BYTES_PER_EVENT
    used_events = (
        total_events if pct >= 100.0 else int(round(total_events * pct / 100.0))
    )
    if max_events > 0:
        used_events = min(used_events, max_events)
    if used_events <= 0:
        raise RuntimeError("No events selected")
    return total_events, used_events


def resolve_listmode_loader(requested, total_events, used_events):
    if requested == "alias":
        return "alias"
    if requested == "owned":
        if used_events != total_events:
            raise RuntimeError("--listmode-loader owned requires the full listmode file")
        return "owned"
    return "owned" if used_events == total_events else "alias"


def listmode_peak_gib(loader, used_events):
    selected_bytes = used_events * BYTES_PER_EVENT
    if loader == "owned":
        return bytes_to_gib(
            selected_bytes + min(selected_bytes, OWNED_READ_BUFFER_BYTES)
        )
    return bytes_to_gib(selected_bytes * 2)


def resolve_cache_budget_mb(value, loader, used_events, headroom_gib, budget_fraction):
    if str(value).lower() != "auto":
        return float(value)

    total_ram_gib = system_memory_gib()
    peak_gib = listmode_peak_gib(loader, used_events)
    safe_cap_gib = max(0.0, total_ram_gib * budget_fraction)
    available_gib = max(0.0, total_ram_gib - peak_gib - headroom_gib)
    budget_gib = min(safe_cap_gib, available_gib)
    if safe_cap_gib > 0.0:
        budget_gib = max(1.0, budget_gib)
    return budget_gib * 1024.0


def load_listmode_dataset(scanner, listmode_path, loader, used_events):
    if loader == "owned":
        return yrt.ListModeLUTOwned(scanner, listmode_path), None

    raw = np.fromfile(
        listmode_path,
        dtype=np.uint32,
        count=used_events * FIELDS_PER_EVENT,
    )
    if raw.size != used_events * FIELDS_PER_EVENT:
        raise RuntimeError(f"Could not read {used_events} events from {listmode_path}")

    events = raw.reshape(-1, FIELDS_PER_EVENT)
    timestamps = np.ascontiguousarray(events[:, 0])
    detector1 = np.ascontiguousarray(events[:, 1])
    detector2 = np.ascontiguousarray(events[:, 2])
    dataset = yrt.ListModeLUTAlias(scanner)
    dataset.bind(timestamps, detector1, detector2)
    return dataset, (raw, timestamps, detector1, detector2)


def make_metal_recipe(args, cache_budget_mb):
    correction_reserve_mb = (
        cache_budget_mb
        if str(args.metal_correction_cache_reserve_mb).lower() == "auto"
        else float(args.metal_correction_cache_reserve_mb)
    )
    return MetalRecipeOptions(
        osem_kernel=args.metal_projector,
        sensitivity_kernel=args.metal_sensitivity_projector,
        cache_max_bytes=int(cache_budget_mb * 1024.0 * 1024.0),
        correction_cache_reserve_bytes=int(correction_reserve_mb * 1024.0 * 1024.0),
        direct_frame_batches_explicit=True,
        direct_frame_batches=True,
        native_float_atomics_explicit=True,
        native_float_atomics=True,
        joseph_adjoint_axis_switch_once_explicit=True,
        joseph_adjoint_axis_switch_once=True,
        threads_per_threadgroup_explicit=True,
        threads_per_threadgroup=int(args.metal_threads_per_threadgroup),
    )


def configure_metal_operator_projector(operator_projector, kernel, recipe):
    if not hasattr(yrt, "ExperimentalMetalOperatorProjectorOptions"):
        raise RuntimeError("This pyyrtpet build does not expose Metal projector options")
    options = yrt.ExperimentalMetalOperatorProjectorOptions()
    options.enabled = True
    options.kernel = kernel
    apply_projector_runtime_options(options, recipe)
    operator_projector.setExperimentalMetalProjectorOptions(options)


def configure_metal_osem(recon, args, recipe):
    if not hasattr(yrt, "ExperimentalMetalProjectorOptions"):
        raise RuntimeError("This pyyrtpet build does not expose Metal OSEM options")
    options = yrt.ExperimentalMetalProjectorOptions()
    options.enabled = True
    options.kernel = recipe.osem_kernel
    options.profiling = bool(args.profile_metal)
    options.resident_images = True
    options.image_psf = bool(args.psf)
    options.cache_enabled = True
    options.cached_corrections = True
    options.cache_max_bytes = recipe.cache_max_bytes
    options.correction_cache_reserve_bytes = recipe.correction_cache_reserve_bytes
    options.max_batch_events = int(args.metal_batch_events)
    apply_projector_runtime_options(
        options, recipe, include_direct_frame_batches=True
    )
    recon.setExperimentalMetalProjectorOptions(options)
    if args.profile_metal and hasattr(recon, "resetExperimentalMetalProjectorTimings"):
        recon.resetExperimentalMetalProjectorTimings()


def build_sensitivity(
    scanner,
    img_params,
    histo_corr,
    dataset,
    lor_motion,
    psf_csv_path,
    args,
    recipe,
):
    sens_img = yrt.ImageOwned(img_params)
    sens_img.allocate()
    np.array(sens_img, copy=False).fill(0.0)

    bin_iter = histo_corr.getBinIter(1, 0)
    proj_params = yrt.ProjectorParams(scanner)
    proj_params.setProjector("Siddon")
    projector = yrt.createOperatorProjector(proj_params, bin_iter)
    configure_metal_operator_projector(projector, recipe.sensitivity_kernel, recipe)

    print(f"Computing sensitivity map with Metal {recipe.sensitivity_kernel}...")
    projector.applyAH(histo_corr, sens_img)

    if args.psf:
        print("Applying image PSF adjoint to sensitivity map...")
        yrt.OperatorPsf(psf_csv_path).applyAH(sens_img, sens_img)

    if args.motion and args.move_sensitivity:
        if not hasattr(yrt, "timeAverageMoveImageMetal"):
            raise RuntimeError("This pyyrtpet build does not expose timeAverageMoveImageMetal")
        print("Moving sensitivity image with Metal...")
        sens_img = yrt.timeAverageMoveImageMetal(
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
    sens_alias.multWithScalar(args.global_scale_factor)
    return sens_alias, sens_np


def image_stats(image):
    values = np.array(image, copy=False)
    return (
        float(np.sum(values, dtype=np.float64)),
        float(np.max(values)),
        int(np.count_nonzero(values)),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal experimental Metal GE OSEM reconstruction."
    )
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--image-params", default="img_param_0.8mm.json")
    parser.add_argument("--out-dir", "--out_dir", default="")
    parser.add_argument("--pct", type=float, default=100.0)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--subsets", type=int, default=17)
    parser.add_argument("--global-scale-factor", type=float, default=DEFAULT_GLOBAL_SCALE_FACTOR)
    parser.add_argument("--listmode-loader", choices=["auto", "owned", "alias"], default="auto")
    parser.add_argument("--metal-projector", choices=["joseph", "siddon"], default="joseph")
    parser.add_argument(
        "--metal-sensitivity-projector",
        choices=["joseph", "siddon"],
        default="joseph",
    )
    parser.add_argument("--metal-cache-budget-mb", default="auto")
    parser.add_argument("--metal-correction-cache-reserve-mb", default="auto")
    parser.add_argument("--metal-cache-headroom-gib", type=float, default=12.0)
    parser.add_argument("--metal-cache-budget-fraction", type=float, default=0.55)
    parser.add_argument("--metal-batch-events", type=int, default=1000000)
    parser.add_argument("--metal-threads-per-threadgroup", type=int, default=512)
    parser.add_argument("--profile-metal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-psf", dest="psf", action="store_false")
    parser.add_argument("--no-motion", dest="motion", action="store_false")
    parser.add_argument("--no-move-sensitivity", dest="move_sensitivity", action="store_false")
    parser.add_argument("--no-write-sensitivity", dest="write_sensitivity", action="store_false")
    parser.set_defaults(
        psf=True,
        motion=True,
        move_sensitivity=True,
        write_sensitivity=True,
        metal_native_float_atomics=True,
        metal_joseph_sample_stride=1,
        metal_joseph_axis_specialization="none",
        metal_joseph_axis_specialized=False,
        metal_joseph_adjoint_accumulation="none",
        metal_joseph_adjoint_axis_switch_once=True,
        metal_joseph_adjoint_incremental_coords=False,
        metal_adjoint_event_order="none",
        metal_adjoint_tile_size=8,
        metal_adjoint_diagnostic_max_batches=0,
        metal_adjoint_diagnostic_stride=1,
        profile_metal_ratio_nonzero=False,
        metal_direct_frame_batches=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    apply_metal_environment(args)

    base = os.path.abspath(args.base)
    out_dir = args.out_dir or os.path.join(base, "recon", "metal_minimal")
    config_dir = os.path.join(base, "GE_config")
    listmode_path = os.path.join(base, "PseudoListMode.yrt")
    image_params_path = (
        args.image_params
        if os.path.isabs(args.image_params)
        else os.path.join(base, args.image_params)
    )

    total_events, used_events = resolve_event_count(
        listmode_path, args.pct, args.max_events
    )
    loader = resolve_listmode_loader(args.listmode_loader, total_events, used_events)
    cache_budget_mb = resolve_cache_budget_mb(
        args.metal_cache_budget_mb,
        loader,
        used_events,
        args.metal_cache_headroom_gib,
        args.metal_cache_budget_fraction,
    )
    recipe = make_metal_recipe(args, cache_budget_mb)

    print(f"base={base}")
    print(f"out_dir={out_dir}")
    print(f"total_events={total_events}")
    print(f"used_events={used_events}")
    print(f"iterations={args.iterations}")
    print(f"subsets={args.subsets}")
    print(f"motion={args.motion}")
    print(f"psf={args.psf}")
    print(f"listmode_loader={loader}")
    print(f"listmode_peak_gib={listmode_peak_gib(loader, used_events):.3f}")
    print(f"metal_projector={recipe.osem_kernel}")
    print(f"metal_sensitivity_projector={recipe.sensitivity_kernel}")
    print(f"metal_cache_budget_mb={cache_budget_mb:.3f}")
    print(f"metal_batch_events={args.metal_batch_events}")
    if args.dry_run:
        print("dry_run=True")
        print("PASS")
        return

    os.makedirs(out_dir, exist_ok=True)
    scanner = yrt.Scanner(os.path.join(config_dir, "GE.json"))
    img_params = yrt.ImageParams(image_params_path)

    voxel_size = np.round(img_params.length_x / img_params.nx, 1)
    psf_csv_path = os.path.join(
        out_dir,
        f"psfKernelImageSpace_{voxel_size}mm_x_{img_params.nx}_y_"
        f"{img_params.ny}_z_{img_params.nz}.csv",
    )
    if args.psf:
        create_psf_kernel_image_space(img_params, psf_csv_path, fwhm_x=2.6, fwhm_y=2.6, fwhm_z=3.1)

    setup_start = time.perf_counter()
    dataset, listmode_keepalive = load_listmode_dataset(
        scanner, listmode_path, loader, used_events
    )
    lor_motion = None
    if args.motion:
        lor_motion = yrt.LORMotion(os.path.join(base, "Motion.yrt"))
        dataset.addLORMotion(lor_motion)

    histo_corr, histo_scatter, histo_randoms, corr_np, histo_acf, histo_norm = (
        load_histos(scanner, config_dir, base)
    )
    fraction = float(used_events) / float(total_events)
    randoms, randoms_np = scale_and_bind_histogram3d(scanner, histo_randoms, fraction)
    scatter, scatter_np = scale_and_bind_histogram3d(scanner, histo_scatter, fraction)

    sensitivity, sensitivity_np = build_sensitivity(
        scanner,
        img_params,
        histo_corr,
        dataset,
        lor_motion,
        psf_csv_path,
        args,
        recipe,
    )
    if args.write_sensitivity:
        sensitivity.writeToFile(os.path.join(out_dir, "sens_img_metal.nii.gz"))
    setup_s = time.perf_counter() - setup_start

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
    configure_metal_osem(recon, args, recipe)

    print("Starting Metal reconstruction...")
    recon_start = time.perf_counter()
    image = recon.reconstruct()
    recon_s = time.perf_counter() - recon_start
    image.writeToFile(os.path.join(out_dir, "metal_osem.nii.gz"))

    image_sum, image_max, image_nonzero = image_stats(image)
    metal_projector_ran = recon.didLastExperimentalMetalProjectorRun()
    print(
        "mode,setup_s,recon_s,metal_projector_ran,"
        "image_sum,image_max,image_nonzero"
    )
    print(
        "metal,"
        f"{setup_s:.6f},"
        f"{recon_s:.6f},"
        f"{metal_projector_ran},"
        f"{image_sum:.9g},"
        f"{image_max:.9g},"
        f"{image_nonzero}"
    )
    if args.profile_metal and hasattr(recon, "getExperimentalMetalProjectorTimings"):
        timings = dict(recon.getExperimentalMetalProjectorTimings())
        profile_keys = (
            "total_s",
            "forward_s",
            "ratio_s",
            "adjoint_s",
            "image_psf_forward_s",
            "image_psf_adjoint_s",
        )
        for key in profile_keys:
            if key in timings:
                print(f"{key}={timings[key]:.6f}")
    if not metal_projector_ran:
        raise RuntimeError("Experimental Metal OSEM projector did not run")
    print("PASS")

    _ = (
        listmode_keepalive,
        corr_np,
        histo_acf,
        histo_norm,
        randoms_np,
        scatter_np,
        sensitivity_np,
    )


if __name__ == "__main__":
    main()
