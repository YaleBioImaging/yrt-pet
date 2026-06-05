#!/usr/bin/env python3
"""Real-data GE CPU-vs-Metal OSEM driver for the experimental projector path."""

import argparse
import copy
import csv
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import pyyrtpet as yrt


METAL_CACHE_BYTES_PER_EVENT = 60.0
METAL_COMPACT_CORRECTION_CACHE_BYTES_PER_EVENT = 3.0 * 4.0
DEFAULT_METAL_CACHE_PRESSURE_CAP_MB = 24576.0
LISTMODE_FIELDS_PER_EVENT = 3
LISTMODE_FIELD_BYTES = 4
LISTMODE_BYTES_PER_EVENT = LISTMODE_FIELDS_PER_EVENT * LISTMODE_FIELD_BYTES
LISTMODE_OWNED_READ_BUFFER_FIELDS = 1 << 30
LISTMODE_OWNED_READ_BUFFER_BYTES = (
    LISTMODE_OWNED_READ_BUFFER_FIELDS * LISTMODE_FIELD_BYTES
)


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
    raw = np.fromfile(
        listmode_path,
        dtype=np.uint32,
        count=event_count * LISTMODE_FIELDS_PER_EVENT,
    )
    if raw.size != event_count * LISTMODE_FIELDS_PER_EVENT:
        raise RuntimeError(
            f"Expected {event_count} events from {listmode_path}, got "
            f"{raw.size // LISTMODE_FIELDS_PER_EVENT}"
        )
    events = raw.reshape(-1, LISTMODE_FIELDS_PER_EVENT)
    timestamps = np.ascontiguousarray(events[:, 0])
    detector1 = np.ascontiguousarray(events[:, 1])
    detector2 = np.ascontiguousarray(events[:, 2])

    dataset = yrt.ListModeLUTAlias(scanner)
    dataset.bind(timestamps, detector1, detector2)
    return dataset, timestamps, detector1, detector2


def load_listmode_lut_owned(scanner, listmode_path):
    return yrt.ListModeLUTOwned(scanner, listmode_path)


def resolve_listmode_loader(args, total_events, used_events):
    if args.listmode_loader == "alias":
        return "alias"
    if args.listmode_loader == "owned":
        if used_events != total_events:
            raise SystemExit(
                "--listmode-loader owned can only be used when all events are "
                "selected. Use --pct 100 with no lower --max-events cap, or use "
                "--listmode-loader auto/alias for capped runs."
            )
        return "owned"
    if used_events == total_events:
        return "owned"
    return "alias"


def listmode_memory_plan(loader, used_events):
    selected_bytes = used_events * LISTMODE_BYTES_PER_EVENT
    alias_peak_bytes = selected_bytes * 2
    owned_read_buffer_bytes = min(selected_bytes, LISTMODE_OWNED_READ_BUFFER_BYTES)
    if loader == "owned":
        resident_bytes = selected_bytes
        python_alias_arrays = False
        peak_bytes = selected_bytes + owned_read_buffer_bytes
    else:
        resident_bytes = selected_bytes
        python_alias_arrays = True
        peak_bytes = alias_peak_bytes
    return {
        "listmode_loader": loader,
        "listmode_selected_bytes": selected_bytes,
        "listmode_selected_gib": bytes_to_gib(selected_bytes),
        "listmode_resident_gib": bytes_to_gib(resident_bytes),
        "listmode_read_buffer_gib": bytes_to_gib(owned_read_buffer_bytes)
        if loader == "owned"
        else 0.0,
        "listmode_estimated_peak_gib": bytes_to_gib(peak_bytes),
        "listmode_python_alias_arrays": python_alias_arrays,
    }


def print_listmode_memory_plan(plan):
    print(
        "listmode_plan="
        f"loader:{plan['listmode_loader']};"
        f"selected_gib:{plan['listmode_selected_gib']:.3f};"
        f"resident_gib:{plan['listmode_resident_gib']:.3f};"
        f"read_buffer_gib:{plan['listmode_read_buffer_gib']:.3f};"
        f"estimated_peak_gib:{plan['listmode_estimated_peak_gib']:.3f};"
        f"python_alias_arrays:{plan['listmode_python_alias_arrays']}"
    )


def load_listmode_dataset(scanner, listmode_path, total_events, used_events, args):
    loader = resolve_listmode_loader(args, total_events, used_events)
    plan = listmode_memory_plan(loader, used_events)
    print_listmode_memory_plan(plan)
    if loader == "owned":
        dataset = load_listmode_lut_owned(scanner, listmode_path)
        return dataset, None, None, None, plan

    dataset, timestamps, detector1, detector2 = load_listmode_lut_alias(
        scanner, listmode_path, used_events
    )
    return dataset, timestamps, detector1, detector2, plan


def resolve_event_count(listmode_path, pct, max_events):
    total_events = os.path.getsize(listmode_path) // LISTMODE_BYTES_PER_EVENT
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
    max_abs_value = float(np.max(flat_abs_diff)) if total > 0 else 0.0
    rmse = float(np.sqrt(np.mean(diff * diff))) if total > 0 else 0.0
    ref_rms = float(np.sqrt(np.mean(cpu * cpu))) if total > 0 else 0.0
    diff_norm = float(np.linalg.norm(flat_diff))
    ref_norm = float(np.linalg.norm(flat_cpu))
    eps = 1.0e-12
    top = []
    if total > 0 and top_k > 0 and max_abs_value > 0.0:
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
        "max_abs_diff": max_abs_value,
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


def maybe_format_limit(value):
    return "" if value is None else f"{value:.9g}"


def check_metric(row, metric_name, limit, failures):
    if limit is None:
        return
    value = row.get(metric_name)
    if value is None:
        failures.append(f"{metric_name}=missing>{limit:.9g}")
    elif value > limit:
        failures.append(f"{metric_name}={value:.9g}>{limit:.9g}")


def apply_metric_validation(row, args):
    if not args.validate_metrics:
        return

    failures = []
    if not row.get("metal_projector_ran", False):
        failures.append("metal_projector_ran=False")
    check_metric(row, "rel_l2", args.max_rel_l2, failures)
    check_metric(row, "nrmse", args.max_nrmse, failures)
    check_metric(row, "sum_rel_diff", args.max_sum_rel_diff, failures)
    check_metric(
        row,
        "mismatch_fraction",
        args.max_mismatch_fraction,
        failures,
    )
    row["validation_passed"] = len(failures) == 0
    row["validation_failures"] = "|".join(failures)

    print(
        "validation_passed,max_rel_l2,max_nrmse,max_sum_rel_diff,"
        "max_mismatch_fraction,failures"
    )
    print(
        f"{row['validation_passed']},"
        f"{maybe_format_limit(args.max_rel_l2)},"
        f"{maybe_format_limit(args.max_nrmse)},"
        f"{maybe_format_limit(args.max_sum_rel_diff)},"
        f"{maybe_format_limit(args.max_mismatch_fraction)},"
        f"{row['validation_failures']}"
    )


def finish_metric_validation(rows, args):
    if not args.validate_metrics:
        return

    failed = [row for row in rows if not row.get("validation_passed", False)]
    print(
        "validation_summary,total_cases,passed_cases,failed_cases\n"
        f"{len(rows)},{len(rows) - len(failed)},{len(failed)}"
    )
    if failed:
        for row in failed:
            print(
                "validation_failure,"
                f"{row.get('case', '')},"
                f"{row.get('used_events', '')},"
                f"{row.get('iterations', '')},"
                f"{row.get('subsets', '')},"
                f"{row.get('validation_failures', '')}"
            )
        raise SystemExit(1)


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


def parse_nonnegative_int_list(value, option_name):
    if value is None:
        return []
    text = str(value)
    if not text:
        return []
    parsed = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = int(item)
        except ValueError as exc:
            raise SystemExit(f"{option_name} expects comma-separated integers") from exc
        if number < 0:
            raise SystemExit(f"{option_name} values must be non-negative")
        parsed.append(number)
    if not parsed:
        raise SystemExit(f"{option_name} did not contain any values")
    return parsed


def parse_nonnegative_float_list(value, option_name):
    if value is None:
        return []
    text = str(value)
    if not text:
        return []
    parsed = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = float(item)
        except ValueError as exc:
            raise SystemExit(f"{option_name} expects comma-separated numbers") from exc
        if number < 0.0:
            raise SystemExit(f"{option_name} values must be non-negative")
        parsed.append(number)
    if not parsed:
        raise SystemExit(f"{option_name} did not contain any values")
    return parsed


def format_cache_budget_label(value):
    return f"{value:g}".replace(".", "p")


def format_field(value):
    if isinstance(value, float):
        return f"{value:.9g}"
    if value is None:
        return ""
    return str(value)


def bytes_to_mb(value):
    return float(value) / (1024.0 * 1024.0)


def bytes_to_gib(value):
    return float(value) / (1024.0 * 1024.0 * 1024.0)


def estimate_metal_correction_cache_bytes_per_event(args):
    if not getattr(args, "metal_cached_corrections", False):
        return 0.0
    return METAL_COMPACT_CORRECTION_CACHE_BYTES_PER_EVENT


def estimate_metal_cache_bytes(event_count, args=None):
    correction_bytes = (
        estimate_metal_correction_cache_bytes_per_event(args)
        if args is not None
        else 0.0
    )
    return int(round(float(event_count) * (METAL_CACHE_BYTES_PER_EVENT + correction_bytes)))


def classify_cache_pressure(planned_cache_mb, pressure_cap_mb):
    if planned_cache_mb <= 0.0:
        return "none"
    if pressure_cap_mb <= 0.0:
        return "unknown"
    if planned_cache_mb >= pressure_cap_mb * 1.25:
        return "very_high"
    if planned_cache_mb >= pressure_cap_mb:
        return "high"
    if planned_cache_mb >= pressure_cap_mb * 0.75:
        return "moderate"
    return "low"


def metal_cache_plan(args, used_events):
    correction_bytes_per_event = estimate_metal_correction_cache_bytes_per_event(args)
    estimated_full_bytes = estimate_metal_cache_bytes(used_events, args)
    budget_bytes = int(max(args.metal_cache_budget_mb, 0.0) * 1024.0 * 1024.0)
    metal_requested = args.compare_metal or args.metal_only

    if not metal_requested:
        fit = "not_requested"
        planned_cache_bytes = 0
    elif args.no_metal_cache:
        fit = "disabled"
        planned_cache_bytes = 0
    elif budget_bytes <= 0:
        fit = "streaming"
        planned_cache_bytes = 0
    elif budget_bytes >= estimated_full_bytes:
        fit = "full"
        planned_cache_bytes = estimated_full_bytes
    else:
        fit = "partial"
        planned_cache_bytes = budget_bytes

    planned_cache_mb = bytes_to_mb(planned_cache_bytes)
    pressure_risk = classify_cache_pressure(
        planned_cache_mb,
        args.metal_cache_pressure_cap_mb,
    )
    return {
        "metal_cache_estimate_bytes_per_event": METAL_CACHE_BYTES_PER_EVENT,
        "metal_cache_estimate_correction_bytes_per_event": correction_bytes_per_event,
        "metal_cache_estimated_full_mb": bytes_to_mb(estimated_full_bytes),
        "metal_cache_estimated_full_gib": bytes_to_gib(estimated_full_bytes),
        "metal_cache_fit": fit,
        "metal_cache_planned_gib": bytes_to_gib(planned_cache_bytes),
        "metal_cache_used_gib": bytes_to_gib(planned_cache_bytes),
        "metal_cache_pressure_risk": pressure_risk,
        "metal_cache_pressure_cap_mb": args.metal_cache_pressure_cap_mb,
    }


def print_metal_cache_plan(plan, args):
    if not (args.compare_metal or args.metal_only):
        return
    print(
        "metal_cache_plan="
        f"fit:{plan['metal_cache_fit']};"
        f"estimated_full_gib:{plan['metal_cache_estimated_full_gib']:.3f};"
        f"budget_mb:{args.metal_cache_budget_mb:g};"
        f"planned_cache_gib:{plan['metal_cache_planned_gib']:.3f};"
        f"pressure_risk:{plan['metal_cache_pressure_risk']};"
        f"pressure_cap_mb:{plan['metal_cache_pressure_cap_mb']:g}"
    )
    if plan["metal_cache_fit"] == "partial":
        print(
            "WARNING: Metal cache budget is below the estimated full-cache "
            "size; cached batches will be mixed with streamed batches."
        )
    if plan["metal_cache_pressure_risk"] in ("high", "very_high"):
        print(
            "WARNING: Metal cache pressure risk is "
            f"{plan['metal_cache_pressure_risk']} for this run. Consider a "
            "smaller --metal-cache-budget-mb if the system becomes laggy."
        )


def update_actual_metal_cache_usage(row, profile):
    cache_used_bytes = profile.get("metal_profile_cache_used_bytes")
    if cache_used_bytes is None:
        return
    row["metal_cache_used_gib"] = bytes_to_gib(cache_used_bytes)


def update_metal_profile_recon_gap(row):
    profile_total = row.get("metal_profile_total_s")
    metal_recon = row.get("metal_recon_s")
    if profile_total is None or metal_recon is None:
        return
    gap = max(0.0, float(metal_recon) - float(profile_total))
    row["metal_recon_profile_gap_s"] = gap
    row["metal_recon_profile_gap_fraction"] = (
        gap / max(float(metal_recon), 1.0e-12)
    )
    compute_update = float(row.get("metal_profile_compute_update_image_s", 0.0))
    image_update = float(row.get("metal_profile_image_update_s", 0.0))
    if compute_update > 0.0:
        compute_gap = max(0.0, compute_update - float(profile_total))
        row["metal_compute_profile_gap_s"] = compute_gap
        row["metal_compute_profile_gap_fraction"] = (
            compute_gap / max(compute_update, 1.0e-12)
        )
    if compute_update > 0.0 or image_update > 0.0:
        post_update_gap = max(0.0, float(metal_recon) - compute_update - image_update)
        row["metal_recon_post_update_gap_s"] = post_update_gap
        row["metal_recon_post_update_gap_fraction"] = (
            post_update_gap / max(float(metal_recon), 1.0e-12)
        )
    lifecycle_total = sum(
        float(row.get(field, 0.0) or 0.0)
        for field in (
            "metal_profile_recon_initialize_s",
            "metal_profile_recon_iterate_s",
            "metal_profile_recon_save_iteration_s",
            "metal_profile_recon_complete_mlem_s",
            "metal_profile_recon_end_s",
        )
    )
    if lifecycle_total > 0.0:
        lifecycle_gap = max(0.0, float(metal_recon) - lifecycle_total)
        row["metal_recon_lifecycle_gap_s"] = lifecycle_gap
        row["metal_recon_lifecycle_gap_fraction"] = (
            lifecycle_gap / max(float(metal_recon), 1.0e-12)
        )


def env_flag(name):
    return os.environ.get(name, "") not in ("", "0")


def metal_joseph_axis_specialization_mode(args):
    explicit_mode = getattr(args, "metal_joseph_axis_specialization", "none")
    if explicit_mode != "none":
        return explicit_mode
    return "both" if args.metal_joseph_axis_specialized else "none"


def apply_metal_environment(args):
    os.environ["YRTPET_METAL_JOSEPH_SAMPLE_STRIDE"] = str(
        max(1, int(args.metal_joseph_sample_stride))
    )
    axis_mode = metal_joseph_axis_specialization_mode(args)
    if axis_mode != "none":
        os.environ["YRTPET_METAL_JOSEPH_AXIS_SPECIALIZED"] = axis_mode
    else:
        os.environ.pop("YRTPET_METAL_JOSEPH_AXIS_SPECIALIZED", None)
    if args.metal_joseph_adjoint_accumulation != "none":
        os.environ["YRTPET_METAL_JOSEPH_ADJOINT_ACCUMULATION"] = (
            args.metal_joseph_adjoint_accumulation
        )
    else:
        os.environ.pop("YRTPET_METAL_JOSEPH_ADJOINT_ACCUMULATION", None)
    if args.metal_adjoint_event_order != "none":
        os.environ["YRTPET_METAL_ADJOINT_EVENT_ORDER"] = (
            args.metal_adjoint_event_order
        )
    else:
        os.environ.pop("YRTPET_METAL_ADJOINT_EVENT_ORDER", None)
    os.environ["YRTPET_METAL_ADJOINT_TILE_SIZE"] = str(
        max(1, int(args.metal_adjoint_tile_size))
    )
    threads_per_threadgroup = int(args.metal_threads_per_threadgroup)
    if threads_per_threadgroup > 0:
        os.environ["YRTPET_METAL_THREADS_PER_THREADGROUP"] = str(
            threads_per_threadgroup
        )
    else:
        os.environ.pop("YRTPET_METAL_THREADS_PER_THREADGROUP", None)
    if args.profile_metal_ratio_nonzero:
        os.environ["YRTPET_METAL_PROFILE_RATIO_NONZERO"] = "1"
    else:
        os.environ.pop("YRTPET_METAL_PROFILE_RATIO_NONZERO", None)


def print_metal_projector_notes(args):
    if not (args.compare_metal or args.metal_only):
        return
    if args.metal_adjoint_event_order == "major-axis":
        print(
            "NOTE: --metal-adjoint-event-order major-axis uses a stable "
            "three-bucket ordering by dominant ray axis. It is intended as "
            "the low-overhead full-data adjoint ordering probe."
        )
    elif args.metal_adjoint_event_order == "line-hash":
        print(
            "NOTE: --metal-adjoint-event-order line-hash is diagnostic-only. "
            "A GE full-data checkpoint was much slower because it requires "
            "full sorting and worsened kernel timing."
        )
    elif args.metal_adjoint_event_order == "tile-round-robin":
        print(
            "NOTE: --metal-adjoint-event-order tile-round-robin is an "
            "experimental O(n) tile-aware interleaving probe. It tries to "
            "spread likely adjoint voxel writes across neighboring GPU lanes; "
            "benchmark against none before using it in a recipe."
        )
    if args.profile_metal_ratio_nonzero:
        print(
            "NOTE: --profile-metal-ratio-nonzero downloads post-ratio Metal "
            "projection values to count zeros. Use its active/zero fractions "
            "for adjoint-compaction planning; its timing is diagnostic-only."
        )
    if args.metal_resident_images:
        print(
            "NOTE: --metal-resident-images keeps the current OSEM image and "
            "EM update image on Metal across subsets. It is an opt-in "
            "host-ratio path experiment; final output is downloaded at "
            "reconstruction end."
        )
    if args.metal_projector != "joseph":
        return

    if not env_flag("YRTPET_METAL_USE_NATIVE_FLOAT_ATOMICS"):
        print(
            "NOTE: For Joseph benchmarks on Apple Silicon, current GE "
            "full-data profiling favors "
            "YRTPET_METAL_USE_NATIVE_FLOAT_ATOMICS=1."
        )
    if args.metal_threads_per_threadgroup == 0:
        print(
            "NOTE: Current GE full-data Joseph profiling favors "
            "--metal-threads-per-threadgroup 512 for benchmark runs. "
            "The default auto setting is preserved for A/B comparisons."
        )
    if args.metal_joseph_forward_texture:
        print(
            "NOTE: --metal-joseph-forward-texture is retained only for A/B "
            "testing; the current GE full-data checkpoint was slower than the "
            "buffer-backed Joseph forward path."
        )
    if args.metal_joseph_sample_stride > 1:
        print(
            "NOTE: --metal-joseph-sample-stride is an experimental "
            "reduced-update Joseph approximation. Use it for A/B timing and "
            "image-quality checks only."
        )
    axis_mode = metal_joseph_axis_specialization_mode(args)
    if axis_mode != "none":
        print(
            "NOTE: Joseph axis specialization is an opt-in full-precision "
            f"cache-admission experiment in {axis_mode} mode. It splits "
            "cached Joseph batches by voxel-scaled dominant axis; forward "
            "mode leaves adjoint on the generic Joseph kernel."
        )
    if args.metal_joseph_adjoint_accumulation == "threadgroup-sample":
        print(
            "NOTE: --metal-joseph-adjoint-accumulation threadgroup-sample is "
            "an opt-in full-precision adjoint atomic experiment. It requires "
            "native float atomics, ignores reduced sample-stride mode, and may "
            "be slower if threadgroup barriers dominate."
        )
    if env_flag("YRTPET_METAL_USE_PRIVATE_UPDATE_BUFFER"):
        print(
            "NOTE: YRTPET_METAL_USE_PRIVATE_UPDATE_BUFFER=1 was neutral to "
            "slightly slower for Joseph in the current GE checkpoint; prefer "
            "native atomics alone unless re-benchmarking."
        )

METAL_PROFILE_FLOAT_FIELDS = [
    "total_s",
    "setup_s",
    "setup_context_s",
    "setup_projector_s",
    "setup_cache_s",
    "setup_bridge_s",
    "setup_can_run_s",
    "forward_s",
    "ratio_s",
    "adjoint_s",
    "metal_path_overhead_s",
    "compute_update_image_s",
    "image_update_s",
    "recon_initialize_s",
    "recon_setup_dynamic_s",
    "recon_initialize_out_image_s",
    "recon_initialize_sens_image_s",
    "recon_corrector_setup_s",
    "recon_initialize_bin_iterators_s",
    "recon_collect_constraints_s",
    "recon_setup_projector_s",
    "recon_prepare_buffers_s",
    "recon_iterate_s",
    "recon_load_subset_s",
    "recon_reset_update_s",
    "recon_compute_update_phase_s",
    "recon_apply_update_phase_s",
    "recon_complete_subset_s",
    "recon_save_iteration_s",
    "recon_complete_mlem_s",
    "recon_end_s",
    "prepare_allocate_images_s",
    "prepare_initialize_output_s",
    "prepare_apply_mask_s",
    "prepare_clear_update_s",
    "prepare_precompute_corrections_s",
    "prepare_init_bin_loader_s",
    "prepare_clear_metal_cache_s",
    "forward_gather_s",
    "forward_gather_cache_build_s",
    "forward_gather_uncached_s",
    "forward_gather_direct_s",
    "forward_gather_constrained_s",
    "forward_pack_s",
    "forward_pack_cache_build_s",
    "forward_pack_uncached_s",
    "forward_batch_upload_s",
    "forward_batch_upload_cache_build_s",
    "forward_batch_upload_uncached_s",
    "forward_image_upload_s",
    "forward_kernel_s",
    "forward_download_s",
    "forward_host_write_s",
    "ratio_pack_s",
    "ratio_batch_upload_s",
    "ratio_kernel_s",
    "ratio_correction_cache_build_s",
    "ratio_nonzero_diagnostic_s",
    "adjoint_gather_s",
    "adjoint_gather_cache_build_s",
    "adjoint_gather_uncached_s",
    "adjoint_gather_direct_s",
    "adjoint_gather_constrained_s",
    "adjoint_pack_s",
    "adjoint_pack_cache_build_s",
    "adjoint_pack_uncached_s",
    "adjoint_batch_upload_s",
    "adjoint_batch_upload_cache_build_s",
    "adjoint_batch_upload_uncached_s",
    "adjoint_image_upload_s",
    "adjoint_kernel_s",
    "adjoint_image_download_s",
    "adjoint_host_image_copy_s",
    "adjoint_update_count_s",
    "adjoint_voxel_hit_count_s",
    "cache_lookup_s",
    "cache_admission_s",
    "cache_admission_gather_s",
    "cache_admission_pack_s",
    "cache_admission_batch_upload_s",
    "cache_admission_correction_build_s",
    "cache_admission_correction_fill_s",
    "cache_admission_correction_upload_s",
    "cache_admission_correction_measurement_s",
    "cache_admission_correction_multiplicative_s",
    "cache_admission_correction_additive_s",
    "cache_admission_correction_in_vivo_s",
    "cache_insert_s",
    "adjoint_max_batch_mean_voxel_hits",
    "adjoint_max_batch_top_1pct_voxel_hit_fraction",
    "adjoint_max_batch_top_0_1pct_voxel_hit_fraction",
    "adjoint_max_batch_mean_tile_hits",
    "adjoint_max_batch_top_1pct_tile_hit_fraction",
    "adjoint_max_batch_top_0_1pct_tile_hit_fraction",
]

METAL_PROFILE_COUNT_FIELDS = [
    "calls",
    "forward_events",
    "forward_batches",
    "adjoint_events",
    "adjoint_nonzero_events",
    "adjoint_batches",
    "adjoint_voxel_updates",
    "adjoint_rays_with_updates",
    "adjoint_max_updates_per_ray",
    "adjoint_voxel_hit_maps",
    "adjoint_batch_hit_voxels",
    "adjoint_voxel_hit_total_updates",
    "adjoint_max_voxel_hits",
    "adjoint_max_batch_p50_voxel_hits",
    "adjoint_max_batch_p90_voxel_hits",
    "adjoint_max_batch_p95_voxel_hits",
    "adjoint_max_batch_p99_voxel_hits",
    "adjoint_max_batch_p999_voxel_hits",
    "adjoint_tile_size",
    "adjoint_voxel_hit_tiles",
    "adjoint_voxel_hit_tile_total_updates",
    "adjoint_max_tile_hits",
    "adjoint_max_batch_p95_tile_hits",
    "adjoint_max_batch_p99_tile_hits",
    "cache_lookups",
    "cache_hits",
    "cache_misses",
    "cache_builds",
    "cache_skips_over_budget",
    "cache_used_bytes",
    "cache_max_bytes",
    "cache_correction_reserve_bytes",
    "uncached_batches",
    "ratio_correction_cache_builds",
    "ratio_correction_cache_hits",
    "ratio_correction_cache_misses",
    "ratio_correction_cache_bytes",
    "ratio_values",
    "ratio_nonzero_values",
    "ratio_zero_values",
    "ratio_nonzero_diagnostic_batches",
]

METAL_PROFILE_PRINT_FIELDS = [
    "metal_profile_calls",
    "metal_profile_total_s",
    "metal_profile_setup_s",
    "metal_profile_setup_context_s",
    "metal_profile_setup_projector_s",
    "metal_profile_setup_cache_s",
    "metal_profile_setup_bridge_s",
    "metal_profile_setup_can_run_s",
    "metal_profile_setup_unprofiled_s",
    "metal_profile_forward_s",
    "metal_profile_ratio_s",
    "metal_profile_adjoint_s",
    "metal_profile_metal_path_overhead_s",
    "metal_profile_compute_update_image_s",
    "metal_profile_image_update_s",
    "metal_profile_recon_initialize_s",
    "metal_profile_recon_setup_dynamic_s",
    "metal_profile_recon_initialize_out_image_s",
    "metal_profile_recon_initialize_sens_image_s",
    "metal_profile_recon_corrector_setup_s",
    "metal_profile_recon_initialize_bin_iterators_s",
    "metal_profile_recon_collect_constraints_s",
    "metal_profile_recon_setup_projector_s",
    "metal_profile_recon_prepare_buffers_s",
    "metal_profile_recon_iterate_s",
    "metal_profile_recon_load_subset_s",
    "metal_profile_recon_reset_update_s",
    "metal_profile_recon_compute_update_phase_s",
    "metal_profile_recon_apply_update_phase_s",
    "metal_profile_recon_complete_subset_s",
    "metal_profile_recon_save_iteration_s",
    "metal_profile_recon_complete_mlem_s",
    "metal_profile_recon_end_s",
    "metal_profile_prepare_allocate_images_s",
    "metal_profile_prepare_initialize_output_s",
    "metal_profile_prepare_apply_mask_s",
    "metal_profile_prepare_clear_update_s",
    "metal_profile_prepare_precompute_corrections_s",
    "metal_profile_prepare_init_bin_loader_s",
    "metal_profile_prepare_clear_metal_cache_s",
    "metal_profile_forward_gather_s",
    "metal_profile_forward_gather_cache_build_s",
    "metal_profile_forward_gather_uncached_s",
    "metal_profile_forward_gather_direct_s",
    "metal_profile_forward_gather_constrained_s",
    "metal_profile_forward_pack_s",
    "metal_profile_forward_pack_cache_build_s",
    "metal_profile_forward_pack_uncached_s",
    "metal_profile_forward_batch_upload_s",
    "metal_profile_forward_batch_upload_cache_build_s",
    "metal_profile_forward_batch_upload_uncached_s",
    "metal_profile_forward_image_upload_s",
    "metal_profile_forward_kernel_s",
    "metal_profile_forward_download_s",
    "metal_profile_forward_host_write_s",
    "metal_profile_ratio_pack_s",
    "metal_profile_ratio_batch_upload_s",
    "metal_profile_ratio_kernel_s",
    "metal_profile_ratio_correction_cache_build_s",
    "metal_profile_ratio_nonzero_diagnostic_s",
    "metal_profile_adjoint_gather_s",
    "metal_profile_adjoint_gather_cache_build_s",
    "metal_profile_adjoint_gather_uncached_s",
    "metal_profile_adjoint_gather_direct_s",
    "metal_profile_adjoint_gather_constrained_s",
    "metal_profile_adjoint_pack_s",
    "metal_profile_adjoint_pack_cache_build_s",
    "metal_profile_adjoint_pack_uncached_s",
    "metal_profile_adjoint_batch_upload_s",
    "metal_profile_adjoint_batch_upload_cache_build_s",
    "metal_profile_adjoint_batch_upload_uncached_s",
    "metal_profile_adjoint_image_upload_s",
    "metal_profile_adjoint_kernel_s",
    "metal_profile_adjoint_image_download_s",
    "metal_profile_adjoint_host_image_copy_s",
    "metal_profile_adjoint_update_count_s",
    "metal_profile_adjoint_voxel_hit_count_s",
    "metal_profile_cache_lookup_s",
    "metal_profile_cache_admission_s",
    "metal_profile_cache_admission_gather_s",
    "metal_profile_cache_admission_pack_s",
    "metal_profile_cache_admission_batch_upload_s",
    "metal_profile_cache_admission_correction_build_s",
    "metal_profile_cache_admission_correction_fill_s",
    "metal_profile_cache_admission_correction_upload_s",
    "metal_profile_cache_admission_correction_measurement_s",
    "metal_profile_cache_admission_correction_multiplicative_s",
    "metal_profile_cache_admission_correction_additive_s",
    "metal_profile_cache_admission_correction_in_vivo_s",
    "metal_profile_cache_insert_s",
    "metal_profile_forward_events",
    "metal_profile_forward_batches",
    "metal_profile_adjoint_events",
    "metal_profile_adjoint_nonzero_events",
    "metal_profile_adjoint_batches",
    "metal_profile_adjoint_voxel_updates",
    "metal_profile_adjoint_rays_with_updates",
    "metal_profile_adjoint_max_updates_per_ray",
    "metal_profile_adjoint_voxel_hit_maps",
    "metal_profile_adjoint_batch_hit_voxels",
    "metal_profile_adjoint_voxel_hit_total_updates",
    "metal_profile_adjoint_max_voxel_hits",
    "metal_profile_adjoint_max_batch_p50_voxel_hits",
    "metal_profile_adjoint_max_batch_p90_voxel_hits",
    "metal_profile_adjoint_max_batch_p95_voxel_hits",
    "metal_profile_adjoint_max_batch_p99_voxel_hits",
    "metal_profile_adjoint_max_batch_p999_voxel_hits",
    "metal_profile_adjoint_max_batch_mean_voxel_hits",
    "metal_profile_adjoint_max_batch_top_1pct_voxel_hit_fraction",
    "metal_profile_adjoint_max_batch_top_0_1pct_voxel_hit_fraction",
    "metal_profile_adjoint_tile_size",
    "metal_profile_adjoint_voxel_hit_tiles",
    "metal_profile_adjoint_voxel_hit_tile_total_updates",
    "metal_profile_adjoint_max_tile_hits",
    "metal_profile_adjoint_max_batch_p95_tile_hits",
    "metal_profile_adjoint_max_batch_p99_tile_hits",
    "metal_profile_adjoint_max_batch_mean_tile_hits",
    "metal_profile_adjoint_max_batch_top_1pct_tile_hit_fraction",
    "metal_profile_adjoint_max_batch_top_0_1pct_tile_hit_fraction",
    "metal_profile_cache_lookups",
    "metal_profile_cache_hits",
    "metal_profile_cache_misses",
    "metal_profile_cache_builds",
    "metal_profile_cache_skips_over_budget",
    "metal_profile_cache_used_bytes",
    "metal_profile_cache_max_bytes",
    "metal_profile_cache_correction_reserve_bytes",
    "metal_profile_uncached_batches",
    "metal_profile_ratio_correction_cache_builds",
    "metal_profile_ratio_correction_cache_hits",
    "metal_profile_ratio_correction_cache_misses",
    "metal_profile_ratio_correction_cache_bytes",
    "metal_profile_ratio_values",
    "metal_profile_ratio_nonzero_values",
    "metal_profile_ratio_zero_values",
    "metal_profile_ratio_nonzero_fraction",
    "metal_profile_ratio_zero_fraction",
    "metal_profile_ratio_nonzero_diagnostic_batches",
    "metal_profile_total_per_call_s",
    "metal_profile_forward_per_call_s",
    "metal_profile_adjoint_per_call_s",
]

METAL_SUBSET_PROFILE_FIELDS = [
    "case",
    "call_index",
    "iteration",
    "subset",
    "iteration_index",
    "subset_index",
    "events",
    "metal_ran",
    "total_s",
    "setup_s",
    "setup_context_s",
    "setup_projector_s",
    "setup_cache_s",
    "setup_bridge_s",
    "setup_can_run_s",
    "forward_s",
    "ratio_s",
    "adjoint_s",
    "metal_path_overhead_s",
    "compute_update_image_s",
    "image_update_s",
    "forward_gather_s",
    "forward_gather_cache_build_s",
    "forward_gather_uncached_s",
    "forward_pack_s",
    "forward_batch_upload_s",
    "forward_image_upload_s",
    "forward_kernel_s",
    "forward_download_s",
    "ratio_pack_s",
    "ratio_batch_upload_s",
    "ratio_kernel_s",
    "ratio_nonzero_diagnostic_s",
    "adjoint_batch_upload_s",
    "adjoint_image_upload_s",
    "adjoint_kernel_s",
    "adjoint_image_download_s",
    "adjoint_host_image_copy_s",
    "adjoint_update_count_s",
    "adjoint_voxel_hit_count_s",
    "cache_lookup_s",
    "cache_admission_s",
    "cache_admission_gather_s",
    "cache_admission_pack_s",
    "cache_admission_batch_upload_s",
    "cache_admission_correction_build_s",
    "cache_admission_correction_fill_s",
    "cache_admission_correction_upload_s",
    "cache_admission_correction_measurement_s",
    "cache_admission_correction_multiplicative_s",
    "cache_admission_correction_additive_s",
    "cache_admission_correction_in_vivo_s",
    "cache_insert_s",
    "forward_events",
    "forward_batches",
    "adjoint_events",
    "adjoint_nonzero_events",
    "adjoint_batches",
    "adjoint_voxel_updates",
    "adjoint_rays_with_updates",
    "adjoint_max_updates_per_ray",
    "adjoint_voxel_hit_maps",
    "adjoint_batch_hit_voxels",
    "adjoint_voxel_hit_total_updates",
    "adjoint_max_voxel_hits",
    "adjoint_max_batch_p50_voxel_hits",
    "adjoint_max_batch_p90_voxel_hits",
    "adjoint_max_batch_p95_voxel_hits",
    "adjoint_max_batch_p99_voxel_hits",
    "adjoint_max_batch_p999_voxel_hits",
    "adjoint_max_batch_mean_voxel_hits",
    "adjoint_max_batch_top_1pct_voxel_hit_fraction",
    "adjoint_max_batch_top_0_1pct_voxel_hit_fraction",
    "adjoint_tile_size",
    "adjoint_voxel_hit_tiles",
    "adjoint_voxel_hit_tile_total_updates",
    "adjoint_max_tile_hits",
    "adjoint_max_batch_p95_tile_hits",
    "adjoint_max_batch_p99_tile_hits",
    "adjoint_max_batch_mean_tile_hits",
    "adjoint_max_batch_top_1pct_tile_hit_fraction",
    "adjoint_max_batch_top_0_1pct_tile_hit_fraction",
    "cache_lookups",
    "cache_hits",
    "cache_misses",
    "cache_builds",
    "cache_skips_over_budget",
    "cache_used_bytes",
    "cache_max_bytes",
    "cache_correction_reserve_bytes",
    "uncached_batches",
    "ratio_correction_cache_builds",
    "ratio_correction_cache_hits",
    "ratio_correction_cache_misses",
    "ratio_correction_cache_bytes",
    "ratio_values",
    "ratio_nonzero_values",
    "ratio_zero_values",
    "ratio_nonzero_fraction",
    "ratio_zero_fraction",
    "ratio_nonzero_diagnostic_batches",
    "memory_before_pressure",
    "memory_before_available_gib",
    "memory_before_free_gib",
    "memory_before_compressed_gib",
    "memory_before_used_gib",
    "memory_before_available_ratio",
    "memory_before_free_ratio",
    "memory_before_compressed_ratio",
    "memory_after_pressure",
    "memory_after_available_gib",
    "memory_after_free_gib",
    "memory_after_compressed_gib",
    "memory_after_used_gib",
    "memory_after_available_ratio",
    "memory_after_free_ratio",
    "memory_after_compressed_ratio",
    "memory_pageouts_delta",
    "memory_compressions_delta",
    "memory_swapouts_delta",
]


def normalized_metal_profile(raw_profile):
    if not raw_profile:
        return {}
    profile = {}
    for field in METAL_PROFILE_FLOAT_FIELDS:
        profile[f"metal_profile_{field}"] = float(raw_profile.get(field, 0.0))
    for field in METAL_PROFILE_COUNT_FIELDS:
        fallback_field = "uncached_chunks" if field == "uncached_batches" else field
        profile[f"metal_profile_{field}"] = int(
            raw_profile.get(field, raw_profile.get(fallback_field, 0))
        )
    calls = max(profile["metal_profile_calls"], 1)
    profile["metal_profile_total_per_call_s"] = (
        profile["metal_profile_total_s"] / calls
    )
    profile["metal_profile_forward_per_call_s"] = (
        profile["metal_profile_forward_s"] / calls
    )
    profile["metal_profile_adjoint_per_call_s"] = (
        profile["metal_profile_adjoint_s"] / calls
    )
    ratio_values = profile.get("metal_profile_ratio_values", 0)
    if ratio_values > 0:
        profile["metal_profile_ratio_nonzero_fraction"] = (
            profile.get("metal_profile_ratio_nonzero_values", 0) / ratio_values
        )
        profile["metal_profile_ratio_zero_fraction"] = (
            profile.get("metal_profile_ratio_zero_values", 0) / ratio_values
        )
    else:
        profile["metal_profile_ratio_nonzero_fraction"] = 0.0
        profile["metal_profile_ratio_zero_fraction"] = 0.0
    setup_subtotal = sum(
        profile.get(field, 0.0)
        for field in (
            "metal_profile_setup_context_s",
            "metal_profile_setup_projector_s",
            "metal_profile_setup_cache_s",
            "metal_profile_setup_bridge_s",
            "metal_profile_setup_can_run_s",
        )
    )
    profile["metal_profile_setup_unprofiled_s"] = max(
        0.0, profile.get("metal_profile_setup_s", 0.0) - setup_subtotal
    )
    return profile


def flatten_memory_snapshot(row, prefix, snapshot):
    snapshot = dict(snapshot or {})
    row[f"{prefix}_sample_available"] = bool(snapshot.get("available", False))
    row[f"{prefix}_pressure"] = snapshot.get("pressure_level", "unavailable")
    row[f"{prefix}_total_gib"] = bytes_to_gib(snapshot.get("total_bytes", 0))
    row[f"{prefix}_available_gib"] = bytes_to_gib(
        snapshot.get("available_bytes", 0)
    )
    row[f"{prefix}_used_gib"] = bytes_to_gib(snapshot.get("used_bytes", 0))
    row[f"{prefix}_free_gib"] = bytes_to_gib(snapshot.get("free_bytes", 0))
    row[f"{prefix}_speculative_gib"] = bytes_to_gib(
        snapshot.get("speculative_bytes", 0)
    )
    row[f"{prefix}_active_gib"] = bytes_to_gib(snapshot.get("active_bytes", 0))
    row[f"{prefix}_inactive_gib"] = bytes_to_gib(
        snapshot.get("inactive_bytes", 0)
    )
    row[f"{prefix}_wired_gib"] = bytes_to_gib(snapshot.get("wired_bytes", 0))
    row[f"{prefix}_compressed_gib"] = bytes_to_gib(
        snapshot.get("compressed_bytes", 0)
    )
    row[f"{prefix}_available_ratio"] = float(
        snapshot.get("available_ratio", 0.0)
    )
    row[f"{prefix}_free_ratio"] = float(snapshot.get("free_ratio", 0.0))
    row[f"{prefix}_compressed_ratio"] = float(
        snapshot.get("compressed_ratio", 0.0)
    )
    for field in (
        "pageins",
        "pageouts",
        "decompressions",
        "compressions",
        "swapins",
        "swapouts",
    ):
        row[f"{prefix}_{field}"] = int(snapshot.get(field, 0))


def add_memory_counter_deltas(row):
    for field in (
        "pageins",
        "pageouts",
        "decompressions",
        "compressions",
        "swapins",
        "swapouts",
    ):
        before = int(row.get(f"memory_before_{field}", 0) or 0)
        after = int(row.get(f"memory_after_{field}", 0) or 0)
        row[f"memory_{field}_delta"] = max(0, after - before)


def normalized_metal_subset_profiles(raw_profiles, case_label=""):
    rows = []
    for index, raw in enumerate(raw_profiles or []):
        raw = dict(raw)
        row = {
            "case": case_label,
            "call_index": index,
            "iteration": int(raw.get("iteration", 0)),
            "subset": int(raw.get("subset", 0)),
            "iteration_index": int(raw.get("iteration_index", -1)),
            "subset_index": int(raw.get("subset_index", -1)),
            "events": int(raw.get("events", 0)),
            "metal_ran": bool(raw.get("metal_ran", False)),
        }
        for field in (
            "setup_s",
            "setup_context_s",
            "setup_projector_s",
            "setup_cache_s",
            "setup_bridge_s",
            "setup_can_run_s",
            "forward_s",
            "ratio_s",
            "adjoint_s",
            "total_s",
            "metal_path_overhead_s",
            "compute_update_image_s",
            "image_update_s",
            "forward_gather_s",
            "forward_gather_cache_build_s",
            "forward_gather_uncached_s",
            "forward_pack_s",
            "forward_batch_upload_s",
            "forward_image_upload_s",
            "forward_kernel_s",
            "forward_download_s",
            "ratio_pack_s",
            "ratio_batch_upload_s",
            "ratio_kernel_s",
            "ratio_nonzero_diagnostic_s",
            "adjoint_batch_upload_s",
            "adjoint_image_upload_s",
            "adjoint_kernel_s",
            "adjoint_image_download_s",
            "adjoint_host_image_copy_s",
            "adjoint_update_count_s",
            "adjoint_voxel_hit_count_s",
            "cache_lookup_s",
            "cache_admission_s",
            "cache_admission_gather_s",
            "cache_admission_pack_s",
            "cache_admission_batch_upload_s",
            "cache_admission_correction_build_s",
            "cache_admission_correction_fill_s",
            "cache_admission_correction_upload_s",
            "cache_admission_correction_measurement_s",
            "cache_admission_correction_multiplicative_s",
            "cache_admission_correction_additive_s",
            "cache_admission_correction_in_vivo_s",
            "cache_insert_s",
            "adjoint_max_batch_mean_voxel_hits",
            "adjoint_max_batch_top_1pct_voxel_hit_fraction",
            "adjoint_max_batch_top_0_1pct_voxel_hit_fraction",
            "adjoint_max_batch_mean_tile_hits",
            "adjoint_max_batch_top_1pct_tile_hit_fraction",
            "adjoint_max_batch_top_0_1pct_tile_hit_fraction",
        ):
            row[field] = float(raw.get(field, 0.0))
        for field in (
            "forward_events",
            "forward_batches",
            "adjoint_events",
            "adjoint_nonzero_events",
            "adjoint_batches",
            "adjoint_voxel_updates",
            "adjoint_rays_with_updates",
            "adjoint_max_updates_per_ray",
            "adjoint_voxel_hit_maps",
            "adjoint_batch_hit_voxels",
            "adjoint_voxel_hit_total_updates",
            "adjoint_max_voxel_hits",
            "adjoint_max_batch_p50_voxel_hits",
            "adjoint_max_batch_p90_voxel_hits",
            "adjoint_max_batch_p95_voxel_hits",
            "adjoint_max_batch_p99_voxel_hits",
            "adjoint_max_batch_p999_voxel_hits",
            "adjoint_tile_size",
            "adjoint_voxel_hit_tiles",
            "adjoint_voxel_hit_tile_total_updates",
            "adjoint_max_tile_hits",
            "adjoint_max_batch_p95_tile_hits",
            "adjoint_max_batch_p99_tile_hits",
            "cache_lookups",
            "cache_hits",
            "cache_misses",
            "cache_builds",
            "cache_skips_over_budget",
            "cache_used_bytes",
            "cache_max_bytes",
            "cache_correction_reserve_bytes",
            "uncached_batches",
            "ratio_correction_cache_builds",
            "ratio_correction_cache_hits",
            "ratio_correction_cache_misses",
            "ratio_correction_cache_bytes",
            "ratio_values",
            "ratio_nonzero_values",
            "ratio_zero_values",
            "ratio_nonzero_diagnostic_batches",
        ):
            row[field] = int(raw.get(field, 0))
        ratio_values = row.get("ratio_values", 0)
        if ratio_values > 0:
            row["ratio_nonzero_fraction"] = (
                row.get("ratio_nonzero_values", 0) / ratio_values
            )
            row["ratio_zero_fraction"] = (
                row.get("ratio_zero_values", 0) / ratio_values
            )
        else:
            row["ratio_nonzero_fraction"] = 0.0
            row["ratio_zero_fraction"] = 0.0
        flatten_memory_snapshot(row, "memory_before", raw.get("memory_before"))
        flatten_memory_snapshot(row, "memory_after", raw.get("memory_after"))
        add_memory_counter_deltas(row)
        rows.append(row)
    return rows


def summarize_metal_subset_profiles(row, subset_profiles):
    if not subset_profiles:
        return
    slowest = max(subset_profiles, key=lambda item: item.get("total_s", 0.0))
    pressure_rank = {"unavailable": 0, "green": 1, "yellow": 2, "red": 3}
    pressure_values = [
        item.get(key, "unavailable")
        for item in subset_profiles
        for key in ("memory_before_pressure", "memory_after_pressure")
    ]
    worst_pressure = max(
        pressure_values,
        key=lambda value: pressure_rank.get(str(value), 0),
        default="unavailable",
    )
    available_ratios = [
        item.get(key, 0.0)
        for item in subset_profiles
        for key in (
            "memory_before_available_ratio",
            "memory_after_available_ratio",
        )
        if item.get(key, 0.0) > 0.0
    ]
    available_gib = [
        item.get(key, 0.0)
        for item in subset_profiles
        for key in ("memory_before_available_gib", "memory_after_available_gib")
        if item.get(key, 0.0) > 0.0
    ]
    free_ratios = [
        item.get(key, 0.0)
        for item in subset_profiles
        for key in ("memory_before_free_ratio", "memory_after_free_ratio")
        if item.get(key, 0.0) > 0.0
    ]
    compressed_ratios = [
        item.get(key, 0.0)
        for item in subset_profiles
        for key in (
            "memory_before_compressed_ratio",
            "memory_after_compressed_ratio",
        )
        if item.get(key, 0.0) > 0.0
    ]
    row["metal_subset_profile_calls"] = len(subset_profiles)
    row["metal_subset_slowest_s"] = slowest.get("total_s", 0.0)
    row["metal_subset_slowest_iteration"] = slowest.get("iteration", "")
    row["metal_subset_slowest_subset"] = slowest.get("subset", "")
    row["metal_subset_max_forward_gather_s"] = max(
        item.get("forward_gather_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_max_ratio_s"] = max(
        item.get("ratio_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_max_adjoint_s"] = max(
        item.get("adjoint_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_setup_context_s"] = sum(
        item.get("setup_context_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_setup_projector_s"] = sum(
        item.get("setup_projector_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_setup_cache_s"] = sum(
        item.get("setup_cache_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_setup_bridge_s"] = sum(
        item.get("setup_bridge_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_setup_can_run_s"] = sum(
        item.get("setup_can_run_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_metal_path_overhead_s"] = sum(
        item.get("metal_path_overhead_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_max_metal_path_overhead_s"] = max(
        item.get("metal_path_overhead_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_compute_update_image_s"] = sum(
        item.get("compute_update_image_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_max_compute_update_image_s"] = max(
        item.get("compute_update_image_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_image_update_s"] = sum(
        item.get("image_update_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_max_image_update_s"] = max(
        item.get("image_update_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_cache_lookup_s"] = sum(
        item.get("cache_lookup_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_cache_admission_s"] = sum(
        item.get("cache_admission_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_cache_admission_gather_s"] = sum(
        item.get("cache_admission_gather_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_cache_admission_pack_s"] = sum(
        item.get("cache_admission_pack_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_cache_admission_batch_upload_s"] = sum(
        item.get("cache_admission_batch_upload_s", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_cache_admission_correction_build_s"] = sum(
        item.get("cache_admission_correction_build_s", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_cache_admission_correction_fill_s"] = sum(
        item.get("cache_admission_correction_fill_s", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_cache_admission_correction_upload_s"] = sum(
        item.get("cache_admission_correction_upload_s", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_cache_admission_correction_measurement_s"] = sum(
        item.get("cache_admission_correction_measurement_s", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_cache_admission_correction_multiplicative_s"] = sum(
        item.get("cache_admission_correction_multiplicative_s", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_cache_admission_correction_additive_s"] = sum(
        item.get("cache_admission_correction_additive_s", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_cache_admission_correction_in_vivo_s"] = sum(
        item.get("cache_admission_correction_in_vivo_s", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_cache_insert_s"] = sum(
        item.get("cache_insert_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_adjoint_update_count_s"] = sum(
        item.get("adjoint_update_count_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_adjoint_voxel_hit_count_s"] = sum(
        item.get("adjoint_voxel_hit_count_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_ratio_nonzero_diagnostic_s"] = sum(
        item.get("ratio_nonzero_diagnostic_s", 0.0) for item in subset_profiles
    )
    row["metal_subset_ratio_values"] = sum(
        item.get("ratio_values", 0) for item in subset_profiles
    )
    row["metal_subset_ratio_nonzero_values"] = sum(
        item.get("ratio_nonzero_values", 0) for item in subset_profiles
    )
    row["metal_subset_ratio_zero_values"] = sum(
        item.get("ratio_zero_values", 0) for item in subset_profiles
    )
    row["metal_subset_ratio_nonzero_diagnostic_batches"] = sum(
        item.get("ratio_nonzero_diagnostic_batches", 0)
        for item in subset_profiles
    )
    if row["metal_subset_ratio_values"] > 0:
        row["metal_subset_ratio_nonzero_fraction"] = (
            row["metal_subset_ratio_nonzero_values"]
            / row["metal_subset_ratio_values"]
        )
        row["metal_subset_ratio_zero_fraction"] = (
            row["metal_subset_ratio_zero_values"]
            / row["metal_subset_ratio_values"]
        )
    row["metal_subset_adjoint_voxel_updates"] = sum(
        item.get("adjoint_voxel_updates", 0) for item in subset_profiles
    )
    row["metal_subset_adjoint_rays_with_updates"] = sum(
        item.get("adjoint_rays_with_updates", 0) for item in subset_profiles
    )
    row["metal_subset_adjoint_max_updates_per_ray"] = max(
        item.get("adjoint_max_updates_per_ray", 0) for item in subset_profiles
    )
    row["metal_subset_adjoint_voxel_hit_maps"] = sum(
        item.get("adjoint_voxel_hit_maps", 0) for item in subset_profiles
    )
    row["metal_subset_adjoint_batch_hit_voxels"] = sum(
        item.get("adjoint_batch_hit_voxels", 0) for item in subset_profiles
    )
    row["metal_subset_adjoint_voxel_hit_total_updates"] = sum(
        item.get("adjoint_voxel_hit_total_updates", 0) for item in subset_profiles
    )
    row["metal_subset_adjoint_max_voxel_hits"] = max(
        item.get("adjoint_max_voxel_hits", 0) for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_p50_voxel_hits"] = max(
        item.get("adjoint_max_batch_p50_voxel_hits", 0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_p90_voxel_hits"] = max(
        item.get("adjoint_max_batch_p90_voxel_hits", 0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_p95_voxel_hits"] = max(
        item.get("adjoint_max_batch_p95_voxel_hits", 0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_p99_voxel_hits"] = max(
        item.get("adjoint_max_batch_p99_voxel_hits", 0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_p999_voxel_hits"] = max(
        item.get("adjoint_max_batch_p999_voxel_hits", 0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_mean_voxel_hits"] = max(
        item.get("adjoint_max_batch_mean_voxel_hits", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_top_1pct_voxel_hit_fraction"] = max(
        item.get("adjoint_max_batch_top_1pct_voxel_hit_fraction", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_top_0_1pct_voxel_hit_fraction"] = max(
        item.get("adjoint_max_batch_top_0_1pct_voxel_hit_fraction", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_tile_size"] = max(
        item.get("adjoint_tile_size", 0) for item in subset_profiles
    )
    row["metal_subset_adjoint_voxel_hit_tiles"] = sum(
        item.get("adjoint_voxel_hit_tiles", 0) for item in subset_profiles
    )
    row["metal_subset_adjoint_voxel_hit_tile_total_updates"] = sum(
        item.get("adjoint_voxel_hit_tile_total_updates", 0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_tile_hits"] = max(
        item.get("adjoint_max_tile_hits", 0) for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_p95_tile_hits"] = max(
        item.get("adjoint_max_batch_p95_tile_hits", 0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_p99_tile_hits"] = max(
        item.get("adjoint_max_batch_p99_tile_hits", 0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_mean_tile_hits"] = max(
        item.get("adjoint_max_batch_mean_tile_hits", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_top_1pct_tile_hit_fraction"] = max(
        item.get("adjoint_max_batch_top_1pct_tile_hit_fraction", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_adjoint_max_batch_top_0_1pct_tile_hit_fraction"] = max(
        item.get("adjoint_max_batch_top_0_1pct_tile_hit_fraction", 0.0)
        for item in subset_profiles
    )
    row["metal_subset_worst_memory_pressure"] = worst_pressure
    if available_ratios:
        row["metal_subset_min_memory_available_ratio"] = min(available_ratios)
    if available_gib:
        row["metal_subset_min_memory_available_gib"] = min(available_gib)
    if free_ratios:
        row["metal_subset_min_memory_free_ratio"] = min(free_ratios)
    if compressed_ratios:
        row["metal_subset_max_memory_compressed_ratio"] = max(compressed_ratios)
    row["metal_subset_pageouts_delta"] = sum(
        item.get("memory_pageouts_delta", 0) for item in subset_profiles
    )
    row["metal_subset_compressions_delta"] = sum(
        item.get("memory_compressions_delta", 0) for item in subset_profiles
    )
    row["metal_subset_swapouts_delta"] = sum(
        item.get("memory_swapouts_delta", 0) for item in subset_profiles
    )


def print_metal_profile(profile):
    if not profile:
        return
    print(",".join(METAL_PROFILE_PRINT_FIELDS))
    print(
        ",".join(
            format_field(profile.get(field))
            for field in METAL_PROFILE_PRINT_FIELDS
        )
    )


def print_metal_subset_profile(subset_profiles):
    if not subset_profiles:
        return
    print("metal_subset_profile")
    print(",".join(METAL_SUBSET_PROFILE_FIELDS))
    for row in subset_profiles:
        print(
            ",".join(
                format_field(row.get(field))
                for field in METAL_SUBSET_PROFILE_FIELDS
            )
        )


def warn_if_requested_adjoint_diagnostics_missing(args, profile):
    if not profile:
        return
    if args.profile_metal_ratio_nonzero:
        if (
            profile.get("metal_profile_adjoint_events", 0) > 0
            and profile.get("metal_profile_ratio_values", 0) == 0
        ):
            print(
                "WARNING: requested Metal ratio nonzero profiling, but no "
                "ratio values were counted. Rebuild pyyrtpet and rerun before "
                "using this CSV for adjoint-compaction planning."
            )
    if args.profile_metal_adjoint_diagnostics:
        if (
            profile.get("metal_profile_adjoint_events", 0) > 0
            and profile.get("metal_profile_adjoint_update_count_s", 0.0) == 0.0
            and profile.get("metal_profile_adjoint_voxel_updates", 0) == 0
        ):
            print(
                "WARNING: requested Metal adjoint update diagnostics, but no "
                "update-count counters were reported. Rebuild pyyrtpet and "
                "rerun; otherwise this benchmark cannot diagnose adjoint "
                "atomic/update pressure."
            )
    if args.profile_metal_adjoint_hit_diagnostics:
        if (
            profile.get("metal_profile_adjoint_events", 0) > 0
            and profile.get("metal_profile_adjoint_voxel_hit_count_s", 0.0) == 0.0
            and profile.get("metal_profile_adjoint_voxel_hit_total_updates", 0) == 0
        ):
            print(
                "WARNING: requested Metal adjoint voxel-hit diagnostics, but "
                "no voxel-hit counters were reported. Rebuild pyyrtpet and "
                "rerun before using this CSV for tiled/hybrid adjoint planning."
            )


def print_sweep_summary(rows):
    fields = [
        "used_events",
        "iterations",
        "subsets",
        "move_sensitivity",
        "metal_cache_budget_mb",
        "metal_correction_cache_reserve_mb",
        "metal_batch_events",
        "metal_threads_per_threadgroup",
        "metal_fused_ratio",
        "metal_profile_adjoint_diagnostics",
        "metal_profile_adjoint_hit_diagnostics",
        "metal_profile_adjoint_contention",
        "metal_profile_ratio_nonzero",
        "metal_adjoint_event_order",
        "metal_adjoint_tile_size",
        "metal_resident_images",
        "metal_joseph_adjoint_accumulation",
        "metal_projector",
        "metal_joseph_forward_texture",
        "metal_joseph_axis_specialized",
        "metal_joseph_axis_specialization",
        "metal_joseph_sample_stride",
        "metal_sensitivity_projector",
        "metal_lazy_corrections",
        "metal_cached_corrections",
        "metal_native_float_atomics",
        "metal_private_update_buffer",
        "metal_only",
        "metal_cache_estimated_full_mb",
        "metal_cache_fit",
        "metal_cache_used_gib",
        "metal_cache_pressure_risk",
        "listmode_loader",
        "listmode_selected_gib",
        "listmode_resident_gib",
        "listmode_read_buffer_gib",
        "listmode_estimated_peak_gib",
        "listmode_python_alias_arrays",
        "setup_s",
        "cpu_recon_s",
        "metal_recon_s",
        "metal_recon_profile_gap_s",
        "metal_recon_profile_gap_fraction",
        "metal_compute_profile_gap_s",
        "metal_compute_profile_gap_fraction",
        "metal_recon_post_update_gap_s",
        "metal_recon_post_update_gap_fraction",
        "metal_recon_lifecycle_gap_s",
        "metal_recon_lifecycle_gap_fraction",
        "metal_over_cpu",
        "max_abs_diff",
        "max_rel_diff",
        "rmse",
        "nrmse",
        "rel_l2",
        "sum_rel_diff",
        "mismatches",
        "mismatch_fraction",
        "metal_projector_ran",
        "metal_profile_calls",
        "metal_profile_total_s",
        "metal_profile_setup_s",
        "metal_profile_setup_context_s",
        "metal_profile_setup_projector_s",
        "metal_profile_setup_cache_s",
        "metal_profile_setup_bridge_s",
        "metal_profile_setup_can_run_s",
        "metal_profile_setup_unprofiled_s",
        "metal_profile_forward_s",
        "metal_profile_forward_gather_s",
        "metal_profile_forward_gather_cache_build_s",
        "metal_profile_forward_gather_uncached_s",
        "metal_profile_forward_gather_direct_s",
        "metal_profile_forward_gather_constrained_s",
        "metal_profile_forward_pack_s",
        "metal_profile_forward_pack_cache_build_s",
        "metal_profile_forward_pack_uncached_s",
        "metal_profile_forward_batch_upload_s",
        "metal_profile_forward_batch_upload_cache_build_s",
        "metal_profile_forward_batch_upload_uncached_s",
        "metal_profile_forward_image_upload_s",
        "metal_profile_forward_kernel_s",
        "metal_profile_forward_download_s",
        "metal_profile_ratio_s",
        "metal_profile_ratio_pack_s",
        "metal_profile_ratio_batch_upload_s",
        "metal_profile_ratio_kernel_s",
        "metal_profile_ratio_correction_cache_build_s",
        "metal_profile_ratio_nonzero_diagnostic_s",
        "metal_profile_ratio_values",
        "metal_profile_ratio_nonzero_values",
        "metal_profile_ratio_zero_values",
        "metal_profile_ratio_nonzero_fraction",
        "metal_profile_ratio_zero_fraction",
        "metal_profile_ratio_nonzero_diagnostic_batches",
        "metal_profile_adjoint_s",
        "metal_profile_metal_path_overhead_s",
        "metal_profile_compute_update_image_s",
        "metal_profile_image_update_s",
        "metal_profile_adjoint_gather_s",
        "metal_profile_adjoint_batch_upload_s",
        "metal_profile_adjoint_image_upload_s",
        "metal_profile_adjoint_kernel_s",
        "metal_profile_adjoint_image_download_s",
        "metal_profile_adjoint_host_image_copy_s",
        "metal_profile_adjoint_update_count_s",
        "metal_profile_adjoint_voxel_hit_count_s",
        "metal_profile_adjoint_voxel_updates",
        "metal_profile_adjoint_rays_with_updates",
        "metal_profile_adjoint_max_updates_per_ray",
        "metal_profile_adjoint_voxel_hit_maps",
        "metal_profile_adjoint_batch_hit_voxels",
        "metal_profile_adjoint_voxel_hit_total_updates",
        "metal_profile_adjoint_max_voxel_hits",
        "metal_profile_adjoint_max_batch_p50_voxel_hits",
        "metal_profile_adjoint_max_batch_p90_voxel_hits",
        "metal_profile_adjoint_max_batch_p95_voxel_hits",
        "metal_profile_adjoint_max_batch_p99_voxel_hits",
        "metal_profile_adjoint_max_batch_p999_voxel_hits",
        "metal_profile_adjoint_max_batch_mean_voxel_hits",
        "metal_profile_adjoint_max_batch_top_1pct_voxel_hit_fraction",
        "metal_profile_adjoint_max_batch_top_0_1pct_voxel_hit_fraction",
        "metal_profile_adjoint_tile_size",
        "metal_profile_adjoint_voxel_hit_tiles",
        "metal_profile_adjoint_voxel_hit_tile_total_updates",
        "metal_profile_adjoint_max_tile_hits",
        "metal_profile_adjoint_max_batch_p95_tile_hits",
        "metal_profile_adjoint_max_batch_p99_tile_hits",
        "metal_profile_adjoint_max_batch_mean_tile_hits",
        "metal_profile_adjoint_max_batch_top_1pct_tile_hit_fraction",
        "metal_profile_adjoint_max_batch_top_0_1pct_tile_hit_fraction",
        "metal_profile_cache_lookup_s",
        "metal_profile_cache_admission_s",
        "metal_profile_cache_admission_gather_s",
        "metal_profile_cache_admission_pack_s",
        "metal_profile_cache_admission_batch_upload_s",
        "metal_profile_cache_admission_correction_build_s",
        "metal_profile_cache_admission_correction_fill_s",
        "metal_profile_cache_admission_correction_upload_s",
        "metal_profile_cache_admission_correction_measurement_s",
        "metal_profile_cache_admission_correction_multiplicative_s",
        "metal_profile_cache_admission_correction_additive_s",
        "metal_profile_cache_admission_correction_in_vivo_s",
        "metal_profile_cache_insert_s",
        "metal_profile_forward_batches",
        "metal_profile_adjoint_batches",
        "metal_profile_cache_hits",
        "metal_profile_cache_misses",
        "metal_profile_cache_builds",
        "metal_profile_cache_skips_over_budget",
        "metal_profile_cache_used_bytes",
        "metal_profile_cache_max_bytes",
        "metal_profile_cache_correction_reserve_bytes",
        "metal_profile_uncached_batches",
        "metal_profile_ratio_correction_cache_builds",
        "metal_profile_ratio_correction_cache_hits",
        "metal_profile_ratio_correction_cache_misses",
        "metal_profile_ratio_correction_cache_bytes",
        "metal_subset_profile_calls",
        "metal_subset_slowest_s",
        "metal_subset_slowest_iteration",
        "metal_subset_slowest_subset",
        "metal_subset_max_forward_gather_s",
        "metal_subset_max_ratio_s",
        "metal_subset_max_adjoint_s",
        "metal_subset_setup_context_s",
        "metal_subset_setup_projector_s",
        "metal_subset_setup_cache_s",
        "metal_subset_setup_bridge_s",
        "metal_subset_setup_can_run_s",
        "metal_subset_metal_path_overhead_s",
        "metal_subset_max_metal_path_overhead_s",
        "metal_subset_compute_update_image_s",
        "metal_subset_max_compute_update_image_s",
        "metal_subset_image_update_s",
        "metal_subset_max_image_update_s",
        "metal_subset_cache_lookup_s",
        "metal_subset_cache_admission_s",
        "metal_subset_cache_admission_gather_s",
        "metal_subset_cache_admission_pack_s",
        "metal_subset_cache_admission_batch_upload_s",
        "metal_subset_cache_admission_correction_build_s",
        "metal_subset_cache_admission_correction_fill_s",
        "metal_subset_cache_admission_correction_upload_s",
        "metal_subset_cache_admission_correction_measurement_s",
        "metal_subset_cache_admission_correction_multiplicative_s",
        "metal_subset_cache_admission_correction_additive_s",
        "metal_subset_cache_admission_correction_in_vivo_s",
        "metal_subset_cache_insert_s",
        "metal_subset_adjoint_update_count_s",
        "metal_subset_adjoint_voxel_hit_count_s",
        "metal_subset_ratio_nonzero_diagnostic_s",
        "metal_subset_ratio_values",
        "metal_subset_ratio_nonzero_values",
        "metal_subset_ratio_zero_values",
        "metal_subset_ratio_nonzero_fraction",
        "metal_subset_ratio_zero_fraction",
        "metal_subset_ratio_nonzero_diagnostic_batches",
        "metal_subset_adjoint_voxel_updates",
        "metal_subset_adjoint_rays_with_updates",
        "metal_subset_adjoint_max_updates_per_ray",
        "metal_subset_adjoint_voxel_hit_maps",
        "metal_subset_adjoint_batch_hit_voxels",
        "metal_subset_adjoint_voxel_hit_total_updates",
        "metal_subset_adjoint_max_voxel_hits",
        "metal_subset_adjoint_max_batch_p50_voxel_hits",
        "metal_subset_adjoint_max_batch_p90_voxel_hits",
        "metal_subset_adjoint_max_batch_p95_voxel_hits",
        "metal_subset_adjoint_max_batch_p99_voxel_hits",
        "metal_subset_adjoint_max_batch_p999_voxel_hits",
        "metal_subset_adjoint_max_batch_mean_voxel_hits",
        "metal_subset_adjoint_max_batch_top_1pct_voxel_hit_fraction",
        "metal_subset_adjoint_max_batch_top_0_1pct_voxel_hit_fraction",
        "metal_subset_adjoint_tile_size",
        "metal_subset_adjoint_voxel_hit_tiles",
        "metal_subset_adjoint_voxel_hit_tile_total_updates",
        "metal_subset_adjoint_max_tile_hits",
        "metal_subset_adjoint_max_batch_p95_tile_hits",
        "metal_subset_adjoint_max_batch_p99_tile_hits",
        "metal_subset_adjoint_max_batch_mean_tile_hits",
        "metal_subset_adjoint_max_batch_top_1pct_tile_hit_fraction",
        "metal_subset_adjoint_max_batch_top_0_1pct_tile_hit_fraction",
        "metal_subset_worst_memory_pressure",
        "metal_subset_min_memory_available_ratio",
        "metal_subset_min_memory_available_gib",
        "metal_subset_min_memory_free_ratio",
        "metal_subset_max_memory_compressed_ratio",
        "metal_subset_pageouts_delta",
        "metal_subset_compressions_delta",
        "metal_subset_swapouts_delta",
        "validation_passed",
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
        "move_sensitivity",
        "metal_cache_budget_mb",
        "metal_correction_cache_reserve_mb",
        "metal_batch_events",
        "metal_threads_per_threadgroup",
        "metal_fused_ratio",
        "metal_profile_adjoint_diagnostics",
        "metal_profile_adjoint_hit_diagnostics",
        "metal_profile_adjoint_contention",
        "metal_profile_ratio_nonzero",
        "metal_adjoint_event_order",
        "metal_adjoint_tile_size",
        "metal_joseph_adjoint_accumulation",
        "metal_projector",
        "metal_joseph_forward_texture",
        "metal_joseph_axis_specialized",
        "metal_joseph_axis_specialization",
        "metal_joseph_sample_stride",
        "metal_sensitivity_projector",
        "metal_lazy_corrections",
        "metal_cached_corrections",
        "metal_native_float_atomics",
        "metal_private_update_buffer",
        "metal_only",
        "metal_cache_estimate_bytes_per_event",
        "metal_cache_estimate_correction_bytes_per_event",
        "metal_cache_estimated_full_mb",
        "metal_cache_estimated_full_gib",
        "metal_cache_fit",
        "metal_cache_planned_gib",
        "metal_cache_used_gib",
        "metal_cache_pressure_risk",
        "metal_cache_pressure_cap_mb",
        "listmode_loader",
        "listmode_selected_bytes",
        "listmode_selected_gib",
        "listmode_resident_gib",
        "listmode_read_buffer_gib",
        "listmode_estimated_peak_gib",
        "listmode_python_alias_arrays",
        "setup_s",
        "cpu_recon_s",
        "metal_recon_s",
        "metal_recon_profile_gap_s",
        "metal_recon_profile_gap_fraction",
        "metal_compute_profile_gap_s",
        "metal_compute_profile_gap_fraction",
        "metal_recon_post_update_gap_s",
        "metal_recon_post_update_gap_fraction",
        "metal_recon_lifecycle_gap_s",
        "metal_recon_lifecycle_gap_fraction",
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
        "sum_rel_diff",
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
        *METAL_PROFILE_PRINT_FIELDS,
        "metal_subset_profile_calls",
        "metal_subset_slowest_s",
        "metal_subset_slowest_iteration",
        "metal_subset_slowest_subset",
        "metal_subset_max_forward_gather_s",
        "metal_subset_max_ratio_s",
        "metal_subset_max_adjoint_s",
        "metal_subset_setup_context_s",
        "metal_subset_setup_projector_s",
        "metal_subset_setup_cache_s",
        "metal_subset_setup_bridge_s",
        "metal_subset_setup_can_run_s",
        "metal_subset_metal_path_overhead_s",
        "metal_subset_max_metal_path_overhead_s",
        "metal_subset_compute_update_image_s",
        "metal_subset_max_compute_update_image_s",
        "metal_subset_image_update_s",
        "metal_subset_max_image_update_s",
        "metal_subset_cache_lookup_s",
        "metal_subset_cache_admission_s",
        "metal_subset_cache_admission_gather_s",
        "metal_subset_cache_admission_pack_s",
        "metal_subset_cache_admission_batch_upload_s",
        "metal_subset_cache_admission_correction_build_s",
        "metal_subset_cache_admission_correction_fill_s",
        "metal_subset_cache_admission_correction_upload_s",
        "metal_subset_cache_admission_correction_measurement_s",
        "metal_subset_cache_admission_correction_multiplicative_s",
        "metal_subset_cache_admission_correction_additive_s",
        "metal_subset_cache_admission_correction_in_vivo_s",
        "metal_subset_cache_insert_s",
        "metal_subset_adjoint_update_count_s",
        "metal_subset_adjoint_voxel_hit_count_s",
        "metal_subset_adjoint_voxel_updates",
        "metal_subset_adjoint_rays_with_updates",
        "metal_subset_adjoint_max_updates_per_ray",
        "metal_subset_adjoint_voxel_hit_maps",
        "metal_subset_adjoint_batch_hit_voxels",
        "metal_subset_adjoint_voxel_hit_total_updates",
        "metal_subset_adjoint_max_voxel_hits",
        "metal_subset_adjoint_max_batch_p50_voxel_hits",
        "metal_subset_adjoint_max_batch_p90_voxel_hits",
        "metal_subset_adjoint_max_batch_p95_voxel_hits",
        "metal_subset_adjoint_max_batch_p99_voxel_hits",
        "metal_subset_adjoint_max_batch_p999_voxel_hits",
        "metal_subset_adjoint_max_batch_mean_voxel_hits",
        "metal_subset_adjoint_max_batch_top_1pct_voxel_hit_fraction",
        "metal_subset_adjoint_max_batch_top_0_1pct_voxel_hit_fraction",
        "metal_subset_adjoint_tile_size",
        "metal_subset_adjoint_voxel_hit_tiles",
        "metal_subset_adjoint_voxel_hit_tile_total_updates",
        "metal_subset_adjoint_max_tile_hits",
        "metal_subset_adjoint_max_batch_p95_tile_hits",
        "metal_subset_adjoint_max_batch_p99_tile_hits",
        "metal_subset_adjoint_max_batch_mean_tile_hits",
        "metal_subset_adjoint_max_batch_top_1pct_tile_hit_fraction",
        "metal_subset_adjoint_max_batch_top_0_1pct_tile_hit_fraction",
        "metal_subset_worst_memory_pressure",
        "metal_subset_min_memory_available_ratio",
        "metal_subset_min_memory_available_gib",
        "metal_subset_min_memory_free_ratio",
        "metal_subset_max_memory_compressed_ratio",
        "metal_subset_pageouts_delta",
        "metal_subset_compressions_delta",
        "metal_subset_swapouts_delta",
        "validation_passed",
        "validation_failures",
    ]
    field_set = set()
    for row in rows:
        field_set.update(key for key in row.keys() if not key.startswith("_"))
    fields = [field for field in preferred_fields if field in field_set]
    fields.extend(sorted(field_set.difference(fields)))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: value for key, value in row.items() if not key.startswith("_")}
            )


def write_metal_subset_profile_csv(path, rows):
    subset_rows = []
    for row in rows:
        subset_rows.extend(row.get("_metal_subset_profile", []))
    if not subset_rows:
        return False
    field_set = set()
    for row in subset_rows:
        field_set.update(row.keys())
    fields = [
        field for field in METAL_SUBSET_PROFILE_FIELDS if field in field_set
    ]
    fields.extend(sorted(field_set.difference(fields)))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in subset_rows:
            writer.writerow(row)
    return True


def default_metal_subset_profile_csv_path(summary_csv):
    if not summary_csv:
        return ""
    root, ext = os.path.splitext(summary_csv)
    return f"{root}_subsets{ext or '.csv'}"


def read_single_summary_row(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one summary row in {path}, got {len(rows)}")
    return rows[0]


def strip_cli_options(argv, value_options, flag_options):
    result = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        option = token.split("=", 1)[0]
        if option in value_options:
            if "=" not in token:
                skip_next = True
            continue
        if option in flag_options:
            continue
        result.append(token)
    return result


def isolated_sweep_base_argv():
    value_options = {
        "--validation-profile",
        "--max-events",
        "--iterations",
        "--subsets",
        "--sweep-events",
        "--sweep-iterations",
        "--sweep-subsets",
        "--metal-batch-events",
        "--metal-chunk-events",
        "--metal-cache-budget-mb",
        "--metal-correction-cache-reserve-mb",
        "--metal-threads-per-threadgroup",
        "--summary-csv",
        "--metal-subset-profile-csv",
        "--out-dir",
        "--out_dir",
    }
    flag_options = {"--isolated-sweep"}
    return strip_cli_options(sys.argv[1:], value_options, flag_options)


def run_isolated_sweep(
    args,
    event_values,
    iteration_values,
    subset_values,
    batch_event_values,
    cache_budget_values,
    correction_reserve_values,
    threadgroup_values,
    base_out_dir,
):
    rows = []
    base_argv = isolated_sweep_base_argv()
    script_path = os.path.abspath(__file__)
    for max_events in event_values:
        for iterations in iteration_values:
            for subsets in subset_values:
                for batch_events in batch_event_values:
                    for cache_budget_mb in cache_budget_values:
                        for correction_reserve_mb in correction_reserve_values:
                            for threads_per_threadgroup in threadgroup_values:
                                case_label = (
                                    f"events_{max_events}_iters_{iterations}"
                                    f"_subsets_{subsets}_batch_{batch_events}"
                                    f"_cachemb_"
                                    f"{format_cache_budget_label(cache_budget_mb)}"
                                    f"_corrmb_"
                                    f"{format_cache_budget_label(correction_reserve_mb)}"
                                    f"_tpg_{threads_per_threadgroup}"
                                )
                                case_out_dir = os.path.join(base_out_dir, case_label)
                                case_summary_csv = os.path.join(
                                    case_out_dir, "summary.csv"
                                )
                                command = [
                                    sys.executable,
                                    script_path,
                                    *base_argv,
                                    "--max-events",
                                    str(max_events),
                                    "--iterations",
                                    str(iterations),
                                    "--subsets",
                                    str(subsets),
                                    "--metal-batch-events",
                                    str(batch_events),
                                    "--metal-cache-budget-mb",
                                    f"{cache_budget_mb:g}",
                                    "--metal-correction-cache-reserve-mb",
                                    f"{correction_reserve_mb:g}",
                                    "--metal-threads-per-threadgroup",
                                    str(threads_per_threadgroup),
                                    "--out-dir",
                                    case_out_dir,
                                    "--summary-csv",
                                    case_summary_csv,
                                ]
                                print(f"isolated_case={case_label}", flush=True)
                                print(
                                    "isolated_command="
                                    + " ".join(shlex.quote(part) for part in command),
                                    flush=True,
                                )
                                completed = subprocess.run(command)
                                if completed.returncode != 0:
                                    raise SystemExit(completed.returncode)
                                row = read_single_summary_row(case_summary_csv)
                                row["case"] = case_label
                                rows.append(row)
    return rows


def export_lors_csv(dataset, path, limit, stride, projection_value):
    if stride <= 0:
        raise SystemExit("--export-lors-stride must be positive")
    if limit < 0:
        raise SystemExit("--export-lors-limit must be non-negative")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exported = 0
    with open(path, "w", newline="") as handle:
        handle.write("# x1,y1,z1,x2,y2,z2,projection_value,dynamic_frame\n")
        writer = csv.writer(handle)
        for bin_index in range(0, dataset.count(), stride):
            if limit > 0 and exported >= limit:
                break
            lor = dataset.getLOR(bin_index)
            frame = dataset.getDynamicFrame(bin_index) if dataset.hasDynamicFraming() else 0
            writer.writerow(
                [
                    f"{lor.point1.x:.9g}",
                    f"{lor.point1.y:.9g}",
                    f"{lor.point1.z:.9g}",
                    f"{lor.point2.x:.9g}",
                    f"{lor.point2.y:.9g}",
                    f"{lor.point2.z:.9g}",
                    f"{projection_value:.9g}",
                    int(frame),
                ]
            )
            exported += 1
    return exported


def build_sensitivity(
    scanner,
    img_params,
    histo_corr,
    dataset,
    lor_motion,
    psf_csv_path,
    use_motion,
    move_sensitivity,
    use_psf,
    global_scale_factor,
    sensitivity_projector,
):
    sens_img = yrt.ImageOwned(img_params)
    sens_img.allocate()
    np.array(sens_img, copy=False).fill(0.0)

    bin_iter = histo_corr.getBinIter(1, 0)
    proj_params = yrt.ProjectorParams(scanner)
    proj_params.setProjector("Siddon")
    operator_siddon = yrt.createOperatorProjector(proj_params, bin_iter)
    if sensitivity_projector == "joseph":
        missing = [
            name
            for name in (
                "setExperimentalMetalProjectorEnabled",
                "setExperimentalMetalProjectorKernel",
            )
            if not hasattr(operator_siddon, name)
        ]
        if missing:
            raise RuntimeError(
                "This pyyrtpet build does not expose OperatorProjector Metal "
                "kernel controls: " + ",".join(missing)
            )
        operator_siddon.setExperimentalMetalProjectorEnabled(True)
        operator_siddon.setExperimentalMetalProjectorKernel("joseph")
        print("Computing sensitivity map with experimental Metal Joseph...")
    else:
        print("Computing sensitivity map with Siddon...")
    operator_siddon.applyAH(histo_corr, sens_img)

    if use_psf:
        print("Applying image PSF to sensitivity map...")
        yrt.OperatorPsf(psf_csv_path).applyA(sens_img, sens_img)

    if use_motion and move_sensitivity:
        print("Moving sensitivity image...")
        sens_img = yrt.timeAverageMoveImage(
            lor_motion,
            sens_img,
            int(dataset.getTimestamp(0)),
            int(dataset.getTimestamp(dataset.count() - 1)),
        )
    elif use_motion:
        print("Skipping sensitivity image motion averaging...")

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
        metal_projector_kernel = (
            "joseph_texture_forward"
            if args.metal_joseph_forward_texture
            else args.metal_projector
        )
        if metal_projector_kernel != "siddon":
            if not hasattr(recon, "setExperimentalMetalProjectorKernel"):
                raise RuntimeError(
                    "This pyyrtpet build does not expose Metal projector "
                    "kernel selection"
                )
            recon.setExperimentalMetalProjectorKernel(metal_projector_kernel)
        if args.metal_fused_ratio:
            if not hasattr(recon, "setExperimentalMetalProjectorFusedRatioEnabled"):
                raise RuntimeError(
                    "This pyyrtpet build does not expose Metal fused ratio controls"
                )
            recon.setExperimentalMetalProjectorFusedRatioEnabled(True)
        if args.metal_resident_images:
            if not hasattr(
                recon,
                "setExperimentalMetalProjectorResidentImagesEnabled",
            ):
                raise RuntimeError(
                    "This pyyrtpet build does not expose Metal resident image "
                    "controls"
                )
            recon.setExperimentalMetalProjectorResidentImagesEnabled(True)
        cache_required = (
            "setExperimentalMetalProjectorCacheEnabled",
            "setExperimentalMetalProjectorCacheMaxBytes",
            "setExperimentalMetalProjectorCorrectionCacheReserveBytes",
            "setExperimentalMetalProjectorMaxBatchEvents",
        )
        missing_cache = [name for name in cache_required if not hasattr(recon, name)]
        if missing_cache:
            raise RuntimeError(
                "This pyyrtpet build does not expose Metal OSEM cache controls: "
                + ",".join(missing_cache)
            )
        recon.setExperimentalMetalProjectorCacheEnabled(not args.no_metal_cache)
        recon.setExperimentalMetalProjectorCacheMaxBytes(
            int(args.metal_cache_budget_mb * 1024.0 * 1024.0)
        )
        recon.setExperimentalMetalProjectorCorrectionCacheReserveBytes(
            int(args.metal_correction_cache_reserve_mb * 1024.0 * 1024.0)
        )
        recon.setExperimentalMetalProjectorMaxBatchEvents(args.metal_batch_events)
        if args.metal_lazy_corrections:
            if not hasattr(
                recon,
                "setExperimentalMetalProjectorLazyCorrectionsEnabled",
            ):
                raise RuntimeError(
                    "This pyyrtpet build does not expose Metal lazy correction "
                    "controls"
                )
            recon.setExperimentalMetalProjectorLazyCorrectionsEnabled(True)
        if args.metal_cached_corrections:
            if not hasattr(
                recon,
                "setExperimentalMetalProjectorCachedCorrectionsEnabled",
            ):
                raise RuntimeError(
                    "This pyyrtpet build does not expose Metal cached "
                    "correction controls"
                )
            recon.setExperimentalMetalProjectorCachedCorrectionsEnabled(True)
        if args.profile_metal:
            required = (
                "setExperimentalMetalProjectorProfilingEnabled",
                "resetExperimentalMetalProjectorTimings",
                "getExperimentalMetalProjectorTimings",
            )
            missing = [name for name in required if not hasattr(recon, name)]
            if missing:
                raise RuntimeError(
                    "This pyyrtpet build does not expose Metal OSEM profiling: "
                    + ",".join(missing)
                )
            recon.setExperimentalMetalProjectorProfilingEnabled(True)
            if args.profile_metal_adjoint_diagnostics:
                if not hasattr(
                    recon,
                    "setExperimentalMetalProjectorAdjointDiagnosticsEnabled",
                ):
                    raise RuntimeError(
                        "This pyyrtpet build does not expose Metal adjoint "
                        "diagnostics"
                    )
                recon.setExperimentalMetalProjectorAdjointDiagnosticsEnabled(True)
            if args.profile_metal_adjoint_hit_diagnostics:
                if not hasattr(
                    recon,
                    "setExperimentalMetalProjectorAdjointHitDiagnosticsEnabled",
                ):
                    raise RuntimeError(
                        "This pyyrtpet build does not expose Metal adjoint "
                        "hit diagnostics"
                    )
                recon.setExperimentalMetalProjectorAdjointHitDiagnosticsEnabled(
                    True
                )
            recon.resetExperimentalMetalProjectorTimings()

    start = time.perf_counter()
    image = recon.reconstruct()
    elapsed = time.perf_counter() - start
    profile = {}
    subset_profile = []
    if use_metal and args.profile_metal:
        profile = normalized_metal_profile(
            dict(recon.getExperimentalMetalProjectorTimings())
        )
        if hasattr(recon, "getExperimentalMetalProjectorSubsetTimings"):
            subset_profile = normalized_metal_subset_profiles(
                recon.getExperimentalMetalProjectorSubsetTimings()
            )
    return recon, image, elapsed, profile, subset_profile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-data mini-hot-spot OSEM, optionally CPU-vs-Metal."
    )
    parser.add_argument("--base", default="/Users/yanischemli/Desktop/mini_hot_spot")
    parser.add_argument("--image-params", default="img_param_0.8mm.json")
    parser.add_argument(
        "--validation-profile",
        choices=[
            "",
            "ge-mini-hotspot-siddon-4k-smoke",
            "ge-mini-hotspot-1m-10it",
        ],
        default="",
        help="Apply a named explicit validation profile.",
    )
    parser.add_argument("--pct", type=float, default=10.0)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--subsets", type=int, default=17)
    parser.add_argument(
        "--listmode-loader",
        choices=["auto", "alias", "owned"],
        default="auto",
        help=(
            "ListModeLUT loader for the GE pseudo-listmode file. auto uses "
            "YRT's C++ owned loader for full-file runs and the NumPy alias "
            "loader for capped runs."
        ),
    )
    parser.add_argument("--global-scale-factor", type=float, default=1.0 / 2.3e5)
    parser.add_argument("--motion", dest="motion", action="store_true", default=True)
    parser.add_argument("--no-motion", dest="motion", action="store_false")
    parser.add_argument(
        "--move-sensitivity",
        dest="move_sensitivity",
        action="store_true",
        default=True,
        help="Apply motion time-averaging to the sensitivity image when motion is enabled.",
    )
    parser.add_argument(
        "--no-move-sensitivity",
        dest="move_sensitivity",
        action="store_false",
        help=(
            "Keep LOR motion enabled but skip the expensive sensitivity image "
            "motion time-averaging step."
        ),
    )
    parser.add_argument("--psf", dest="psf", action="store_true", default=True)
    parser.add_argument("--no-psf", dest="psf", action="store_false")
    parser.add_argument("--compare-metal", action="store_true")
    parser.add_argument(
        "--metal-only",
        action="store_true",
        help=(
            "Run only the experimental Metal OSEM path and skip CPU reference "
            "and image-difference metrics. Use after smaller CPU-vs-Metal "
            "validation has established numerical agreement."
        ),
    )
    parser.add_argument(
        "--profile-metal",
        action="store_true",
        help="Collect coarse timing buckets from the experimental Metal OSEM path.",
    )
    parser.add_argument(
        "--profile-metal-adjoint-diagnostics",
        action="store_true",
        help=(
            "With --profile-metal, run an extra diagnostic Metal pass that "
            "counts Siddon/Joseph adjoint voxel updates per ray. This perturbs "
            "timing and should be used for bottleneck diagnosis, not baseline "
            "timing."
        ),
    )
    parser.add_argument(
        "--profile-metal-adjoint-hit-diagnostics",
        action="store_true",
        help=(
            "With --profile-metal, run an extra diagnostic Metal pass that "
            "builds a per-batch voxel-hit count image for the Siddon/Joseph "
            "adjoint. Reports per-voxel and 8x8x8 tile contention metrics. "
            "This is heavier than normal profiling and should not be used for "
            "baseline timing."
        ),
    )
    parser.add_argument(
        "--profile-metal-adjoint-contention",
        action="store_true",
        help=(
            "Alias for --profile-metal-adjoint-hit-diagnostics with the "
            "full-precision Joseph contention metrics emphasized in the CSV."
        ),
    )
    parser.add_argument(
        "--profile-metal-ratio-nonzero",
        action="store_true",
        help=(
            "With --profile-metal, download post-ratio Metal projection "
            "values and count active versus zero ratio entries. This perturbs "
            "timing and is intended only for deciding whether an adjoint "
            "compaction path is worth implementing."
        ),
    )
    parser.add_argument(
        "--metal-fused-ratio",
        action="store_true",
        help=(
            "Use the experimental fused Metal forward-ratio-adjoint path. "
            "Default stays on the host-ratio Metal projector path because the "
            "fused path is not yet faster on full GE listmode data."
        ),
    )
    parser.add_argument(
        "--metal-resident-images",
        action="store_true",
        help=(
            "Keep the experimental Metal OSEM image and EM update buffers "
            "resident across subsets, then download the final image at the "
            "end. This is opt-in, only applies to the host-ratio Metal path, "
            "and leaves CPU/CUDA/default Metal behavior unchanged."
        ),
    )
    parser.add_argument(
        "--metal-projector",
        choices=["siddon", "joseph"],
        default="siddon",
        help=(
            "Experimental Metal projector kernel for the OSEM opt-in path. "
            "siddon preserves the validated CPU-vs-Metal comparison path; "
            "joseph is Metal-only unless a separate reference is provided. "
            "For Joseph full-data benchmarks, current profiling favors "
            "YRTPET_METAL_USE_NATIVE_FLOAT_ATOMICS=1."
        ),
    )
    parser.add_argument(
        "--metal-joseph-forward-texture",
        action="store_true",
        help=(
            "Use a texture-backed experimental Joseph forward projection while "
            "leaving Joseph adjoint/backprojection on the existing buffer path. "
            "Requires --metal-projector joseph and "
            "--metal-sensitivity-projector joseph. Retained for A/B tests; "
            "not recommended by the current GE full-data checkpoint."
        ),
    )
    parser.add_argument(
        "--metal-joseph-sample-stride",
        type=int,
        default=1,
        help=(
            "Experimental Joseph-only reduced-update approximation. 1 keeps "
            "the validated Joseph sampling; values greater than 1 sample every "
            "Nth dominant-axis plane and widen the sample weight accordingly. "
            "Use only for A/B timing and image-quality checks."
        ),
    )
    parser.add_argument(
        "--metal-joseph-axis-specialized",
        action="store_true",
        help=(
            "Backward-compatible alias for "
            "--metal-joseph-axis-specialization both."
        ),
    )
    parser.add_argument(
        "--metal-joseph-axis-specialization",
        choices=["none", "forward", "both"],
        default="none",
        help=(
            "Experimental Joseph-only full-precision A/B path. forward splits "
            "cached Metal OSEM batches by voxel-scaled dominant axis and uses "
            "axis-specific Joseph shaders only for forward projection; both "
            "also uses axis-specific Joseph adjoint/backprojection. Default "
            "keeps the single generic Joseph shader path."
        ),
    )
    parser.add_argument(
        "--metal-joseph-adjoint-accumulation",
        choices=["none", "threadgroup-sample"],
        default=os.environ.get("YRTPET_METAL_JOSEPH_ADJOINT_ACCUMULATION", "none"),
        help=(
            "Joseph-only full-precision adjoint accumulation experiment. none "
            "preserves the current native/global atomic path; "
            "threadgroup-sample uses a small per-sample threadgroup hash table "
            "before flushing to global atomics. Requires native float atomics "
            "and is intended only for A/B projector timing."
        ),
    )
    parser.add_argument(
        "--metal-sensitivity-projector",
        choices=["siddon", "joseph"],
        default="siddon",
        help=(
            "Projector used to build the sensitivity image. siddon preserves "
            "the existing CPU path; joseph uses experimental Metal Joseph "
            "backprojection of the same correction histogram and is intended "
            "for Metal-only Joseph tests."
        ),
    )
    parser.add_argument(
        "--no-metal-cache",
        action="store_true",
        help="Disable experimental Metal LOR/batch caching.",
    )
    parser.add_argument(
        "--metal-lazy-corrections",
        action="store_true",
        help=(
            "Experimental Metal-only path that skips full-listmode correction "
            "precompute and computes randoms/scatter/sensitivity factors while "
            "packing each ratio batch."
        ),
    )
    parser.add_argument(
        "--metal-cached-corrections",
        action="store_true",
        help=(
            "Experimental Metal-only path that skips full-listmode correction "
            "precompute, stores compact correction values with each cached "
            "Metal batch, and reuses them across iterations. Uncached batches "
            "fall back to the lazy correction path."
        ),
    )
    parser.add_argument(
        "--metal-cache-budget-mb",
        default="1024.0",
        help=(
            "Maximum persistent cache budget in MB for experimental Metal "
            "LOR/batch data. Accepts one value or a comma-separated sweep "
            "list. Subsets larger than the budget are partly cached by batch "
            "and partly streamed."
        ),
    )
    parser.add_argument(
        "--metal-correction-cache-reserve-mb",
        default="0.0",
        help=(
            "Reserve this many MB of the experimental Metal cache for compact "
            "OSEM correction terms when --metal-cached-corrections is enabled. "
            "Accepts one value or a comma-separated sweep list; 0 preserves "
            "the default greedy full-batch cache admission."
        ),
    )
    parser.add_argument(
        "--metal-cache-pressure-cap-mb",
        type=float,
        default=DEFAULT_METAL_CACHE_PRESSURE_CAP_MB,
        help=(
            "Conservative local cache size cap used only for pre-run warnings "
            "and CSV risk labels. It does not change the Metal cache budget."
        ),
    )
    parser.add_argument(
        "--metal-batch-events",
        default="1000000",
        help=(
            "Maximum event count per uncached Metal projector batch. Accepts "
            "one value or a comma-separated sweep list; 0 means process a "
            "full subset at once."
        ),
    )
    parser.add_argument(
        "--metal-threads-per-threadgroup",
        default=os.environ.get("YRTPET_METAL_THREADS_PER_THREADGROUP", "0"),
        help=(
            "Override Metal compute threads per threadgroup for experimental "
            "projector benchmarks. Accepts one non-negative value or a "
            "comma-separated sweep list; 0 keeps the backend auto choice."
        ),
    )
    parser.add_argument(
        "--metal-adjoint-event-order",
        choices=["none", "major-axis", "line-hash", "tile-round-robin"],
        default=os.environ.get("YRTPET_METAL_ADJOINT_EVENT_ORDER", "none"),
        help=(
            "Experimental Metal batch ordering probe for adjoint atomic "
            "contention. none preserves the current order; major-axis uses a "
            "stable three-bucket grouping by dominant ray axis and is the "
            "intended full-data probe; line-hash pseudo-randomizes LOR order "
            "with a full sort and should be used only for small diagnostics; "
            "tile-round-robin interleaves coarse representative image tiles "
            "with O(n) host work. "
            "CPU/CUDA behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--metal-adjoint-tile-size",
        type=int,
        default=int(os.environ.get("YRTPET_METAL_ADJOINT_TILE_SIZE", "8")),
        help=(
            "Voxel edge length for --metal-adjoint-event-order "
            "tile-round-robin. Ignored by other order modes."
        ),
    )
    parser.add_argument(
        "--metal-chunk-events",
        dest="metal_batch_events",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
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
        "--sweep-subsets",
        default="",
        help="Comma-separated --subsets values to run in one process.",
    )
    parser.add_argument(
        "--isolated-sweep",
        action="store_true",
        help=(
            "Run each sweep case in a fresh Python process. This is slower but "
            "avoids cumulative Metal/Python resource pressure in full-data "
            "benchmark sweeps."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional CSV path for single-run or sweep summary rows.",
    )
    parser.add_argument(
        "--metal-subset-profile-csv",
        default="",
        help=(
            "Optional CSV path for per-subset experimental Metal timing and "
            "memory-pressure telemetry. If omitted, --summary-csv writes a "
            "'_subsets' sidecar when --profile-metal records are available."
        ),
    )
    parser.add_argument(
        "--validate-metrics",
        action="store_true",
        help="Fail if metric thresholds are exceeded.",
    )
    parser.add_argument(
        "--max-rel-l2",
        type=float,
        default=None,
        help="Validation limit for relative L2 image error.",
    )
    parser.add_argument(
        "--max-nrmse",
        type=float,
        default=None,
        help="Validation limit for normalized RMSE.",
    )
    parser.add_argument(
        "--max-sum-rel-diff",
        type=float,
        default=None,
        help="Validation limit for absolute relative image-sum difference.",
    )
    parser.add_argument(
        "--max-mismatch-fraction",
        type=float,
        default=None,
        help="Optional validation limit for exact-tolerance mismatch fraction.",
    )
    parser.add_argument(
        "--no-write-images",
        action="store_true",
        help="Skip writing sensitivity/CPU/Metal NIfTI outputs.",
    )
    parser.add_argument(
        "--export-lors-csv",
        default="",
        help="Optional CSV path for exporting selected real-data LORs.",
    )
    parser.add_argument(
        "--export-lors-limit",
        type=int,
        default=0,
        help="Maximum number of LORs to export; 0 exports all selected events.",
    )
    parser.add_argument(
        "--export-lors-stride",
        type=int,
        default=1,
        help="Export every Nth selected LOR.",
    )
    parser.add_argument(
        "--export-lors-value",
        type=float,
        default=1.0,
        help="Projection value written for each exported LOR.",
    )
    parser.add_argument(
        "--export-lors-only",
        action="store_true",
        help="Export LORs and exit before sensitivity/OSEM setup.",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default="")
    return parser.parse_args()


def enforce_metal_safety(args, used_events):
    if not (args.compare_metal or args.metal_only):
        return
    mode_name = "--metal-only" if args.metal_only else "--compare-metal"
    if args.psf:
        raise SystemExit(
            "The current experimental Metal OSEM projector path cannot run with "
            f"image PSF enabled. Re-run with {mode_name} --no-psf."
        )
    if args.allow_unsafe_metal:
        print(
            "WARNING: --allow-unsafe-metal bypasses safeguards for the "
            "experimental Metal backprojector. Scale this path gradually."
        )
        return
    if args.iterations != 1 or args.subsets != 1:
        raise SystemExit(
            f"Refusing {mode_name} with more than one iteration/subset. The "
            "current Metal adjoint is experimental and should be scaled from "
            "single-iteration/single-subset repo-side tests first. Use "
            "--iterations 1 --subsets 1 for Metal smoke comparisons."
        )
    if used_events > args.metal_event_limit:
        raise SystemExit(
            f"Refusing {mode_name} for {used_events} events. The default "
            f"safety limit is {args.metal_event_limit}. Try --max-events "
            f"{args.metal_event_limit} --iterations 1 --subsets 1, or pass "
            "--allow-unsafe-metal only for intentional repo-side stress tests."
        )


def apply_validation_profile(args):
    if args.validation_profile == "":
        return
    if args.validation_profile == "ge-mini-hotspot-siddon-4k-smoke":
        args.compare_metal = True
        args.metal_only = False
        args.psf = False
        args.metal_projector = "siddon"
        args.metal_sensitivity_projector = "siddon"
        args.max_events = 4096
        args.iterations = 1
        args.subsets = 1
        args.move_sensitivity = False
        args.validate_metrics = True
        args.fail_on_mismatch = False
        if args.max_rel_l2 is None:
            args.max_rel_l2 = 5.0e-4
        if args.max_nrmse is None:
            args.max_nrmse = 5.0e-4
        if args.max_sum_rel_diff is None:
            args.max_sum_rel_diff = 1.0e-5
        if args.max_mismatch_fraction is None:
            args.max_mismatch_fraction = 1.0e-5
        return
    if args.validation_profile != "ge-mini-hotspot-1m-10it":
        raise SystemExit(f"Unknown validation profile: {args.validation_profile}")

    args.compare_metal = True
    args.psf = False
    args.max_events = 1000000
    args.iterations = 10
    args.subsets = 1
    args.validate_metrics = True
    if args.max_rel_l2 is None:
        args.max_rel_l2 = 5.0e-3
    if args.max_nrmse is None:
        args.max_nrmse = 5.0e-3
    if args.max_sum_rel_diff is None:
        args.max_sum_rel_diff = 1.0e-4


def validate_metric_args(args):
    if args.profile_metal_adjoint_contention and not args.profile_metal:
        raise SystemExit("--profile-metal-adjoint-contention requires --profile-metal")
    if args.profile_metal_adjoint_contention:
        args.profile_metal_adjoint_hit_diagnostics = True
    if args.profile_metal and not (args.compare_metal or args.metal_only):
        raise SystemExit("--profile-metal requires --compare-metal or --metal-only")
    if args.profile_metal_ratio_nonzero and not args.profile_metal:
        raise SystemExit("--profile-metal-ratio-nonzero requires --profile-metal")
    if args.profile_metal_adjoint_diagnostics and not args.profile_metal:
        raise SystemExit(
            "--profile-metal-adjoint-diagnostics requires --profile-metal"
        )
    if args.profile_metal_adjoint_hit_diagnostics and not args.profile_metal:
        raise SystemExit(
            "--profile-metal-adjoint-hit-diagnostics requires --profile-metal"
        )
    if args.metal_resident_images and not (args.compare_metal or args.metal_only):
        raise SystemExit("--metal-resident-images requires --compare-metal or --metal-only")
    if args.metal_resident_images and args.metal_fused_ratio:
        raise SystemExit("--metal-resident-images currently requires the host-ratio path")
    if args.validate_metrics and not args.compare_metal:
        raise SystemExit("--validate-metrics requires --compare-metal")
    if args.metal_only and args.compare_metal:
        raise SystemExit("--metal-only and --compare-metal are mutually exclusive")
    if args.metal_only and args.validate_metrics:
        raise SystemExit("--metal-only cannot validate CPU-vs-Metal metrics")
    if args.metal_joseph_forward_texture and args.metal_projector != "joseph":
        raise SystemExit(
            "--metal-joseph-forward-texture requires --metal-projector joseph"
        )
    if args.metal_joseph_sample_stride < 1:
        raise SystemExit("--metal-joseph-sample-stride must be positive")
    if args.metal_joseph_sample_stride > 1 and args.metal_projector != "joseph":
        raise SystemExit(
            "--metal-joseph-sample-stride greater than 1 requires "
            "--metal-projector joseph"
        )
    axis_mode = metal_joseph_axis_specialization_mode(args)
    if axis_mode != "none":
        if args.metal_projector != "joseph":
            raise SystemExit(
                "--metal-joseph-axis-specialization requires "
                "--metal-projector joseph"
            )
        if args.metal_joseph_forward_texture:
            raise SystemExit(
                "--metal-joseph-axis-specialization is for the buffer-backed "
                "Joseph path; do not combine it with "
                "--metal-joseph-forward-texture"
            )
        if args.metal_joseph_sample_stride != 1:
            raise SystemExit(
                "--metal-joseph-axis-specialization currently requires "
                "--metal-joseph-sample-stride 1"
            )
        if args.metal_joseph_adjoint_accumulation != "none":
            raise SystemExit(
                "--metal-joseph-axis-specialization currently requires "
                "--metal-joseph-adjoint-accumulation none"
            )
    if args.metal_adjoint_tile_size < 1:
        raise SystemExit("--metal-adjoint-tile-size must be positive")
    if args.metal_joseph_adjoint_accumulation != "none":
        if args.metal_projector != "joseph":
            raise SystemExit(
                "--metal-joseph-adjoint-accumulation requires "
                "--metal-projector joseph"
            )
        if args.metal_joseph_sample_stride != 1:
            raise SystemExit(
                "--metal-joseph-adjoint-accumulation threadgroup-sample "
                "currently requires --metal-joseph-sample-stride 1"
            )
        if not env_flag("YRTPET_METAL_USE_NATIVE_FLOAT_ATOMICS"):
            raise SystemExit(
                "--metal-joseph-adjoint-accumulation threadgroup-sample "
                "requires YRTPET_METAL_USE_NATIVE_FLOAT_ATOMICS=1"
            )
    if args.compare_metal and args.metal_projector != "siddon":
        raise SystemExit(
            "--compare-metal requires --metal-projector siddon because the CPU "
            "reference path does not have a Joseph projector"
        )
    if args.metal_sensitivity_projector != "siddon":
        if not args.metal_only:
            raise SystemExit(
                "--metal-sensitivity-projector joseph is currently allowed only "
                "with --metal-only because the CPU reference path does not have "
                "a Joseph sensitivity generator"
            )
        if args.metal_projector != args.metal_sensitivity_projector:
            raise SystemExit(
                "--metal-sensitivity-projector joseph requires "
                "--metal-projector joseph so the OSEM numerator and denominator "
                "use the same experimental projector family"
            )
    if (
        args.metal_joseph_forward_texture
        and args.metal_sensitivity_projector != "joseph"
    ):
        raise SystemExit(
            "--metal-joseph-forward-texture requires "
            "--metal-sensitivity-projector joseph"
        )
    if args.metal_batch_events < 0:
        raise SystemExit("--metal-batch-events must be non-negative")
    if args.metal_cached_corrections and args.no_metal_cache:
        raise SystemExit(
            "--metal-cached-corrections requires the experimental Metal cache"
        )
    if args.metal_cache_pressure_cap_mb < 0.0:
        raise SystemExit("--metal-cache-pressure-cap-mb must be non-negative")
    for name in (
        "max_rel_l2",
        "max_nrmse",
        "max_sum_rel_diff",
        "max_mismatch_fraction",
    ):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            option = "--" + name.replace("_", "-")
            raise SystemExit(f"{option} must be non-negative")


def default_output_dir(args):
    return os.path.join(
        args.base,
        f"out_mini_hot_spot_{args.motion}_DD_False_PSF_{args.psf}",
    )


def run_case(args, out_dir, write_images, case_label="", emit_pass=True):
    apply_metal_environment(args)
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
    print(f"move_sensitivity={args.motion and args.move_sensitivity}")
    print(f"psf={args.psf}")
    print(f"compare_metal={args.compare_metal}")
    print(f"metal_only={args.metal_only}")
    print(f"metal_projector={args.metal_projector}")
    print(f"metal_joseph_forward_texture={args.metal_joseph_forward_texture}")
    print(f"metal_joseph_axis_specialization={metal_joseph_axis_specialization_mode(args)}")
    print(f"metal_joseph_sample_stride={args.metal_joseph_sample_stride}")
    print(
        "metal_joseph_adjoint_accumulation="
        f"{args.metal_joseph_adjoint_accumulation}"
    )
    print(f"metal_profile_ratio_nonzero={args.profile_metal_ratio_nonzero}")
    print(f"metal_adjoint_event_order={args.metal_adjoint_event_order}")
    print(f"metal_adjoint_tile_size={args.metal_adjoint_tile_size}")
    print(f"metal_threads_per_threadgroup={args.metal_threads_per_threadgroup}")
    print(f"metal_resident_images={args.metal_resident_images}")
    print(f"metal_sensitivity_projector={args.metal_sensitivity_projector}")
    print(f"metal_lazy_corrections={args.metal_lazy_corrections}")
    print(f"metal_cached_corrections={args.metal_cached_corrections}")
    print(
        "metal_correction_cache_reserve_mb="
        f"{args.metal_correction_cache_reserve_mb}"
    )
    cache_plan = metal_cache_plan(args, used_events)
    print_metal_cache_plan(cache_plan, args)
    print_metal_projector_notes(args)

    row = {
        "case": case_label,
        "total_events": total_events,
        "used_events": used_events,
        "iterations": args.iterations,
        "subsets": args.subsets,
        "move_sensitivity": args.motion and args.move_sensitivity,
        "metal_cache_budget_mb": args.metal_cache_budget_mb,
        "metal_correction_cache_reserve_mb": (
            args.metal_correction_cache_reserve_mb
        ),
        "metal_batch_events": args.metal_batch_events,
        "metal_threads_per_threadgroup": args.metal_threads_per_threadgroup,
        "metal_fused_ratio": args.metal_fused_ratio,
        "metal_profile_adjoint_diagnostics": (
            args.profile_metal_adjoint_diagnostics
        ),
        "metal_profile_adjoint_hit_diagnostics": (
            args.profile_metal_adjoint_hit_diagnostics
        ),
        "metal_profile_adjoint_contention": (
            args.profile_metal_adjoint_contention
        ),
        "metal_profile_ratio_nonzero": args.profile_metal_ratio_nonzero,
        "metal_adjoint_event_order": args.metal_adjoint_event_order,
        "metal_adjoint_tile_size": args.metal_adjoint_tile_size,
        "metal_resident_images": args.metal_resident_images,
        "metal_projector": args.metal_projector,
        "metal_joseph_forward_texture": args.metal_joseph_forward_texture,
        "metal_joseph_axis_specialized": (
            metal_joseph_axis_specialization_mode(args) != "none"
        ),
        "metal_joseph_axis_specialization": (
            metal_joseph_axis_specialization_mode(args)
        ),
        "metal_joseph_sample_stride": args.metal_joseph_sample_stride,
        "metal_joseph_adjoint_accumulation": (
            args.metal_joseph_adjoint_accumulation
        ),
        "metal_sensitivity_projector": args.metal_sensitivity_projector,
        "metal_lazy_corrections": args.metal_lazy_corrections,
        "metal_cached_corrections": args.metal_cached_corrections,
        "metal_native_float_atomics": env_flag(
            "YRTPET_METAL_USE_NATIVE_FLOAT_ATOMICS"
        ),
        "metal_private_update_buffer": env_flag(
            "YRTPET_METAL_USE_PRIVATE_UPDATE_BUFFER"
        ),
        "metal_only": args.metal_only,
    }
    row.update(cache_plan)
    setup_start = time.perf_counter()
    dataset, timestamps, detector1, detector2, listmode_plan = load_listmode_dataset(
        scanner, listmode_path, total_events, used_events, args
    )
    row.update(listmode_plan)

    lor_motion = None
    if args.motion:
        lor_motion = yrt.LORMotion(os.path.join(args.base, "Motion.yrt"))
        dataset.addLORMotion(lor_motion)

    if args.export_lors_csv:
        exported = export_lors_csv(
            dataset,
            args.export_lors_csv,
            args.export_lors_limit,
            args.export_lors_stride,
            args.export_lors_value,
        )
        row["exported_lors"] = exported
        print(f"exported_lors_csv={args.export_lors_csv}")
        print(f"exported_lors={exported}")
        if args.export_lors_only:
            row["setup_s"] = time.perf_counter() - setup_start
            if emit_pass:
                print("PASS")
            return row

    histo_corr, histo_scatter, histo_randoms, corr_np, histo_acf, histo_norm = (
        load_histos(scanner, config_dir, args.base)
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
        args.motion,
        args.move_sensitivity,
        args.psf,
        args.global_scale_factor,
        args.metal_sensitivity_projector,
    )
    if write_images:
        sensitivity.writeToFile(os.path.join(out_dir, "sens_img.nii.gz"))
    setup_elapsed = time.perf_counter() - setup_start

    if args.metal_only:
        print("Starting experimental Metal reconstruction...")
        (
            metal_recon,
            metal_image,
            metal_elapsed,
            metal_profile,
            metal_subset_profile,
        ) = run_osem(
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
        for subset_row in metal_subset_profile:
            subset_row["case"] = case_label
        if write_images:
            metal_image.writeToFile(os.path.join(out_dir, "metal_osem.nii.gz"))
        metal_stats = image_stats(metal_image)
        row.update(
            {
                "setup_s": setup_elapsed,
                "metal_recon_s": metal_elapsed,
                "metal_projector_ran": metal_recon.didLastExperimentalMetalProjectorRun(),
                "metal_image_sum": metal_stats["sum"],
                "metal_image_max": metal_stats["max"],
                "metal_image_nonzero": metal_stats["nonzero"],
            }
        )
        row.update(metal_profile)
        summarize_metal_subset_profiles(row, metal_subset_profile)
        if metal_subset_profile:
            row["_metal_subset_profile"] = metal_subset_profile
        update_actual_metal_cache_usage(row, metal_profile)
        update_metal_profile_recon_gap(row)
        print(
            "mode,setup_s,recon_s,metal_projector_ran,image_sum,image_max,"
            "image_nonzero"
        )
        print(
            f"metal,{setup_elapsed:.6f},{metal_elapsed:.6f},"
            f"{metal_recon.didLastExperimentalMetalProjectorRun()},"
            f"{metal_stats['sum']:.9g},{metal_stats['max']:.9g},"
            f"{metal_stats['nonzero']}"
        )
        print_metal_profile(metal_profile)
        print_metal_subset_profile(metal_subset_profile)
        warn_if_requested_adjoint_diagnostics_missing(args, metal_profile)
        if not metal_recon.didLastExperimentalMetalProjectorRun():
            raise SystemExit("Metal projector path did not run")
        if emit_pass:
            print("PASS")
        return row

    print("Starting CPU reconstruction...")
    cpu_recon, cpu_image, cpu_elapsed, _cpu_profile, _cpu_subset_profile = run_osem(
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
    row.update(
        {
            "setup_s": setup_elapsed,
            "cpu_recon_s": cpu_elapsed,
            "cpu_image_sum": cpu_stats["sum"],
            "cpu_image_max": cpu_stats["max"],
            "cpu_image_nonzero": cpu_stats["nonzero"],
        }
    )

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
    (
        metal_recon,
        metal_image,
        metal_elapsed,
        metal_profile,
        metal_subset_profile,
    ) = run_osem(
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
    for subset_row in metal_subset_profile:
        subset_row["case"] = case_label
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
    sum_scale = max(abs(cpu_stats["sum"]), 1.0e-12)
    sum_rel_diff = abs(metal_stats["sum"] - cpu_stats["sum"]) / sum_scale
    row.update(
        {
            "metal_recon_s": metal_elapsed,
            "metal_over_cpu": metal_over_cpu,
            "metal_projector_ran": metal_recon.didLastExperimentalMetalProjectorRun(),
            "metal_image_sum": metal_stats["sum"],
            "metal_image_max": metal_stats["max"],
            "metal_image_nonzero": metal_stats["nonzero"],
            "sum_rel_diff": sum_rel_diff,
        }
    )
    row.update(metal_profile)
    summarize_metal_subset_profiles(row, metal_subset_profile)
    if metal_subset_profile:
        row["_metal_subset_profile"] = metal_subset_profile
    update_actual_metal_cache_usage(row, metal_profile)
    update_metal_profile_recon_gap(row)
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
    print_metal_profile(metal_profile)
    print_metal_subset_profile(metal_subset_profile)
    warn_if_requested_adjoint_diagnostics_missing(args, metal_profile)
    print_diff_diagnostics(diff_stats)

    if not metal_recon.didLastExperimentalMetalProjectorRun():
        raise SystemExit("Metal projector path did not run")
    apply_metric_validation(row, args)
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
    apply_validation_profile(args)
    cache_budget_values = parse_nonnegative_float_list(
        args.metal_cache_budget_mb, "--metal-cache-budget-mb"
    )
    args.metal_cache_budget_mb = cache_budget_values[0]
    args.metal_cache_budget_values = cache_budget_values
    correction_reserve_values = parse_nonnegative_float_list(
        args.metal_correction_cache_reserve_mb,
        "--metal-correction-cache-reserve-mb",
    )
    args.metal_correction_cache_reserve_mb = correction_reserve_values[0]
    args.metal_correction_cache_reserve_values = correction_reserve_values
    batch_event_values = parse_nonnegative_int_list(
        args.metal_batch_events, "--metal-batch-events"
    )
    args.metal_batch_events = batch_event_values[0]
    args.metal_batch_event_values = batch_event_values
    threadgroup_values = parse_nonnegative_int_list(
        args.metal_threads_per_threadgroup, "--metal-threads-per-threadgroup"
    )
    args.metal_threads_per_threadgroup = threadgroup_values[0]
    args.metal_threads_per_threadgroup_values = threadgroup_values
    validate_metric_args(args)
    apply_metal_environment(args)
    if args.threads > 0:
        yrt.setNumThreads(args.threads)

    sweep_events = parse_positive_int_list(args.sweep_events, "--sweep-events")
    sweep_iterations = parse_positive_int_list(
        args.sweep_iterations, "--sweep-iterations"
    )
    sweep_subsets = parse_positive_int_list(args.sweep_subsets, "--sweep-subsets")
    is_sweep = bool(
        sweep_events
        or sweep_iterations
        or sweep_subsets
        or len(batch_event_values) > 1
        or len(cache_budget_values) > 1
        or len(correction_reserve_values) > 1
        or len(threadgroup_values) > 1
    )
    event_values = sweep_events or [args.max_events]
    iteration_values = sweep_iterations or [args.iterations]
    subset_values = sweep_subsets or [args.subsets]
    rows = []

    if is_sweep:
        base_out_dir = args.out_dir or os.path.join(args.base, "out_metal_ge_osem_sweep")
        print(
            "sweep_plan="
            f"events:{','.join(str(value) for value in event_values)};"
            f"iterations:{','.join(str(value) for value in iteration_values)};"
            f"subsets:{','.join(str(value) for value in subset_values)};"
            "metal_batch_events:"
            f"{','.join(str(value) for value in batch_event_values)};"
            "metal_cache_budget_mb:"
            f"{','.join(f'{value:g}' for value in cache_budget_values)};"
            "metal_correction_cache_reserve_mb:"
            f"{','.join(f'{value:g}' for value in correction_reserve_values)};"
            "metal_threads_per_threadgroup:"
            f"{','.join(str(value) for value in threadgroup_values)}",
            flush=True,
        )
        if args.isolated_sweep:
            rows = run_isolated_sweep(
                args,
                event_values,
                iteration_values,
                subset_values,
                batch_event_values,
                cache_budget_values,
                correction_reserve_values,
                threadgroup_values,
                base_out_dir,
            )
        else:
            for max_events in event_values:
                for iterations in iteration_values:
                    for subsets in subset_values:
                        for batch_events in batch_event_values:
                            for cache_budget_mb in cache_budget_values:
                                for correction_reserve_mb in correction_reserve_values:
                                    for threads_per_threadgroup in threadgroup_values:
                                        case_args = copy.copy(args)
                                        case_args.max_events = max_events
                                        case_args.iterations = iterations
                                        case_args.subsets = subsets
                                        case_args.metal_batch_events = batch_events
                                        case_args.metal_cache_budget_mb = (
                                            cache_budget_mb
                                        )
                                        case_args.metal_correction_cache_reserve_mb = (
                                            correction_reserve_mb
                                        )
                                        case_args.metal_threads_per_threadgroup = (
                                            threads_per_threadgroup
                                        )
                                        case_label = (
                                            f"events_{max_events}_iters_{iterations}"
                                            f"_subsets_{subsets}_batch_{batch_events}"
                                            f"_cachemb_"
                                            f"{format_cache_budget_label(cache_budget_mb)}"
                                            f"_corrmb_"
                                            f"{format_cache_budget_label(correction_reserve_mb)}"
                                            f"_tpg_{threads_per_threadgroup}"
                                        )
                                        case_out_dir = os.path.join(
                                            base_out_dir, case_label
                                        )
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
        subset_csv = (
            args.metal_subset_profile_csv
            or default_metal_subset_profile_csv_path(args.summary_csv)
        )
        if subset_csv and write_metal_subset_profile_csv(subset_csv, rows):
            print(f"metal_subset_profile_csv={subset_csv}")
        finish_metric_validation(rows, args)
        print("PASS")
        return

    out_dir = args.out_dir or default_output_dir(args)
    rows.append(
        run_case(
            args,
            out_dir,
            write_images=not args.no_write_images,
            emit_pass=not args.validate_metrics,
        )
    )
    if args.summary_csv:
        write_summary_csv(args.summary_csv, rows)
        print(f"summary_csv={args.summary_csv}")
    subset_csv = (
        args.metal_subset_profile_csv
        or default_metal_subset_profile_csv_path(args.summary_csv)
    )
    if subset_csv and write_metal_subset_profile_csv(subset_csv, rows):
        print(f"metal_subset_profile_csv={subset_csv}")
    finish_metric_validation(rows, args)
    if args.validate_metrics:
        print("PASS")


if __name__ == "__main__":
    main()
