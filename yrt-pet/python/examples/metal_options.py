#!/usr/bin/env python3
"""Shared helpers for experimental Metal reconstruction example options."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetalRecipeOptions:
    osem_kernel: str
    sensitivity_kernel: str
    cache_max_bytes: int
    correction_cache_reserve_bytes: int
    direct_frame_batches_explicit: bool
    direct_frame_batches: bool
    native_float_atomics_explicit: bool
    native_float_atomics: bool
    joseph_adjoint_axis_switch_once_explicit: bool
    joseph_adjoint_axis_switch_once: bool
    threads_per_threadgroup_explicit: bool
    threads_per_threadgroup: int


def selected_metal_projector_kernel(args):
    return (
        "joseph_texture_forward"
        if args.metal_joseph_forward_texture
        else args.metal_projector
    )


def build_metal_recipe_options(args):
    native_float_atomics = getattr(args, "metal_native_float_atomics", None)
    cache_max_bytes = int(args.metal_cache_budget_mb * 1024.0 * 1024.0)
    correction_cache_reserve_bytes = int(
        args.metal_correction_cache_reserve_mb * 1024.0 * 1024.0
    )
    return MetalRecipeOptions(
        osem_kernel=selected_metal_projector_kernel(args),
        sensitivity_kernel=args.metal_sensitivity_projector,
        cache_max_bytes=cache_max_bytes,
        correction_cache_reserve_bytes=correction_cache_reserve_bytes,
        direct_frame_batches_explicit=True,
        direct_frame_batches=bool(args.metal_direct_frame_batches),
        native_float_atomics_explicit=native_float_atomics is not None,
        native_float_atomics=bool(native_float_atomics)
        if native_float_atomics is not None
        else False,
        joseph_adjoint_axis_switch_once_explicit=True,
        joseph_adjoint_axis_switch_once=bool(
            args.metal_joseph_adjoint_axis_switch_once
        ),
        threads_per_threadgroup_explicit=True,
        threads_per_threadgroup=int(args.metal_threads_per_threadgroup),
    )


def set_option_if_available(options, name, value):
    if hasattr(options, name):
        setattr(options, name, value)


def apply_projector_runtime_options(
    options, metal_recipe_options, include_direct_frame_batches=False
):
    if include_direct_frame_batches:
        set_option_if_available(
            options,
            "direct_frame_batches_explicit",
            metal_recipe_options.direct_frame_batches_explicit,
        )
        set_option_if_available(
            options,
            "direct_frame_batches",
            metal_recipe_options.direct_frame_batches,
        )
    if metal_recipe_options.native_float_atomics_explicit:
        set_option_if_available(
            options,
            "native_float_atomics_explicit",
            metal_recipe_options.native_float_atomics_explicit,
        )
        set_option_if_available(
            options,
            "native_float_atomics",
            metal_recipe_options.native_float_atomics,
        )
    set_option_if_available(
        options,
        "joseph_adjoint_axis_switch_once_explicit",
        metal_recipe_options.joseph_adjoint_axis_switch_once_explicit,
    )
    set_option_if_available(
        options,
        "joseph_adjoint_axis_switch_once",
        metal_recipe_options.joseph_adjoint_axis_switch_once,
    )
    set_option_if_available(
        options,
        "threads_per_threadgroup_explicit",
        metal_recipe_options.threads_per_threadgroup_explicit,
    )
    set_option_if_available(
        options,
        "threads_per_threadgroup",
        metal_recipe_options.threads_per_threadgroup,
    )
