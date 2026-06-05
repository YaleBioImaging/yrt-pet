#!/usr/bin/env python3
"""Experimental PyTorch MPS to Metal smoke test.

This script is intentionally isolated from the reconstruction paths. It probes
whether a PyTorch MPS tensor can be passed to the existing Metal smoke kernel
and stay resident on MPS while the kernel runs. Current PyTorch builds may not
expose a precompiled .metallib loader, so the default path falls back to
torch.mps.compile_shader on the same SmokeKernels.metal source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_metallib() -> Path:
    return (
        _repo_root()
        / "build_metal_smoke"
        / "src"
        / "backends"
        / "metal"
        / "SmokeKernels.metallib"
    )


def _default_source() -> Path:
    return _repo_root() / "yrt-pet" / "src" / "backends" / "metal" / "SmokeKernels.metal"


def _print_probe_header(args: argparse.Namespace) -> None:
    print("PyTorch MPS / Metal smoke probe")
    print(f"torch: {torch.__version__}")
    print(f"mps built: {torch.backends.mps.is_built()}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"has torch.mps.compile_shader: {hasattr(torch.mps, 'compile_shader')}")
    print(f"has torch.mps.load_metallib: {hasattr(torch.mps, 'load_metallib')}")
    print(f"metallib: {args.metallib} (exists={args.metallib.exists()})")
    print(f"kernel source: {args.source} (exists={args.source.exists()})")


def _load_kernel_from_metallib(metallib: Path):
    load_metallib = getattr(torch.mps, "load_metallib", None)
    if load_metallib is None:
        raise RuntimeError("torch.mps.load_metallib is not available in this PyTorch build")
    if not metallib.exists():
        raise FileNotFoundError(f"Metal library not found: {metallib}")
    return load_metallib(str(metallib)), "load_metallib"


def _load_kernel_from_source(source_path: Path):
    compile_shader = getattr(torch.mps, "compile_shader", None)
    if compile_shader is None:
        raise RuntimeError("torch.mps.compile_shader is not available in this PyTorch build")
    if not source_path.exists():
        raise FileNotFoundError(f"Metal source not found: {source_path}")
    return compile_shader(source_path.read_text()), "compile_shader"


def _call_smoke_add_one(lib, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
    kernel = getattr(lib, "smoke_add_one")
    try:
        kernel(
            input_tensor,
            output_tensor,
            grid=(input_tensor.numel(), 1, 1),
            threadgroup=(min(input_tensor.numel(), 256), 1, 1),
        )
    except TypeError:
        # Older PyTorch MPS shader wrappers infer the launch geometry.
        kernel(input_tensor, output_tensor)


def run_smoke(args: argparse.Namespace) -> int:
    _print_probe_header(args)

    if not torch.backends.mps.is_available():
        try:
            torch.empty(1, device="mps")
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"ERROR: MPS tensor allocation failed: {type(exc).__name__}: {exc}")
        return 2

    load_errors: list[str] = []
    lib = None
    load_path = ""

    try:
        lib, load_path = _load_kernel_from_metallib(args.metallib)
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        load_errors.append(f"metallib path unavailable: {type(exc).__name__}: {exc}")

    if lib is None and not args.no_source_fallback:
        try:
            lib, load_path = _load_kernel_from_source(args.source)
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            load_errors.append(f"source fallback unavailable: {type(exc).__name__}: {exc}")

    if lib is None:
        for error in load_errors:
            print(f"ERROR: {error}")
        return 3

    for note in load_errors:
        print(f"NOTE: {note}")
    print(f"kernel load path: {load_path}")

    x = torch.arange(args.count, dtype=torch.float32, device="mps")
    y = torch.empty_like(x)
    _call_smoke_add_one(lib, x, y)
    torch.mps.synchronize()

    expected = x + 1.0
    if not torch.allclose(y, expected):
        print(f"ERROR: smoke_add_one produced {y.detach().cpu().tolist()}")
        print(f"expected: {expected.detach().cpu().tolist()}")
        return 4

    print(f"input device: {x.device}")
    print(f"output device: {y.device}")
    print(f"output values: {y.detach().cpu().tolist()}")
    print("PASS: smoke_add_one accepted MPS tensors and produced the expected result")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metallib", type=Path, default=_default_metallib())
    parser.add_argument("--source", type=Path, default=_default_source())
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--no-source-fallback",
        action="store_true",
        help=(
            "Require torch.mps.load_metallib and fail instead of compiling "
            "SmokeKernels.metal with torch.mps.compile_shader."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_smoke(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
