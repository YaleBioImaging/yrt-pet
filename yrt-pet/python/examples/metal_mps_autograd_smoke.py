#!/usr/bin/env python3
"""Experimental PyTorch MPS autograd smoke test for Metal kernels.

This script is intentionally isolated from reconstruction and DIP code. It
checks the smallest useful pattern for a future DIP forward model:

    PyTorch MPS tensor -> Metal forward kernel -> PyTorch autograd backward
    -> Metal adjoint/gradient kernel -> PyTorch MPS gradient tensor
"""

from __future__ import annotations

import argparse
import sys

import torch


AUTOGRAD_SMOKE_SOURCE = """
#include <metal_stdlib>

using namespace metal;

kernel void mps_autograd_add_one_forward(device const float* input [[buffer(0)]],
                                         device float* output [[buffer(1)]],
                                         uint id [[thread_position_in_grid]])
{
    output[id] = input[id] + 1.0f;
}

kernel void mps_autograd_identity_backward(
    device const float* gradOutput [[buffer(0)]],
    device float* gradInput [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    gradInput[id] = gradOutput[id];
}
"""


def _print_probe_header() -> None:
    print("PyTorch MPS / Metal autograd smoke probe")
    print(f"torch: {torch.__version__}")
    print(f"mps built: {torch.backends.mps.is_built()}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"has torch.mps.compile_shader: {hasattr(torch.mps, 'compile_shader')}")


def _compile_library():
    compile_shader = getattr(torch.mps, "compile_shader", None)
    if compile_shader is None:
        raise RuntimeError("torch.mps.compile_shader is not available in this PyTorch build")
    return compile_shader(AUTOGRAD_SMOKE_SOURCE)


def _launch_1d(kernel, count: int, *buffers: torch.Tensor) -> None:
    try:
        kernel(
            *buffers,
            grid=(count, 1, 1),
            threadgroup=(min(count, 256), 1, 1),
        )
    except TypeError:
        # Older PyTorch MPS shader wrappers infer the launch geometry.
        kernel(*buffers)


class MetalAddOne(torch.autograd.Function):
    """Toy Metal op with an explicit Metal backward kernel."""

    _lib = None

    @staticmethod
    def _library():
        if MetalAddOne._lib is None:
            MetalAddOne._lib = _compile_library()
        return MetalAddOne._lib

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.device.type != "mps":
            raise ValueError("MetalAddOne expects an MPS tensor")
        if input_tensor.dtype != torch.float32:
            raise ValueError("MetalAddOne expects a float32 tensor")

        input_contiguous = input_tensor.contiguous()
        output = torch.empty_like(input_contiguous)
        lib = MetalAddOne._library()
        _launch_1d(
            lib.mps_autograd_add_one_forward,
            input_contiguous.numel(),
            input_contiguous,
            output,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output_contiguous = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output_contiguous)
        lib = MetalAddOne._library()
        _launch_1d(
            lib.mps_autograd_identity_backward,
            grad_output_contiguous.numel(),
            grad_output_contiguous,
            grad_input,
        )
        return grad_input


def run_smoke(args: argparse.Namespace) -> int:
    _print_probe_header()

    if not torch.backends.mps.is_available():
        try:
            torch.empty(1, device="mps")
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"ERROR: MPS tensor allocation failed: {type(exc).__name__}: {exc}")
        return 2

    try:
        x = torch.arange(args.count, dtype=torch.float32, device="mps", requires_grad=True)
        weights = torch.linspace(0.25, 1.25, args.count, dtype=torch.float32, device="mps")
        y = MetalAddOne.apply(x)
        loss = (y * weights).sum()
        loss.backward()
        torch.mps.synchronize()
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"ERROR: autograd smoke failed: {type(exc).__name__}: {exc}")
        return 3

    expected_y = x.detach() + 1.0
    expected_grad = weights

    if not torch.allclose(y, expected_y):
        print(f"ERROR: forward produced {y.detach().cpu().tolist()}")
        print(f"expected: {expected_y.detach().cpu().tolist()}")
        return 4

    if x.grad is None:
        print("ERROR: backward did not produce x.grad")
        return 5

    if not torch.allclose(x.grad, expected_grad):
        print(f"ERROR: backward produced {x.grad.detach().cpu().tolist()}")
        print(f"expected grad: {expected_grad.detach().cpu().tolist()}")
        return 6

    print(f"input device: {x.device}")
    print(f"output device: {y.device}")
    print(f"gradient device: {x.grad.device}")
    print(f"output values: {y.detach().cpu().tolist()}")
    print(f"gradient values: {x.grad.detach().cpu().tolist()}")
    print("PASS: Metal forward and backward kernels participated in PyTorch autograd on MPS")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_smoke(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
