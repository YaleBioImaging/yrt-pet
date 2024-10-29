#!/bin/env python

import numpy as np
import argparse


def gaussian_1d(size, sigma, vox_size):
    """Generates a 1D Gaussian kernel."""
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (x * vox_size / sigma) ** 2)
    return kernel / kernel.sum()  # Normalize


def generate_3d_psf(size_x, sigma_x, vx, size_y, sigma_y, vy, size_z, sigma_z, vz):
    """Generates a 3D separable PSF kernel from Gaussian parameters for x, y, z axes."""
    # Generate 1D Gaussian kernels for each axis
    k_x = gaussian_1d(size_x, sigma_x, vx)
    k_y = gaussian_1d(size_y, sigma_y, vy)
    k_z = gaussian_1d(size_z, sigma_z, vz)

    K_3d = separable_to_full_psf_kernel(k_x, k_y, k_z)

    return k_x, k_y, k_z, K_3d


def separable_to_full_psf_kernel(k_x, k_y, k_z):
    K_xy = np.outer(k_x, k_y)
    K_3d = K_xy[:, :, np.newaxis] * k_z[np.newaxis, np.newaxis, :]

    return K_3d


def sigma_to_fwhm(sigma: float):
    return sigma * 2.3548200450309493


def fwhm_to_sigma(fwhm: float):
    return fwhm / 2.3548200450309493


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a separable PSF kernel with a given voxel size and a FWHM')
    parser.add_argument("--fx", dest="fx", required=True, help="FWHM in X")
    parser.add_argument("--fy", dest="fy", required=True, help="FWHM in Y")
    parser.add_argument("--fz", dest="fz", required=True, help="FWHM in Z")
    parser.add_argument("--vx", dest="vx", required=True, help="Voxel size in X")
    parser.add_argument("--vy", dest="vy", required=True, help="Voxel size in Y")
    parser.add_argument("--vz", dest="vz", required=True, help="Voxel size in Z")
    parser.add_argument("--size_x", dest="size_x", required=False, default=7,
                        help="Size of the kernel in the X dimension")
    parser.add_argument("--size_y", dest="size_y", required=False, default=7,
                        help="Size of the kernel in the X dimension")
    parser.add_argument("--size_z", dest="size_z", required=False, default=7,
                        help="Size of the kernel in the X dimension")
    parser.add_argument("-o", "--output", dest="output", required=True, help="Output CSV file")

    args = parser.parse_args()

    size_x, sigma_x = args.size_x, fwhm_to_sigma(float(args.fx))
    size_y, sigma_y = args.size_y, fwhm_to_sigma(float(args.fy))
    size_z, sigma_z = args.size_z, fwhm_to_sigma(float(args.fz))
    vx = float(args.vx)
    vy = float(args.vy)
    vz = float(args.vz)

    k_x, k_y, k_z, psf_3d = generate_3d_psf(size_x, sigma_x, vx, size_y, sigma_y, vy, size_z, sigma_z, vz)

    size = max(size_x, size_y, size_z)

    arr_to_save = np.zeros((4, size))
    arr_to_save[0, :size_x] = k_x
    arr_to_save[1, :size_y] = k_y
    arr_to_save[2, :size_z] = k_z
    arr_to_save[3, :3] = size_x, size_y, size_z

    with open(args.output, 'w') as f:
        np.savetxt(f, arr_to_save, delimiter=',')
