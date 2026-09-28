# Projection-Based PSF File Format

Projection-space PSF models a sampled 1D transverse response in the
Distance-Driven (DD) projector on CPU and GPU. Kernels may vary with the
radial position of the LOR. This model uses a different CSV format from
[image-space PSF](imagepsf_file.md).

## CSV Format

Use a single numeric CSV row with the following layout:

```text
s_step,k_spacing,kernel_size,kernel_0_values...,kernel_1_values...,...
```

| Field | Meaning |
|-------|---------|
| `s_step` | Positive spacing in mm between radial kernel-selection bins |
| `k_spacing` | Positive spacing in mm between samples within a kernel |
| `kernel_size` | Positive odd integer specifying the number of samples in every kernel |
| Remaining values | One or more kernels, concatenated in increasing radial-bin order |

Provide exactly `kernel_size` values per kernel, without a text header or
extra padding. Samples are equally spaced and centered around zero, with
the middle sample at zero. The reader does not normalize coefficients;
for a nonnegative response with unit integral, use
`k_spacing * sum(kernel) = 1` for each kernel.

For each LOR, `s` is its absolute transverse distance from the image center,
accounting for the image offset. The zero-based kernel index is
`floor(s / s_step)`, capped at the last available kernel, without interpolation
between kernels. A file containing one kernel applies it to all LORs.

## Example

Two radial bins with three samples per kernel and 1 mm sample spacing:

```text
50,1,3,0.25,0.5,0.25,0.3,0.4,0.3
```

Here, the first kernel is selected for `0 <= s < 50` mm, and the second for
`s >= 50` mm. These illustrative kernels should be replaced with an
appropriate measured or fitted response.

## Usage

For the command line, add `--projector DD --proj_psf proj_psf.csv` to
`yrtpet_reconstruct`, `yrtpet_forward_project`, or `yrtpet_backproject`.
For OSEM in Python, configure the PSF before generating sensitivity images:

```python
osem.setProjector("DD")
osem.addProjPSF("proj_psf.csv")
```

Projection-space PSF may be combined with one uniform or variant image-space
PSF and with TOF. Use the same PSF configuration for sensitivity generation
and reconstruction. See [OSEM](../python/osem.md) and
[Projector](../python/projector.md) for the surrounding workflow.
