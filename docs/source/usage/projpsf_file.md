# Projection-Based PSF File Format

Projection-space PSF models a sampled 1D response in the Distance-Driven
(DD) projector. It is available on CPU and GPU. It is separate from
[image-space PSF](imagepsf_file.md); the single and dual Gaussian variant LUT
formats must not be passed to the projection-space reader.

## CSV Format

Use a single numeric CSV row with the following layout:

```text
s_step,k_spacing,kernel_size,kernel_0_values...,kernel_1_values...,...
```

| Field | Meaning |
|-------|---------|
| `s_step` | Positive spacing in mm between radial kernel-selection bins |
| `k_spacing` | Positive spacing in mm between samples within a kernel |
| `kernel_size` | Positive odd number of samples in every kernel |
| Remaining values | One or more kernels, concatenated in increasing radial-bin order |

The number of values after the first three must be an exact multiple of
`kernel_size`. Do not add a text header, split kernels onto separate rows, or
pad the row with extra zeros. The reader takes the metadata and kernels from
the first row.

For each LOR, let `s` be its absolute transverse distance from the scanner
origin. The selected kernel index is `floor(s / s_step)`, capped at the last
available kernel. Thus a single kernel applies to all LORs. This is radial
bin selection, not the X/Y/Z nearest-neighbor lookup of image-space PSF.

## Example

A single three-sample kernel with 1 mm sample spacing can be written as:

```text
50,1,3,0.25,0.5,0.25
```

Two radial bins with three samples each can be written as:

```text
50,1,3,0.25,0.5,0.25,0.3,0.4,0.3
```

Here, the first kernel is selected for `0 <= s < 50` mm, and the second for
`s >= 50` mm. These illustrative kernels should be replaced with an
appropriate measured or fitted response.

Samples are centered around zero at offsets
`(j - (kernel_size - 1) / 2) * k_spacing`. The projector integrates a linearly
interpolated response with one zero sample beyond each end. The support
half-width is `(kernel_size + 1) / 2 * k_spacing` mm. The reader does not
normalize coefficients. For a nonnegative response with unit integral under
this interpolation, normalize so that `k_spacing * sum(kernel) = 1`.
The adjoint uses the reversed kernel.

## Usage

For the command line, add `--projector DD --proj_psf proj_psf.csv` to
`yrtpet_reconstruct`, `yrtpet_forward_project`, or `yrtpet_backproject`.
For OSEM in Python, configure the PSF before generating sensitivity images:

```python
osem.setProjector("DD")
osem.addProjPSF("proj_psf.csv")
```

For direct projection, set `proj_params.projPsf_fname = "proj_psf.csv"` before
constructing the projector, or call `addProjPSF("proj_psf.csv")` on the DD
projector or its `OperatorProjector` before applying it. The GPU projection
operator provides the same configuration method.

Projection-space PSF may be combined with one uniform or variant image-space
PSF and with TOF. Use the same PSF configuration for sensitivity generation
and reconstruction. See [OSEM](../python/osem.md) and
[Projector](../python/projector.md) for the surrounding workflow.
