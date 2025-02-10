# Frequenty asked questions

- I have a NIfTI image and I want to generate the corresponding image parameters
  JSON file
    - With Python, this one-liner:
      ``yrt.ImageOwned("my_image.nii").getParams().serialize("my_params.json")``
      - Without Python, this can be done manually quite easily.
        The only caveat is that the image parameters file needs to specify
        an *offset*, which is the position of the center of the image.
        This is different from the *origin* of the image, which is the physical
        position of the first voxel.
        - $`p_{0_x} = o_x - \frac{l_x}{2} + \frac{v_x}{2}`$
        where $`p_{0_x}`$ is the origin, $`o_x`$ is the image offset,
        $`l_x`$ is the image length and $`v_x`$ is the voxel size in the
        X dimension. The same equation applies in the Y and Z dimensions

Relating to CASToR:
- I have scanner parameters from CASToR and want to use them in YRT-PET
  - For the Look-Up-Table:
    - CASToR provides a tool (`castor-scannerLUTExplorer`) that allows one to
    display each detecting element
    - The following command line will generate a log file containing a *text*
    description of each detector:
      ``castor-scannerLUTExplorer -sf <my geom file> -g -o <output log file>``
    - Then, the script in
    `scripts/data_conversion/convert_CASToR_to_YRT-PET_LUT.py` takes that log
    file and converts it into a YRT-PET-ready Look-Up-Table.
  - For the Scanner parameters JSON file:
    - Most of the scanner parameters can be determined manually.
    Though some caveats are worth mentioning:
      - Note that the minimum angle difference (named `minAngDiff` in
      YRT-PET's JSON file and `min angle difference` in CASToR's hscan file)
      is described in YRT-PET in terms of *number of detector elements* while
      in CASToR it is described in degrees.
    - Similarly, the maximum ring difference `maxRingDiff` property is defined
    in terms of the number of rings.
- I have a List-mode file in the CASToR format, how do I convert it in the
  YRT-PET default ListMode format (`LM`) ?
  - When there is no Time-of-Flight (TOF) and no Attenuation correction, the
  CASToR list-mode datatype is equivalent to YRT-PET's.
  - When there is TOF in the List-mode, CASToR's structure encodes the TOF
  slightly differently. The script in
  `scripts/convert_CASToR_to_YRT-PET_list-mode.py` does the conversion.
    - Note that the script does not handle normalisation information
    (CASToR 3.2).
    - YRT-PET's default List-Mode format does not encode as many fields as
    CASToR's, so a one-to-one equivalence is not to be expected.
- My images seem to be flipped in the Z direction compared to CASToR...
  - This might be because CASToR uses a left-handed coordinate system while
  YRT-PET uses a right-handed coordinate system.
