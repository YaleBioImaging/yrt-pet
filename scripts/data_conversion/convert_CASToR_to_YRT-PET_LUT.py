#!/bin/env python
"""Convert output of castor-scannerLUTExplorer script"""

# %% Imports

import numpy as np
import tqdm
import argparse
import re

# %% Main function

def make_lut_from_LUTExplorer(fname):
    """Read detector positions into LUT for YRT-PET"""
    lut_raw = [s.strip() for s in tqdm.tqdm(open(fname, 'rt').readlines(),
                                            desc='Read input')
               if s.strip()]
    idx_start = lut_raw.index('castor-scannerLUTExplorer() -> Start global ' +
                              'exploration of all elements') + 1
    num_crystals = int(re.search(
        'Total number of crystals:\s*([0-9]+)',
        [l for l in lut_raw[:idx_start]
         if 'Total number of crystals' in l][0]).groups()[0])
    lut_rawc = lut_raw[idx_start:idx_start + num_crystals]
    re_lut = re.compile(
        r':\s*([0-9.+-e]+)\s*;\s*([0-9.+-e]+)\s*;\s*([0-9.+-e]+)' +
        r'\s*.Orientation.*:' +
        r'\s*([0-9.+-e]+)\s*;\s*([0-9.+-e]+)\s*;\s*([0-9.+-e]+)')
    lut = np.array([
        [np.float32(f) for f in re.search(re_lut, ll).groups()]
        for ll in tqdm.tqdm( lut_rawc, desc='Process detector')])
    return lut

# %% Command line interface

# Usage: Use CASToR's executable to generate the log file of all crystal detectors from the geom file:
#  castor-scannerLUTExplorer -sf <my geom file> -g -o <output log file>
# Then use this executable on the "<output log file>" generated to create the LUT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='LUT generation script for YRT-PET')
    parser.add_argument('-i', '--input', dest='input_file', type=str,
                        required=True,
                        help='Input file (output of ' +
                        'castor-scannerLUTExplorer)')
    parser.add_argument('-o', '--output', dest='output_file', type=str,
                        required=True, help='Output LUT file')

    args_p = parser.parse_args()

    lut = make_lut_from_LUTExplorer(args_p.input_file)

    lut.tofile(args_p.output_file)

