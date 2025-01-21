#!/bin/env python
"""This script converts castor list mode data to YRT-PET format

When no TOF information is present in the castor list mode this script simply
converts <uint32><uint32><uint32> to <float32><uint32><uint32>

"""

# %% Imports

import os
import argparse
import struct
import tqdm
import numpy as np

# %% Helper functions

def parse_header(fname):
    # Load header
    with open(fname, 'rt') as fid:
        header_lines = [s.strip() for s in fid.readlines()
                        if s.strip()]
    # Parse header
    header = {}
    for line in header_lines:
        line_s = [s.strip() for s in line.split(':')]
        header[line_s[0]] = ':'.join(line_s[1:])
    return header


# %% Command line interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Converts a CASToR ListMode file (uint32-uint32-uint32) ' +
        'to a YRT-PET ListMode file (float32-uint32-uint32)')
    parser.add_argument('-i', '--input', dest='input_file', type=str,
                        required=True, nargs='+',
                        help='Input list-mode filename for Castor (.Cdh)')
    parser.add_argument('-o', '--output', dest='output_file', type=str,
                        required=True,
                        help='Output list-mode filename (.lmDat)')
    parser.add_argument('--flag_tof_out', dest='flag_tof_out',
                        action='store_true',
                        help='When set, the output list-mode includes ' +
                        'time-of-flight information')

    args = parser.parse_args()

    with open(args.output_file, 'wb') as f:
        for input_file in tqdm.tqdm(args.input_file, desc='Input files'):

            header = parse_header(input_file)
            data_file = os.path.join(os.path.split(input_file)[0],
                                     header['Data filename'])
            # TODO: support normalization
            flag_tof_in = bool(int(header.get('TOF information flag', False)))
            flag_atn_in = bool(int(header.get('Attenuation correction flag',
                                              False)))
            data_mode = header['Data mode']
            num_events = np.uint64(header['Number of events'])
            if data_mode != 'list-mode':
                raise NotImplementedError('Only list-mode format supported')
            data_raw = np.fromfile(data_file, dtype='uint32')
            num_fields = 3 + flag_tof_in + flag_atn_in
            castor_events = data_raw.reshape((-1, num_fields))
            if castor_events.shape[0] != num_events:
                raise RuntimeError('Number of events mismatch')

            for row in tqdm.tqdm(castor_events, desc='Events', leave=False):
                f.write(row[0])
                f.write(row[-2])
                f.write(row[-1])
                if args.flag_tof_out:
                    tof = struct.unpack('>f', struct.pack('>I', row[-3]))[0]
                    f.write(np.float32(tof))
