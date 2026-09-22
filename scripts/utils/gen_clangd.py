#!/usr/bin/env python3
"""
Generate .clangd file for use with Clang language server
"""

# %% Imports

import os
import subprocess
import sysconfig
import argparse
import json
import jinja2
try:
    import pybind11
except (ImportError, ModuleNotFoundError):
    pybind11 = None

script_dir = os.path.dirname(os.path.abspath(__file__))

# %% Arguments

parser = argparse.ArgumentParser(
    'Generate .clangd',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--compile_commands', type=os.path.expanduser,
    help='Location of compile_commands.json file')
parser.add_argument(
    '--base_dir', type=os.path.expanduser,
    help='Path to YRT-PET base folder (containing CMakeLists.txt)')
parser.add_argument(
    '--gpu_arch', default=None,
    help='CUDA GPU architecture (e.g. sm_86). '
         'Auto-detected via nvidia-smi if not provided.')

args_p = parser.parse_args()

# %% Template
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(script_dir),
    trim_blocks=True,
    lstrip_blocks=True)
template = env.get_template('.clangd.tpl')

# %% Parse compile_commands.json

with open(args_p.compile_commands, 'rt') as fid:
    comp_cmd = json.load(fid)

directories = [entry['directory'] for entry in comp_cmd
               if 'directory' in entry]
build_dir = os.path.commonpath(directories)

# Absolute base directory
base_dir = os.path.abspath(args_p.base_dir)

# Python.h
python_include_dir = sysconfig.get_path('include')
pybind11_include_dir = pybind11.get_include() if pybind11 else None

# %% GPU architecture

gpu_arch = args_p.gpu_arch
if gpu_arch is None:
    try:
        cap = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            text=True).strip().split('\n')[0]
        gpu_arch = 'sm_' + cap.replace('.', '')
        print(f'Detected GPU architecture: {gpu_arch}')
    except (subprocess.CalledProcessError, FileNotFoundError):
        gpu_arch = 'sm_86'
        print(f'nvidia-smi not available, defaulting to {gpu_arch}')

# %% Write output

out_fname = os.path.join(base_dir, '.clangd')
if os.path.isfile(out_fname):
    raise FileExistsError(f'File "{out_fname}" already exists.')
with open(out_fname, 'wt') as fid:
    fid.write(template.render(yrt_pet_src_dir=base_dir,
                              yrt_pet_build_dir=build_dir,
                              python_include_dir=python_include_dir,
                              pybind11_include_dir=pybind11_include_dir,
                              gpu_arch=gpu_arch))

print(f'"{out_fname}" created')
