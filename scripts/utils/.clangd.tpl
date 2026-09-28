CompileFlags:
  # 1. Remove the broken relative paths coming from the CDB
  Remove:
    - "-forward-unknown-to-host-compiler"
    - "--options-file*"
    - "-arch=*"
    - "--compiler-options*"
    - "-Xcompiler*"

  # 2. Add them back relative to the .clangd file
  Add:
    - "-I{{ yrt_pet_src_dir }}/include"
    - "-I{{ yrt_pet_src_dir }}/src"
    - "-I{{ yrt_pet_build_dir }}/external/JSON/include"
    - "-I{{ yrt_pet_build_dir }}/external/Catch/include"
    - "-Wno-unknown-cuda-version"

Index:
  Background: Build

---
If:
  PathMatch: ".*\\.cu[h]?$"
CompileFlags:
  Add:
    - "-xcuda"
    - "--cuda-gpu-arch={{ gpu_arch }}"
{% if python_include_dir %}
    - "-I{{ python_include_dir }}"
{% endif %}
{% if pybind11_include_dir %}
    - "-I{{ pybind11_include_dir }}"
{% endif %}
