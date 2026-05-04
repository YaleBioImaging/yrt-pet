CompileFlags:
  # 1. Remove the broken relative paths coming from the CDB
  Remove:
    - "-forward-unknown-to-host-compiler"
    - "--options-file*"
    - "-arch=*"
    - "--compiler-options*"
    - "-Xcompiler*"

  # 2. Add them back relative to the .clangd file (portable!)
  Add:
    - "-xcuda"
    - "--cuda-gpu-arch=sm_70"
    - "-I{{ yrt_pet_src_dir }}/include"
    - "-I{{ yrt_pet_src_dir }}/src"
    - "-I{{ yrt_pet_build_dir }}/external/JSON/include"
    - "-I{{ yrt_pet_build_dir }}/external/Catch/include"
    - "-I{{ python_include_dir }}"
    - "-Wno-unknown-cuda-version"

Index:
  Background: Build
