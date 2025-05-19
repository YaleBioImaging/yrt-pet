# Adding plugins to YRT-PET

YRT-PET can be compiled alongside external plugins that can add several things:
- Additional List-mode or Histogram data format support
- Additional executables added to the compilation pipeline
- Additional Python interface functions

Plugins take the form of a folder with source code and a CMakeLists.txt file.
This folder has to be copied (or symbolically linked in) the `plugins` directory
of this repository.

See the `yrt-pet-csv` repository for a minimal example of a plugin that adds
support for a CSV list-mode format.
