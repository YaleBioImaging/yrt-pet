Aggregation of the basic ideas on how the reconstruction code should be made.
Note: Italic was used to highlight part that were still unsure or not decided.

# Code standard
We use clang-format for the code formatting. The variable naming standard goes with the following rules:
- All variables should have the following format: `prefixes_variableNameInCamelCase_suffixes`
- Private or protected member variables should have the prefix `m`
- Pointer variables, both raw and smart, should have the prefix `p`
- Reference variables should have the prefix `r`
- A device object or pointer should have the prefix `d`
- The prefix `h` can be used to highlight that a variable is host-side
- Parameters should have the prefix `p` when the variable can be confused with a local or a member variable
- Suffixes should be used only when adding units to a variable
- Constants (both `constexpr`s and macros) should be in all-caps
- Exceptions can be made in a variable's name when it would damage code readability or mathematical coherence
- Exceptions to `camelCase` can be made if the variable has a single-letter word. ex: `x`, `y`, `n`

# Reconstruction code basis:
* Core coding language: C++ with c++20 standard
    * Interface for python
* Build system: cmake
* Testing tools:
    * C++: *catch2*
    * python: pytest
* Multi-threading: openMP
    * *Using MPI? Not sure planned*
* GPU: CUDA

# Priority functionality:
- [ ] Scatter estimation with Time-of-flight
- [x] Additive corrections
- [X] GPU projector
- [X] Motion correction directly from List-mode files(s)
- [X] Multiplicative corrections
- [x] PSF inclusion in the projector

# Wish list for a full product:
* [ ] **Fully 3D Multiple bed**
* [ ] Quantitative accuracy
* [ ] Dead time correction
* [ ] Decay correction
* [ ] Dynamic reconstruction
* [ ] Gated reconstruction
