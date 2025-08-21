# Contributing

## Git feature development workflow
- The main branch is the stable branch of the project, it is designed to
  remain "clean" (i.e. tested and documented).
- Work on the project should be performed on *branches*. A branch is created
  for each feature, bug, etc. and progress is committed on that branch.
- Once the work is ready to *merge* (i.e. the code is complete, tested and
  documented) it can be submitted for review: the process is called *merge
  request*. Follow Github's instructions to submit a pull request.
- The request is then reviewed by one of the project's maintainers. After
  discussions, it should eventually be merged to the main branch by the
  maintainer.
- The branch on which the work was performed can then be deleted (the history is
  preserved). The branch commits will be squashed before the merge (this can
  be done in Github's web interface).
- If the main branch has changed between the beginning of the work and the
  pull request submission, the branch should be *rebased* to the main branch
  (in practice, tests should be run after a rebase to avoid potential
  regressions).
  - In some cases, unexpected rebase are reported (for instance if the history
    is A&#x2013;B&#x2013;C and B is merged to A, a later rebase of C to A may cause
    conflicts that should not exist). In such cases, two fixes are possible:
    - Launching an interactive rebase (`git rebase -i <main branch>`) and dropping the commits that would be
      duplicated.
  - After a rebase, `git push` by default will not allow an update that is not `fast-forward`
    with the corresponding remote branch, causing an error when trying to push.
    `git push --force-with-lease` can be used to force a push while checking that the remote branch has not changed.
    Note that this will lose history

## Testing

There are two types of tests available, unit tests and integration tests.

- Unit tests are short validation programs that check the operation of
  individual modules. To run the unit test suite, simply run `make test` after
  compiling (with `make`).
- Integration test data is currently not publicly available.


## Code standard
Aggregation of the basic ideas on how the reconstruction code should be made.
Note: Italic was used to highlight part that were still unsure or not decided.

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

## Reconstruction code basis
- Core coding language: C++ with c++20 standard
    - Interface for Python
- Build system: `cmake`
- Testing tools:
    - C++: `catch2`
    - Python: `pytest`
- Multi-threading: `std::thread`
- GPU: `CUDA`

## Priority functionality
- [ ] Scatter estimation with Time-of-flight
- [x] Additive corrections
- [X] GPU projector
- [X] Motion correction directly from List-mode files(s)
- [X] Multiplicative corrections
- [x] PSF inclusion in the projector

## Wish list for a full product
- [ ] **Fully 3D Multiple bed**
- [ ] Quantitative accuracy
- [ ] Dead time correction
- [ ] Decay correction
- [ ] Dynamic reconstruction
- [ ] Gated reconstruction

