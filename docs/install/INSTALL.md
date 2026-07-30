---
myst:
  html_meta:
    "description": "AMD Data Science library for Data Frames."
    "keywords": "hipMM, RMM, data science, RAPIDS, AMD Data Science, GPU, GPU API, memory-management, memory-allocation, memory-pools"
---

# Building and installing hipMM

> **IMPORTANT:**
> You can install hipMM via AMD PyPI (recommended for regular users) or build
> and install it from source (for developers).

> **NOTE:**
>These instructions use the AMD MI300 GPU (gfx942 architecture).
> However, this is only for example purposes. hipMM supports only AMD GPUs.

## Install hipMM via AMD PyPI

Packaged versions of hipMM and its dependencies are distributed via
[AMD PyPI](https://pypi.amd.com/simple):

```sh
pip install amd-hipmm==1.0.0b1 --extra-index-url=https://pypi.amd.com/simple
```

## Build hipMM from source

### Install Conda

It is recommended to install hipMM inside of a predefined Conda environment to
ensure all dependencies are correctly installed, and it is working properly. A
minimal free version of Conda is [Miniforge](https://conda-forge.org/download/).

### Get hipMM dependencies

* You must have a full ROCm 6.4.0 or later installation on your system. See
  [ROCm installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
  for more information. This guide assumes that the ROCm path is `/opt/rocm`.
* hipMM supports Ubuntu 22.04
* GPU requirements: gfx942 or gfx90a
* `gcc` : version 9.3+
* `cmake` : version 3.26.4+
* hipMM requires Python version 3.10 and the following Python packages:

  - `scikit-build` or `scikit-build-core` depending on the hipMM version
  - `hip-python`
  - `hip-python-as-cuda`
  - `cython`

For more details, see the [pyproject.toml](../../python/pyproject.toml) file.

### Steps to build hipMM from source

To install hipMM from source, ensure the dependencies are met and follow the
steps below:

1. Clone the repository and submodules

   ```sh
   git clone --recurse-submodules https://github.com/AMD-Ecosystem/hipMM.git hipMM
   cd hipMM
   ```

2. Create the Conda development environment `hipmm_dev`:

   ```sh
   # create the conda environment (assuming in base `hipMM` directory)
   conda env create --name hipmm_dev --file conda/environments/all_rocm_arch-x86_64.yaml
   conda activate hipmm_dev
   ```

3. You can install HIP Python and the optional Numba HIP dependency via the
   Github-distributed `numba-hip` package. Select dependencies of Numba HIP
   that agree with your ROCm installation by providing a parameter
   `rocm-${ROCM_MAJOR}-${ROCM-MINOR}-${ROCM-PATCH}` (example: `rocm-6-4-0`) in
   square brackets:

   > **IMPORTANT:**
   > Some `hipMM` dependencies are currently distributed via Test PyPI.

   ```sh
   # conda activate hipmm_dev
   pip install --upgrade pip
   pip install --extra-index-url=https://test.pypi.org/simple \
      numba-hip[rocm-6-4-0]@git+https://github.com/rocm/numba-hip.git
   ```

4. Build and install `librmm` and `rmm` using `build.sh`.

   The `build.sh` command creates the `build` directory at the root of cloned
   hipMM git repository. Use `build.sh -h` to display the help text for the
   script. You can build and install `librmm` and `rmm` separately, and you can
   also build without installing using the `-n` option.

   > **Note:**
   > When building and installing `librmm` only, you can also do this outside
   > of the Conda environment as described in
   > [Installing librmm using CMake and make](#installing-librmm-using-cmake-and-make).

   ```sh
   conda activate hipmm_dev
   export CXX="hipcc"  # Cython CXX compiler, adjust according to your setup.
   export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake" # Locate ROCm CMake packages
   export RAPIDS_CMAKE_HIP_ARCHITECTURES="gfx942" # set AMD GPU architecture(s)

   ./build.sh librmm rmm  # Build and install librmm and rmm (can also use the default ./build.sh)
   ```

   > **Note:** When rebuilding it is recommended to remove previous build
   > files. When you are using the `./build.sh` script, this can be
   > accomplished by additionally specifying `clean`. For example:
   `./build.sh clean rmm`.

5. Build, install, and test the `rmm` python package, in the `python` folder:

   ```sh
   conda activate hipmm_dev
   export CXX="hipcc"  # Cython CXX compiler, adjust according to your setup.
   export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake" # Locate ROCm CMake packages
   export RAPIDS_CMAKE_HIP_ARCHITECTURES="gfx942" # set AMD GPU architecture(s)

   cd python/rmm
   python setup.py build_ext --inplace
   python setup.py install
   pytest -v
   ```

6. Build the `rmm` python package and create a binary wheel, in the `python`
   folder:

   ```sh
   conda activate hipmm_dev
   export CXX="hipcc"  # Cython CXX compiler, adjust according to your setup.
   export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake" # Locate ROCm CMake packages
   export RAPIDS_CMAKE_HIP_ARCHITECTURES="gfx942" # set AMD GPU architecture(s)

   cd python/rmm
   python3 setup.py bdist_wheel
   ```

The build process is complete, and you are ready to develop for the
hipMM OSS project.

### Installing librmm using CMake and make

As an alternative to the above process, you can build and install `librmm`
using `CMake` and `make` commands, and then run tests.

> **Note:**
> This step for C++/HIP-only build of librmm does not require an active Conda
> environment.

As shown in the following commands, when compiling for AMD GPUs you must export
the `CXX` environment variable before building so that the Cython build process
uses a HIP-enabled C++ compiler.

You should also provide the location of ROCm CMake scripts to `CMake` using the
`CMAKE_PREFIX_PATH` CMake/environment variable.

```sh
export CXX="hipcc"                                # Cython CXX compiler, adjust according to your setup.
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake" # ROCm CMake packages
export RAPIDS_CMAKE_HIP_ARCHITECTURES="gfx942"   # set AMD GPU architecture(s)
mkdir build                                       # make a build directory
cd build                                          # enter the build directory
cmake .. -DCMAKE_INSTALL_PREFIX=<customizable_writable_path>     # configure CMake installation path, which must be writeable by the current user
make -j                                           # install the header only library librmm.so ... '-j' will start a parallel job using the number of physical cores available on your system
make install                                      # install the header only library librmm.so to the CMake installation path
```

Optionally run the C++ unit tests:

```sh
$ cd build  # if you are not already in build directory
$ make test  # this optional command will run the hipMM C++ unit tests.
```

### Caching third-party dependencies

hipMM uses [CPM.cmake](https://github.com/TheLartians/CPM.cmake) to handle
third-party dependencies like `spdlog`, `Thrust`, `GoogleTest`,
`GoogleBenchmark`. In general you won't have to worry about third-party
dependencies. If `CMake` finds an appropriate version on your system,
it uses it. Otherwise, those dependencies will be downloaded as part of the
build.

**Note:** You can help by setting `CMAKE_PREFIX_PATH` to point to the
installed location of the third-party software.

If you frequently start new builds from scratch, consider setting the
environment variable `CPM_SOURCE_CACHE` to an external download
directory to avoid repeated downloads of the third-party dependencies.
