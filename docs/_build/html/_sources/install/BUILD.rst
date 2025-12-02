.. MIT License
..
.. Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.

.. meta::
  :description: hipMM documentation and API reference library
  :keywords: hipMM, RMM, data science, RAPIDS, ROCm-DS, GPU, GPU API, memory-management, memory-allocation, memory-pools

.. _hipMM-build:

Building and installing hipMM from source
=========================================

For developers, the following topic provides details on building the C++
components, running tests and benchmarks, and for building
the full hipMM installation including the Python package. End users should see the :ref:`Installation instructions <hipmm-install>`.

Build procedure for C++ components
----------------------------------

Build the hipMM C++/HIP components using the following command:

.. code:: bash

   ./build.sh lib tests benchmarks

Here, ``tests`` and ``benchmarks`` are optional flags that enable the
respective functionality.

.. note::

   To fetch the dependencies, ``git`` needs to be installed on your system.

Running tests and benchmarks
----------------------------

To run the tests use:

.. code:: bash

   ctest --test-dir cpp/build/

To run the benchmarks, use:

.. code:: bash

   find cpp/build/benchmarks/ -type f -executable -exec {} \;

Building and installing hipMM including the Python layer
--------------------------------------------------------

You will perform the following steps:

1. `Install Conda <#step-1-install-conda>`__
2. `Clone the hipMM
   repository <#step-2-clone-the-hipmm-repository>`__
3. `Create and activate hipMM Conda environment <#step-3-create-and-activate-the-hipmm-conda-environment>`__
4. `Build and install hipMM <#step-4-build-and-install-hipmm>`__
5. `Verify correctness of
   installation <#step-5-verify-correct-installation>`__

Step 1: Install conda
~~~~~~~~~~~~~~~~~~~~~

hipMM must be built inside of a predefined Conda environment to ensure
that it is working properly. While the hipMM build process fetches C++
dependencies itself, it has Cython and Python dependencies (CuPy, Numba
HIP, hipMM, HIP Python, ROCm LLVM Python) that need to be installed into
the hipMM Conda environment before you can build and use the package.

On an x86 Linux machine, you can download and install `Miniforge <https://conda-forge.org/download/>`__ via:

.. code:: bash

   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   sh Miniforge3-Linux-x86_64.sh

For other architectures and operating systems take a look at the webpage
of `Miniforge <https://conda-forge.org/download/>`__.

Step 2: Clone the hipMM repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a work directory ``/tmp/hipmm`` and clone hipMM into this
repository:

.. code:: bash

   mkdir -p /tmp/hipmm # NOTE: feel free to adapt

   cd /tmp/hipmm
   git clone https://github.com/ROCm-DS/hipMM hipmm -b release/rocmds-25.10

Step 3: Create and activate the hipMM conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the following conda environment
(``conda/environments/build-release_rocm-latest_arch-x86_64.yaml``):

.. code:: yaml

   channels:
   - rapidsai
   - rapidsai-nightly
   - conda-forge
   dependencies:
   - c-compiler
   - clang-tools==16.0.6
   - clang==16.0.6
   - cmake>=3.26.4,<4,!=3.30.0
   - cxx-compiler
   - cython>=3.0.0
   - fmt>=11.0.2,<12
   - gcc_linux-64=11.*
   - gcovr>=5.0
   - identify>=2.5.20
   - ipython
   - make
   - ninja
   - numpy>=1.23,<3.0a0
   - pre-commit
   - pytest
   - pytest-cov
   - python>=3.10,<3.13
   - rapids-build-backend>=0.3.0,<0.4.0.dev0
   - scikit-build-core >=0.10.0
   - spdlog>=1.14.1,<1.15
   - pip
   - pip:
   - --pre
   - --extra-index-url=https://pypi.amd.com/rocm-7.0.2/simple
   - rocm-llvm-python~=7.0.2.0
   - hip-python~=7.0.2.0
   - hip-python-as-cuda~=7.0.2.0
   - numba-hip~=0.1.3
   - amd-cupy~=13.5.1
   name: hipmm_dev

Create and activate the ``hipmm_dev`` Conda environment via:

.. code:: bash

   cd /tmp/hipmm/hipmm

   conda env create --name hipmm_dev --file conda/environments/build-release_rocm-latest_arch-x86_64.yaml
   conda activate hipmm_dev

Step 4: Build and install hipMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``build.sh`` command creates the ``build`` directory in the ``cpp``
subfolder of the cloned hipMM git repository. You can build and install
``librmm`` and ``rmm`` separately, and you can also build without
installing using the ``-n`` option. Use ``build.sh -h`` to display the help
text for the script.

.. code:: bash

   (hipmm_dev) $ export CXX="hipcc"    # Cython CXX compiler, adjust according to your setup.
   (hipmm_dev) $ export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake"     # Locate ROCm CMake packages
   (hipmm_dev) $ ./build.sh librmm rmm     # Build and install librmm and rmm (can also use the default ./build.sh)

.. note::

   When rebuilding it is recommended to remove previous build files. When you are using the ``./build.sh`` script, this
   can be accomplished by additionally specifying ``clean``. For example: ``./build.sh clean rmm``.

Step 5: Verify correct installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After building and installing hipMM for use in the Conda environment ``hipmm_dev``, you can
verify the correctness of the installation as follows:

.. code:: bash

   conda activate hipmm_dev

   python3

Then enter the following Python code commands:

.. code:: python

   import hipmm

   print(hipmm.__version__)

You should see output similar to:

.. code:: text

   Python 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import hipmm
   >>> print(hipmm.__version__)
   '3.0.00'
