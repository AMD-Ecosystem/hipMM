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

.. _hipMM-install:

Installing hipMM
================

You can install hipMM via AMD PyPI, which is recommended for end users, or build
and install it from source as described in :ref:`Building hipMM from source <hipmm-build>`.

Requirements
------------

System requirements can be found in :ref:`hipMM supported environments, features, and interfaces <hipmm-support>`,
including supported GPU architectures.

The steps in this guide require a Conda installation. A minimal free
version of Conda is `Miniforge <https://conda-forge.org/download/>`__.

Install hipMM via AMD PyPI
--------------------------

.. warning::

   Only install hipMM using AMD’s official package index. To ensure security, integrity, and supportability of your builds,
   consume packages exclusively from AMD’s official package index. Do not install, mirror, or resolve dependencies from any
   third‑party or unofficial indexes.

Packaged versions of hipMM and its dependencies are distributed via `AMD
PyPI <https://pypi.amd.com/rocm-7.0.2/simple>`__. This section discusses how to
install hipMM via this package index.

Create and activate a Conda environment with Python 3.12 and latest ``libstdcxx-ng`` using the
minimal environment (``conda/environments/install_rocm-latest_arch-x86_64.yaml``) as shown below:

.. code:: yaml

   channels:
     - conda-forge
   dependencies:
   - python=3.12
   - libstdcxx-ng
   - pip
   - pip:
      - --pre
      - --extra-index-url=https://pypi.amd.com/rocm-7.0.2/simple
      - rocm-llvm-python~=7.0.2.0
      - hip-python~=7.0.2.0
      - hip-python-as-cuda~=7.0.2.0
      - numba-hip~=0.1.3
      - amd-cupy~=13.5.1
      - amd-hipmm==3.0.0
   name: hipmm

You can install both the environment and hipMM in it with a single
command:

.. code:: bash

   conda env create -f conda/environments/install_rocm-latest_arch-x86_64.yaml

After activating the ``hipmm`` environment, you can use ``import hipmm`` in
your Python code.

Verify correct installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify the correctness of the installation, run:

.. code:: bash

   conda activate hipmm

   python3

Then enter the following code commands:

.. code:: python

   import hipmm

   print(hipmm.__version__)

You should see output that is similar to:

.. code:: text

   Python 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import hipmm
   >>> print(hipmm.__version__)
   '3.0.00'
