..
    MIT License

    Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

.. meta::
  :description: hipMM documentation and API reference library
  :keywords: hipMM, RMM, data science, RAPIDS, ROCm-DS, GPU, GPU API, memory-management, memory-allocation, memory-pools

.. _hipMM-support:

*******************************************************
hipMM supported environments, features, and interfaces
*******************************************************

hipMM requires ROCm 7.0.2 running on a `ROCm-supported operating system <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/reference/system-requirements.html#supported-operating-systems>`__. Using Ubuntu 22.04 or later is recommended.
See `ROCm installation for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/>`_
for installation instructions.

In particular, ensure that the following ROCm components are installed:

- `hipBLAS <https://rocm.docs.amd.com/projects/hipBLAS/en/latest/index.html>`__
- `hipFFT <https://rocm.docs.amd.com/projects/hipFFT/en/latest/index.html>`__
- `hipRAND <https://rocm.docs.amd.com/projects/hipRAND/en/latest/index.html>`__
- `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`__
- `hipSPARSE <https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/>`__

You should also have ``gcc`` 11.* and  ``cmake`` 3.26.4 (or later) installed.

hipMM is supported on gfx942 and gfx90a.
