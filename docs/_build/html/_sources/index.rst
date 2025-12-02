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

.. _hipMM-index:

********************************************************************
HIP Memory Manager (hipMM) documentation
********************************************************************

The HIP Memory Manager (hipMM) provides advanced GPU memory management utilities used across the ROCm-DS libraries.
Based on the RAPIDS® Memory Manager (RMM 25.02), hipMM supports efficient allocation, pooling, and data movement, ensuring
stable performance in complex, multi-library GPU workflows.

hipMM 3.0.0 improves memory usage efficiency for workloads that leverage hipDF, hipGRAPH, hipRAFT, and hipVS.

The hipMM code is open and hosted at `https://github.com/ROCm-DS/hipMM <https://github.com/ROCm-DS/hipMM>`_.

The hipMM documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * `hipDF supported environments, features, and interfaces <./install/hipMM-support.html>`_
    * `Install hipMM <./install/INSTALL.html>`_
    * `Build hipMM <./install/BUILD.html>`_

  .. grid-item-card:: Reference

    * :ref:`hipMM-python-api`

To contribute to the documentation refer to `Contributing to ROCm-DS  <https://rocm.docs.amd.com/projects/rocm-ds/en/latest/contribute/contributing.html>`__.

You can find licensing information on the `Licenses <https://rocm.docs.amd.com/projects/rocm-ds/en/latest/about/license.html>`__ page.
