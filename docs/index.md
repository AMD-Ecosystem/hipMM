<head>
  <meta charset="UTF-8">
  <meta name="description" content="hipMM documentation and API reference library.">
  <meta name="keywords" content="hipMM, RMM, data science, RAPIDS, AMD Data Science, GPU, GPU API, memory-management, memory-allocation, memory-pools">
</head>

********************************************************************
HIP Memory Manager (hipMM) documentation
********************************************************************

The HIP Memory Manager (hipMM) provides advanced GPU memory management utilities used across the AMD Data Science libraries.
Based on the RAPIDS® Memory Manager (RMM 26.10), hipMM supports efficient allocation, pooling, and data movement, ensuring
stable performance in complex, multi-library GPU workflows.

hipMM 5.0.0 improves memory usage efficiency for workloads that leverage hipDF, hipGRAPH, hipRAFT, and hipVS.

The hipMM code is open and hosted at `https://github.com/AMD-Ecosystem/hipMM <https://github.com/AMD-Ecosystem/hipMM>`_.

The hipMM documentation is structured as follows:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Installation

    * [hipMM supported environments, features, and interfaces](./install/hipMM-support.html)
    * [Install hipMM](./install/INSTALL.html)
    * [Build hipMM](./install/BUILD.html)

:::

:::{grid-item-card} Reference

    * [hipMM Python API](./reference/hipmm/python_api.rst)

:::
::::

To contribute to the documentation refer to `Contributing to AMD Data Science  <https://rocm.docs.amd.com/projects/datascience/en/latest/contribute/contributing.html>`__.

You can find licensing information on the `Licenses <https://rocm.docs.amd.com/projects/datascience/en/latest/about/license.html>`__ page.
