# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import re

from rocm_docs import ROCmDocs

version_number = "1.0.0b1"
left_nav_title = f"hipMM {version_number} Documentation"

# for PDF output on Read the Docs
project = "hipMM Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number
cpp_maximum_signature_line_length = 10
setting_all_article_info = True
all_article_info_os = ["linux"]
all_article_info_author = ""

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-ds"}

extensions = [
    "rocm_docs",
    "breathe",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "numpydoc",
    # "sphinx_markdown_tables",
    # "sphinx.ext.doctest",
    # "sphinx.ext.linkcode",
    # "IPython.sphinxext.ipython_console_highlighting",
    # "IPython.sphinxext.ipython_directive",
    # "nbsphinx",
    # "recommonmark",
    # "sphinx_copybutton",
]

external_toc_path = "./sphinx/_toc.yml"
doxygen_root = "doxygen"
doxysphinx_enabled = False
doxygen_project = {
    "name": "doxygen",
    "path": "doxygen/xml",
}

external_projects_current_project = "hipMM"
