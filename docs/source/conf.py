# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys


# Autodoc only needs the Python sources.  Import them directly so documentation
# builds do not compile the native runtime (in particular on Read the Docs).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HASEonGPU'
copyright = '2026, Erik Zenker, Carlchristian Eckert, Dr. Daniel Albach, Tim Hanel'
author = 'Erik Zenker, Carlchristian Eckert, Dr. Daniel Albach, Tim Hanel'
release = '2.2.0'
version = '2.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

templates_path = ['_templates']
# Autosummary pages are generated into an ignored source directory. Exclude
# stale pages for names retired by the breaking frontend redesign so an
# incremental documentation build cannot try to autodoc removed public names.
exclude_patterns = [
    "generated/HASEonGPU.CrossSectionData.rst",
    "generated/HASEonGPU.GainMedium.rst",
    "generated/HASEonGPU.GainMediumGeometry.rst",
    "generated/HASEonGPU.Gmsh.rst",
    "generated/HASEonGPU.Grid.rst",
    "generated/HASEonGPU.LaserProperties.rst",
    "generated/HASEonGPU.MeshTopology.rst",
    "generated/HASEonGPU.PhiASE.rst",
    "generated/HASEonGPU.SpectralDecomposition.rst",
    "generated/HASEonGPU.SurfaceOptics.rst",
    "generated/HASEonGPU.TimeSteppedSimulation.rst",
    "generated/HASEonGPU.VolumeTopology.rst",
    "generated/HASEonGPU.calcGainFromState.rst",
    "generated/HASEonGPU.vtkWedge.rst",
    "generated/HASEonGPU.writeGainMediumVtk.rst",
    "generated/HASEonGPU.unitDimension.rst",
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
