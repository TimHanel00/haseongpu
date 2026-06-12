# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

__version__ = "2.0.0"

import HASEonGPU_Bindings
from HASEonGPU_Bindings import *
from pyInclude import *
for _name in ("HostMesh", "ExperimentParameters", "ComputeParameters", "Mesh", "calcPhiASE"):
    globals().pop(_name, None)
del _name
del HASEonGPU_Bindings.HostMesh
