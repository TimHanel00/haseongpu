# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extensible solver roles used by the public simulation API."""


class ASESolver:
    """Base class for algorithms that evaluate amplified spontaneous emission.

    Frontends may define additional solver descriptors by deriving from this
    class. The current native adapter supports :class:`MonteCarloASESolver`
    only and reports other descriptors before launching transport.
    """


class PumpSolver:
    """Base class for algorithms that evaluate optical pumping.

    The role is deliberately separate from a physical ``Pump`` definition.
    The current native adapter supports :class:`MonteCarloPumpSolver` only.
    """
