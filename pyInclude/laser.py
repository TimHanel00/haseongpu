# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Laser spectra and pump-beam configuration used by the Python interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .solvers import PumpSolver
from .mesh import MeshSelection
from hase_units import LENGTH, POWER, Quantity, requireQuantity, units


# Symmetric seven-point Dunavant rule (degree five). These weights sum to one,
# so the weighted mean is multiplied by each physical triangle's area.
_TRIANGLE_QUADRATURE_BARYCENTRIC = np.asarray(
    [
        [1 / 3, 1 / 3, 1 / 3],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.797426985353087, 0.101286507323456, 0.101286507323456],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
    ],
    dtype=np.float64,
)
_TRIANGLE_QUADRATURE_WEIGHTS = np.asarray(
    [
        0.225,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ],
    dtype=np.float64,
)



def _positive_normalized(values, name):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total weight")
    return values / total


def _unit_vector(value, name):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite three-vector")
    length = float(np.linalg.norm(value))
    if length <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return tuple(value / length)


@dataclass(frozen=True)
class PumpSpectrum:
    """Normalized discrete wavelength distribution for a physical pump.

    Parameters
    ----------
    wavelengths
        One-dimensional positive length :class:`Quantity`.
    weights
        Non-negative relative powers, one per wavelength. They are normalized
        to sum to one and do not change :attr:`Pump.totalPower`.

    Examples
    --------
    ``PumpSpectrum(units.nm * [930, 940], [0.25, 0.75])``
    """

    wavelengths: object
    """Positive wavelength samples as a one-dimensional length quantity."""
    weights: object
    """Normalized fraction of :attr:`Pump.totalPower` assigned to each line."""

    def __post_init__(self):
        quantity = requireQuantity(self.wavelengths, LENGTH, "pump spectrum wavelengths")
        wavelengths = np.asarray(quantity.siValue, dtype=np.float64).reshape(-1)
        weights = _positive_normalized(self.weights, "pump spectrum weights")
        if wavelengths.size != weights.size:
            raise ValueError("pump spectrum wavelengths and weights must have the same length")
        if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
            raise ValueError("pump spectrum wavelengths must be finite and positive")
        object.__setattr__(self, "wavelengths", Quantity(quantity.magnitude, quantity.unit))
        object.__setattr__(self, "weights", weights)

    @classmethod
    def monochromatic(cls, wavelength):
        """Construct a one-line spectrum at a positive scalar wavelength.

        Parameters
        ----------
        wavelength
            Scalar physical length quantity.
        """
        wavelength = requireQuantity(wavelength, LENGTH, "pump wavelength")
        return cls(Quantity([wavelength.magnitude], wavelength.unit), [1.0])


@dataclass(frozen=True)
class PumpAngularDistribution:
    """Discrete launch directions in the injector's inward-local frame.

    Parameters
    ----------
    polarAngles
        Radians away from the inward surface normal, restricted to
        ``[0, pi/2)`` so histories launch into the mesh.
    azimuthalAngles
        Radians around the inward normal, one per polar angle.
    weights
        Non-negative relative probabilities, normalized to sum to one.
    """

    polarAngles: object
    """Launch angles in radians away from the inward surface normal."""
    azimuthalAngles: object
    """Launch angles in radians around the inward surface normal."""
    weights: object
    """Normalized probability assigned to each discrete launch direction."""

    def __post_init__(self):
        polar = np.asarray(self.polarAngles, dtype=np.float64).reshape(-1)
        azimuthal = np.asarray(self.azimuthalAngles, dtype=np.float64).reshape(-1)
        weights = _positive_normalized(self.weights, "pump angular weights")
        if polar.size != azimuthal.size or polar.size != weights.size:
            raise ValueError("pump angular angles and weights must have the same length")
        if np.any(~np.isfinite(polar)) or np.any((polar < 0.0) | (polar >= 0.5 * np.pi)):
            raise ValueError("pump polar angles must be finite and in [0, pi/2)")
        if np.any(~np.isfinite(azimuthal)):
            raise ValueError("pump azimuthal angles must be finite")
        object.__setattr__(self, "polarAngles", polar)
        object.__setattr__(self, "azimuthalAngles", azimuthal)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def collimated(cls):
        """Return one direction parallel to the inward surface normal."""
        return cls([0.0], [0.0], [1.0])

    @classmethod
    def uniformCone(cls, halfAngle, *, polarSamples=8, azimuthalSamples=16):
        """Discretize a uniform-solid-angle cone around the inward normal.

        Parameters
        ----------
        halfAngle
            Cone half-angle in radians, strictly between zero and ``pi/2``.
        polarSamples
            Number of equal-solid-angle polar strata.
        azimuthalSamples
            Number of uniform azimuthal strata per polar stratum.

        Returns
        -------
        PumpAngularDistribution
            Distribution containing ``polarSamples * azimuthalSamples``
            equally weighted directions.
        """
        if halfAngle <= 0.0 or halfAngle >= 0.5 * np.pi:
            raise ValueError("halfAngle must be in (0, pi/2)")
        cos_edges = np.linspace(1.0, np.cos(float(halfAngle)), int(polarSamples) + 1)
        polar = np.arccos(0.5 * (cos_edges[:-1] + cos_edges[1:]))
        azimuthal = (np.arange(int(azimuthalSamples)) + 0.5) * (2.0 * np.pi / int(azimuthalSamples))
        theta, phi = np.meshgrid(polar, azimuthal, indexing="ij")
        return cls(theta.reshape(-1), phi.reshape(-1), np.ones(theta.size))


@dataclass(frozen=True)
class UniformPumpProfile:
    """Uniform relative power density over the injector aperture."""

    kind: str = field(default="uniform", init=False)
    """Profile discriminator consumed by the pump transport adapter."""


@dataclass(frozen=True)
class SuperGaussianPumpProfile:
    """Elliptical super-Gaussian relative irradiance profile.

    Parameters
    ----------
    radiusU, radiusV
        Positive length scales along the profile axes. At either radius the
        relative weight is ``exp(-1)``. ``radiusV=None`` uses ``radiusU``.
    exponent
        Positive super-Gaussian order. ``2`` gives a Gaussian-shaped profile;
        larger values approach a flat top.
    center
        Three-component physical position in mesh coordinates.
    axisU, axisV
        Orthogonal, finite world-space directions. They are normalized by the
        constructor.

    Notes
    -----
    The profile controls relative spatial sampling. Registered
    :attr:`Pump.totalPower` remains the aperture-integrated physical power.
    """

    radiusU: Quantity
    """Positive ``exp(-1)`` profile radius along :attr:`axisU`."""
    radiusV: Quantity | None = None
    """Positive ``exp(-1)`` radius along :attr:`axisV`; defaults to radiusU."""
    exponent: float = 40.0
    """Positive super-Gaussian order; 2 is Gaussian and large values flatten."""
    center: Quantity = Quantity((0.0, 0.0, 0.0), units.m)
    """Three-dimensional physical centre of the transverse profile."""
    axisU: object = (1.0, 0.0, 0.0)
    """Normalized world-space direction defining the first profile axis."""
    axisV: object = (0.0, 1.0, 0.0)
    """Normalized world-space direction defining the second profile axis."""
    kind: str = field(default="super-gaussian", init=False)
    """Profile discriminator consumed by the pump transport adapter."""

    def __post_init__(self):
        radiusV = self.radiusU if self.radiusV is None else self.radiusV
        radiusU = requireQuantity(self.radiusU, LENGTH, "pump profile radiusU")
        radiusV = requireQuantity(radiusV, LENGTH, "pump profile radiusV")
        center = requireQuantity(self.center, LENGTH, "pump profile center")
        if np.shape(center.magnitude) != (3,):
            raise ValueError("pump profile center must be a three-vector")
        if float(radiusU.siValue) <= 0.0 or float(radiusV.siValue) <= 0.0 or self.exponent <= 0.0:
            raise ValueError("super-Gaussian radii and exponent must be positive")
        if np.any(~np.isfinite(center.siValue)):
            raise ValueError("pump profile center must be a finite three-vector")
        axisU = np.asarray(_unit_vector(self.axisU, "axisU"))
        axisV = np.asarray(_unit_vector(self.axisV, "axisV"))
        if abs(float(np.dot(axisU, axisV))) > 1.0e-10:
            raise ValueError("axisU and axisV must be orthogonal")
        object.__setattr__(self, "radiusU", radiusU)
        object.__setattr__(self, "radiusV", radiusV)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "axisU", tuple(axisU))
        object.__setattr__(self, "axisV", tuple(axisV))

    def weightAt(self, points, coordinateUnit):
        """Evaluate relative profile weights at mesh-coordinate points.

        Parameters
        ----------
        points
            Numeric array whose final dimension has length three.
        coordinateUnit
            Length :class:`Unit` represented by the numeric coordinates.

        Returns
        -------
        numpy.ndarray
            Dimensionless weights with shape ``points.shape[:-1]``.
        """
        points = np.asarray(points, dtype=np.float64)
        center = np.asarray(self.center.toValue(coordinateUnit))
        relative = points - center
        u = relative @ np.asarray(self.axisU) / float(self.radiusU.toValue(coordinateUnit))
        v = relative @ np.asarray(self.axisV) / float(self.radiusV.toValue(coordinateUnit))
        return np.exp(-((u * u + v * v) ** (0.5 * self.exponent)))


@dataclass(frozen=True)
class Pump:
    """Physical pump light, independent of injection and ray sampling.

    Parameters
    ----------
    totalPower
        Positive aperture-integrated power :class:`Quantity`, not irradiance
        and not per-ray power.
    spectrum
        Normalized wavelength distribution.
    profile
        Uniform or super-Gaussian relative spatial distribution.
    angularDistribution
        Discrete launch directions in each injector face's inward-local frame.
    name
        Optional human-readable identifier.
    """

    totalPower: Quantity
    """Positive aperture-integrated incident power shared by all pump rays."""
    spectrum: PumpSpectrum
    """Normalized wavelength fractions into which total power is divided."""
    profile: object = field(default_factory=UniformPumpProfile)
    """Relative irradiance shape normalized over the injector aperture."""
    angularDistribution: PumpAngularDistribution = field(default_factory=PumpAngularDistribution.collimated)
    """Discrete ray directions expressed in each face's inward-local frame."""
    name: str | None = None
    """Optional human-readable label; it has no effect on transport physics."""

    def __post_init__(self):
        power = requireQuantity(self.totalPower, POWER, "pump totalPower")
        if not np.isfinite(power.siValue) or float(power.siValue) <= 0.0:
            raise ValueError("pump totalPower must be finite and positive")
        if not isinstance(self.spectrum, PumpSpectrum):
            raise TypeError("pump spectrum must be PumpSpectrum")
        if not isinstance(self.angularDistribution, PumpAngularDistribution):
            raise TypeError("pump angularDistribution must be PumpAngularDistribution")
        if not isinstance(self.profile, (UniformPumpProfile, SuperGaussianPumpProfile)):
            raise TypeError("pump profile must be UniformPumpProfile or SuperGaussianPumpProfile")


class GaussianPump(Pump):
    """Convenience pump with a Gaussian or super-Gaussian profile.

    Parameters
    ----------
    totalPower
        Positive aperture-integrated power quantity.
    spectrum
        Pump wavelength distribution.
    waist
        Scalar radius or two-component ``(u, v)`` radius quantity.
    exponent
        Positive profile order; defaults to the Gaussian value ``2``.
    center
        Three-component profile center as a length quantity.
    axisU, axisV
        Orthogonal world-space profile directions.
    angularDistribution
        Optional launch distribution; omitted means collimated.
    name
        Optional human-readable identifier.
    """

    def __init__(
        self,
        *,
        totalPower,
        spectrum,
        waist,
        exponent=2.0,
        center=Quantity((0.0, 0.0, 0.0), units.m),
        axisU=(1.0, 0.0, 0.0),
        axisV=(0.0, 1.0, 0.0),
        angularDistribution=None,
        name=None,
    ):
        waist = requireQuantity(waist, LENGTH, "GaussianPump waist")
        if np.shape(waist.magnitude) == ():
            radii = (waist, waist)
        elif np.shape(waist.magnitude) == (2,):
            radii = (
                Quantity(np.asarray(waist.magnitude)[0], waist.unit),
                Quantity(np.asarray(waist.magnitude)[1], waist.unit),
            )
        else:
            raise ValueError("waist must be a scalar or a two-vector quantity")
        super().__init__(
            totalPower=totalPower,
            spectrum=spectrum,
            profile=SuperGaussianPumpProfile(
                radiusU=radii[0],
                radiusV=radii[1],
                exponent=exponent,
                center=center,
                axisU=axisU,
                axisV=axisV,
            ),
            angularDistribution=(
                PumpAngularDistribution.collimated()
                if angularDistribution is None
                else angularDistribution
            ),
            name=name,
        )


@dataclass(frozen=True)
class SurfacePumpInjector:
    """Place a physical pump on selected exterior triangular mesh faces.

    Parameters
    ----------
    surface
        Surface-kind :class:`MeshSelection` belonging to the simulation mesh.
        The selected triangles form the launch aperture. For each triangle the
        oriented exterior normal defines the opposite, inward launch direction;
        a tangent frame supplies local coordinates for the pump's spatial and
        angular distributions. Triangle area participates in normalizing the
        profile so :attr:`Pump.totalPower` is the integral over the complete
        selected aperture, rather than power per face or per ray.

    Notes
    -----
    This object specifies *where and how pump histories enter*. It does not set
    the optical behavior when any ray later reaches those faces. The same faces
    therefore still need an exterior model through :meth:`Simulation.addBoundary`.

    Examples
    --------
    ``injector = SurfacePumpInjector(mesh.surface("pump entrance"))``
    """

    surface: MeshSelection
    """Exterior triangular faces forming the pump's physical launch aperture."""

    def __post_init__(self):
        if not isinstance(self.surface, MeshSelection) or self.surface.kind != "surface":
            raise TypeError("SurfacePumpInjector.surface must be mesh.surface(...)")


@dataclass(frozen=True)
class PlanarPumpRelay:
    """Affine re-imaging link between planar boundary selections.

    Parameters
    ----------
    exitSurface, entrySurface
        Surface selections belonging to the same mesh.
    flipU, flipV
        Mirror the corresponding local aperture coordinate.
    rotation
        In-plane rotation in radians.
    offset
        Two-component displacement in mesh coordinate units.
    tilt
        Two angular deflections in radians.
    magnification
        Positive transverse scale factor.
    transmission
        Dimensionless surviving power fraction in ``[0, 1]``.
    """

    exitSurface: MeshSelection
    """Exterior plane from which routed pump rays leave the mesh."""
    entrySurface: MeshSelection
    """Exterior plane through which routed pump rays re-enter the mesh."""
    flipU: bool = False
    """Mirror the first local transverse coordinate during re-imaging."""
    flipV: bool = False
    """Mirror the second local transverse coordinate during re-imaging."""
    rotation: float = 0.0
    """In-plane image rotation in radians."""
    offset: tuple[float, float] = (0.0, 0.0)
    """Image displacement in mesh-coordinate units along entry-plane axes."""
    tilt: tuple[float, float] = (0.0, 0.0)
    """Angular deflection in radians about the two entry-plane axes."""
    magnification: float = 1.0
    """Positive transverse image-size scale factor."""
    transmission: float = 1.0
    """Fraction of routed ray weight surviving the relay, from zero to one."""

    def __post_init__(self):
        if not isinstance(self.exitSurface, MeshSelection) or self.exitSurface.kind != "surface":
            raise TypeError("relay exitSurface must be mesh.surface(...)")
        if not isinstance(self.entrySurface, MeshSelection) or self.entrySurface.kind != "surface":
            raise TypeError("relay entrySurface must be mesh.surface(...)")
        if self.exitSurface.mesh is not self.entrySurface.mesh:
            raise ValueError("relay surfaces must belong to the same mesh")
        if self.magnification <= 0.0:
            raise ValueError("relay magnification must be positive")
        if self.transmission < 0.0 or self.transmission > 1.0:
            raise ValueError("relay transmission must be in [0, 1]")
        if len(tuple(self.offset)) != 2 or len(tuple(self.tilt)) != 2:
            raise ValueError("relay offset and tilt must be two-vectors")

    @classmethod
    def retroreflect(cls, surface, *, transmission=1.0):
        """Create a relay returning to the same surface selection.

        Parameters
        ----------
        surface
            Entry and exit surface selection.
        transmission
            Dimensionless surviving power fraction in ``[0, 1]``.
        """
        return cls(surface, surface, transmission=transmission)


@dataclass(frozen=True)
class MonteCarloPumpSolver(PumpSolver):
    """Monte Carlo controls shared by all registered pumps.

    Parameters
    ----------
    rayCount
        Positive number of pump histories per evaluation. It affects sampling
        noise, not physical pump power.
    seed
        Reproducible unsigned 32-bit random seed.
    maxSteps
        Optional number of outer time steps during which pumping is active.
        ``None`` leaves pumping active for the complete run.
    """

    rayCount: int = 100_000
    """Pump histories sampled per evaluation; changes noise, not total power."""
    seed: int = 5489
    """Unsigned 32-bit seed controlling reproducible pump sampling."""
    maxSteps: int | None = None
    """Outer steps with pumping enabled; ``None`` keeps it enabled throughout."""

    def __post_init__(self):
        if self.rayCount <= 0:
            raise ValueError("MonteCarloPumpSolver.rayCount must be positive")
        if self.seed < 0 or self.seed >= 2**32:
            raise ValueError("MonteCarloPumpSolver.seed must fit uint32")
        if self.maxSteps is not None and self.maxSteps < 0:
            raise ValueError("MonteCarloPumpSolver.maxSteps must be non-negative")


def integratePumpProfile(topology, surface, profile):
    """Integrate a pump profile over selected physical boundary area.

    Each face uses the symmetric seven-point Dunavant triangle rule, exact for
    polynomial integrands through degree five. Its barycentric weights sum to
    one, so the weighted mean is multiplied by the physical triangle area. A
    uniform profile therefore returns exactly the selected aperture area.

    Parameters
    ----------
    topology
        Owning :class:`UnstructuredMesh`.
    surface
        Surface-kind selection belonging to ``topology``.
    profile
        :class:`UniformPumpProfile` or :class:`SuperGaussianPumpProfile`.

    Returns
    -------
    Quantity
        Profile-weighted area in ``topology.coordinateUnit ** 2``.
    """
    if not isinstance(surface, MeshSelection) or surface.kind != "surface":
        raise TypeError("surface must be mesh.surface(...)")
    if surface.mesh is not topology:
        raise ValueError("surface selection belongs to a different mesh")
    mask = surface.mask()
    cell_faces = np.argwhere(mask)
    if cell_faces.size == 0:
        raise ValueError("pump profile integration selected no exterior faces")
    indices = np.asarray(topology.facePointIndices)[mask]
    triangles = np.asarray(topology.points, dtype=np.float64)[indices]
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    points = np.einsum("qb,tbc->tqc", _TRIANGLE_QUADRATURE_BARYCENTRIC, triangles)
    values = (
        np.ones(points.shape[:2])
        if isinstance(profile, UniformPumpProfile)
        else profile.weightAt(points, topology.coordinateUnit)
    )
    return Quantity(
        float(np.sum(area * (values @ _TRIANGLE_QUADRATURE_WEIGHTS))),
        topology.coordinateUnit**2,
    )
