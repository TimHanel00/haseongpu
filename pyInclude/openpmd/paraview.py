from __future__ import annotations

from pathlib import Path

import numpy as np


_DEFAULT_PATTERN_STEM = "laserPumpCladding_%06T"
_PARAVIEW_BACKEND_SUFFIXES = (
    ("adios2", "bp"),
    ("hdf5", "h5"),
)


def _default_series_pattern(io):
    """Return a ParaView series pattern supported by the active provider."""
    variants = getattr(io, "variants", {})
    extensions = set(getattr(io, "file_extensions", ()))
    for variant, suffix in _PARAVIEW_BACKEND_SUFFIXES:
        if variants.get(variant, False) and suffix in extensions:
            return f"{_DEFAULT_PATTERN_STEM}.{suffix}"
    available = ", ".join(sorted(extensions)) or "none"
    raise RuntimeError(
        "writeParaviewState requires an openPMD-api build with ADIOS2 or HDF5 support; "
        f"the active provider reports file extensions: {available}"
    )


def _access(io, name):
    if hasattr(io, "Access_Type"):
        return getattr(io.Access_Type, name)
    return getattr(io.Access, name)


def _unit_dimensionless(io):
    return {}


def _reset_scalar_record(iteration, name, values, primitive_shape, axis_labels):
    import openpmd_api as io

    # Public mesh arrays are deliberately read-only. openPMD's store_chunk
    # requires a writable contiguous buffer, so always materialize one here.
    data = np.array(values, copy=True, order="C").reshape(-1)
    record = iteration.meshes[name]
    record.set_attribute("geometry", "other")
    record.set_attribute("geometryParameters", "topology=extruded_triangular_prism")
    record.set_attribute("dataOrder", "C")
    record.set_attribute("hasePrimitiveShape", list(primitive_shape))
    record.axis_labels = ["flatIndex"]
    record.grid_spacing = [1.0]
    record.grid_global_offset = [0.0]
    record.grid_unit_SI = 1.0
    record.unit_dimension = _unit_dimensionless(io)

    component = record[io.Mesh_Record_Component.SCALAR]
    component.unit_SI = 1.0
    component.position = [0.0]
    component.reset_dataset(io.Dataset(data.dtype, data.shape))
    component.store_chunk(data)
    record.set_attribute("haseAxes", list(axis_labels))


def writeParaviewState(
    state,
    outputDir,
    claddingAbsorption=1.0,
    pattern=None,
    handleName="laserPumpCladding.pmd",
):
    """Append a state to an openPMD series and create a ParaView handle.

    Parameters
    ----------
    state : TimeStepState
        Completed :class:`TimeStepState`. Its ``step`` selects the openPMD
        iteration and its physical ``time`` is stored in seconds.
    outputDir : path-like
        Directory containing the series and ``.pmd`` handle.
    claddingAbsorption : float, optional
        Numeric reciprocal-length factor multiplying :attr:`TimeStepState.phiAse`
        for the derived
        ``cladding_absorption`` record. It is retained for legacy visualization
        workflows and must already use the intended reciprocal-length scaling.
    pattern : str or path-like, optional
        Optional openPMD filename pattern containing an iteration placeholder
        such as ``%06T``. By default ADIOS2 ``.bp`` is used when available,
        otherwise HDF5 ``.h5`` is selected from the active ``openpmd_api``
        provider. An explicit pattern is not replaced.
    handleName : str or path-like, optional
        Name of the text ``.pmd`` file that points ParaView to ``pattern``.

    Returns
    -------
    pathlib.Path
        Path to the written ParaView handle.
    """
    import openpmd_api as io

    pattern = _default_series_pattern(io) if pattern is None else str(pattern)
    output_dir = Path(outputDir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / handleName).write_text(pattern + "\n", encoding="utf-8")

    series_path = output_dir / pattern
    has_existing_series = any(output_dir.glob(pattern.replace("%06T", "*")))
    access = _access(io, "append") if has_existing_series else _access(io, "create")
    series = io.Series(str(series_path), access)
    series.set_software("HASEonGPU")

    iteration = series.iterations[int(state.step)]
    from hase_units import units

    iteration.time = float(state.time.toValue(units.s))
    iteration.dt = 1.0
    iteration.time_unit_SI = 1.0

    topology = state.mesh
    beta_cells = np.asarray(state.sampledExcitationFraction)
    primitive_shape = beta_cells.shape
    _reset_scalar_record(iteration, "beta_cells", beta_cells, primitive_shape, ["point", "level"])

    if state.phiAse is not None:
        phi_ase = np.asarray(state.phiAse)
        _reset_scalar_record(iteration, "phi_ase", phi_ase, phi_ase.shape, ["point", "level"])
        _reset_scalar_record(
            iteration,
            "cladding_absorption",
            phi_ase * np.float64(claddingAbsorption),
            phi_ase.shape,
            ["point", "level"],
        )
    if state.sampledDExcitationDtAse is not None:
        dndt_ase = np.asarray(state.sampledDExcitationDtAse)
        _reset_scalar_record(iteration, "dndt_ase", dndt_ase, dndt_ase.shape, ["point", "level"])
    if state.dExcitationDtPump is not None:
        dndt_pump = np.asarray(state.dExcitationDtPump)
        _reset_scalar_record(iteration, "dndt_pump", dndt_pump, dndt_pump.shape, ["point", "level"])
    if state.excitationFraction is not None:
        beta_volume = np.asarray(state.excitationFraction)
        _reset_scalar_record(iteration, "beta_volume", beta_volume, beta_volume.shape, ["cell", "layer"])

    if topology is not None:
        _reset_scalar_record(iteration, "points", np.asarray(topology.points), topology.points.shape, ["point", "coordinate"])
        if hasattr(topology, "cellConnectivity"):
            connectivity = np.asarray(topology.cellConnectivity, dtype=np.uint32)
            _reset_scalar_record(
                iteration,
                "cell_point_indices",
                connectivity,
                connectivity.shape,
                ["cell", "local_vertex"],
            )
        else:
            connectivity = np.asarray(topology.trianglePointIndices, dtype=np.uint32)
            _reset_scalar_record(
                iteration,
                "triangle_point_indices",
                connectivity,
                connectivity.shape,
                ["cell", "local_vertex"],
            )

    iteration.close()
    series.close()
    return output_dir / handleName
