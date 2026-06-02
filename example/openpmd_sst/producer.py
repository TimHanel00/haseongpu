#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import openpmd_api as io


DIMENSIONLESS = {}
LENGTH = {io.Unit_Dimension.L: 1.0}


def access_create_linear():
    if hasattr(io, "Access_Type"):
        return io.Access_Type.create_linear
    return io.Access.create_linear


def reset_scalar_record(record, data, axis_labels):
    record.set_attribute("geometry", "other")
    record.set_attribute("geometryParameters", "topology=unstructured_triangular_prism")
    record.set_attribute("dataOrder", "C")
    record.axis_labels = axis_labels
    record.grid_spacing = [1.0] * data.ndim
    record.grid_global_offset = [0.0] * data.ndim
    record.grid_unit_SI = 1.0
    record.unit_dimension = DIMENSIONLESS
    component = record[io.Mesh_Record_Component.SCALAR]
    component.unit_SI = 1.0
    component.position = [0.0] * data.ndim
    component.reset_dataset(io.Dataset(data.dtype, data.shape))
    component.store_chunk(data)


def reset_component(record, component_name, data, axis_labels, unit_dimension):
    record.set_attribute("geometry", "other")
    record.set_attribute("geometryParameters", "topology=unstructured_triangular_prism")
    record.set_attribute("dataOrder", "C")
    record.axis_labels = axis_labels
    record.grid_spacing = [1.0] * data.ndim
    record.grid_global_offset = [0.0] * data.ndim
    record.grid_unit_SI = 1.0
    record.unit_dimension = unit_dimension
    component = record[component_name]
    component.unit_SI = 1.0
    component.position = [0.0] * data.ndim
    component.reset_dataset(io.Dataset(data.dtype, data.shape))
    component.store_chunk(data)


def write_random_core_mesh(stream):
    rng = np.random.default_rng(seed=42)
    number_of_points = 8
    number_of_cells = 5
    number_of_levels = 4
    number_of_interfaces = 2

    vertices_x = rng.random(number_of_points, dtype=np.float64)
    vertices_y = rng.random(number_of_points, dtype=np.float64)
    connectivity = rng.integers(
        0, number_of_points, size=(number_of_cells, 3), dtype=np.uint32
    )
    neighbors = rng.integers(
        -1, number_of_cells, size=(number_of_cells, 3), dtype=np.int32
    )
    forbidden_edges = rng.integers(-1, 3, size=(number_of_cells, 3), dtype=np.int32)
    normal_points = rng.integers(
        0, number_of_points, size=(number_of_cells, 3), dtype=np.uint32
    )
    cell_center_x = rng.random(number_of_cells, dtype=np.float64)
    cell_center_y = rng.random(number_of_cells, dtype=np.float64)
    cell_normal_x = rng.uniform(-1.0, 1.0, size=(number_of_cells, 3))
    cell_normal_y = rng.uniform(-1.0, 1.0, size=(number_of_cells, 3))
    surface = rng.random(number_of_cells, dtype=np.float32)
    beta_volume = rng.random((number_of_cells, number_of_levels - 1), dtype=np.float64)
    cladding_cell_type = rng.integers(0, 2, size=number_of_cells, dtype=np.uint32)
    reflectivity = rng.random((number_of_cells, number_of_interfaces), dtype=np.float32)
    point_beta = rng.random((number_of_points, number_of_levels), dtype=np.float64)
    refractive_index = np.asarray([1.5, 1.0, 1.5, 1.0], dtype=np.float32)
    lambda_absorption = np.asarray([900.0, 1000.0, 1100.0], dtype=np.float64)
    lambda_emission = np.asarray([900.0, 1000.0, 1100.0], dtype=np.float64)
    sigma_absorption = np.asarray([1.0e-21, 2.0e-21, 1.5e-21], dtype=np.float64)
    sigma_emission = np.asarray([1.5e-21, 2.5e-21, 2.0e-21], dtype=np.float64)

    config = {"adios2": {"engine": {"parameters": {"DataTransport": "WAN"}}}}
    series = io.Series(stream, access_create_linear(), config)
    series.set_software("HASEonGPU-openPMD-SST-prototype")

    iteration = series.snapshots()[0]
    iteration.time = 0.0
    iteration.dt = 1.0
    iteration.time_unit_SI = 1.0
    iteration.set_attribute("number_of_points", number_of_points)
    iteration.set_attribute("number_of_cells", number_of_cells)
    iteration.set_attribute("number_of_levels", number_of_levels)
    iteration.set_attribute("thickness", 1.0)
    iteration.set_attribute("n_tot", 1.0)
    iteration.set_attribute("crystal_t_fluo", 1.0)
    iteration.set_attribute("cladding_number", 0)
    iteration.set_attribute("cladding_absorption", 0.0)
    iteration.set_attribute("min_rays_per_sample", 1)
    iteration.set_attribute("max_rays_per_sample", 1)
    iteration.set_attribute("mse_threshold", 1.0)
    iteration.set_attribute("repetitions", 1)
    iteration.set_attribute("adaptive_steps", 1)
    iteration.set_attribute("use_reflections", False)
    iteration.set_attribute("spectral_resolution", 3)
    iteration.set_attribute("backend", "Host_Cpu_CpuSerial")
    iteration.set_attribute("max_gpus", 1)

    reset_component(iteration.meshes["core_vertices"], "x", vertices_x, ["point"], LENGTH)
    reset_component(iteration.meshes["core_vertices"], "y", vertices_y, ["point"], LENGTH)
    reset_scalar_record(
        iteration.meshes["core_connectivity"],
        connectivity,
        ["cell", "local_vertex"],
    )
    reset_scalar_record(
        iteration.meshes["core_neighbors"],
        neighbors,
        ["cell", "local_side"],
    )
    reset_scalar_record(
        iteration.meshes["core_forbidden_edges"],
        forbidden_edges,
        ["cell", "local_side"],
    )
    reset_scalar_record(
        iteration.meshes["core_normal_points"],
        normal_points,
        ["cell", "local_side"],
    )
    reset_component(iteration.meshes["core_cell_center"], "x", cell_center_x, ["cell"], LENGTH)
    reset_component(iteration.meshes["core_cell_center"], "y", cell_center_y, ["cell"], LENGTH)
    reset_scalar_record(
        iteration.meshes["core_cell_normal_x"],
        cell_normal_x,
        ["cell", "local_side"],
    )
    reset_scalar_record(
        iteration.meshes["core_cell_normal_y"],
        cell_normal_y,
        ["cell", "local_side"],
    )
    reset_scalar_record(iteration.meshes["core_surface"], surface, ["cell"])
    reset_scalar_record(
        iteration.meshes["core_beta_volume"],
        beta_volume,
        ["cell", "layer"],
    )
    reset_scalar_record(
        iteration.meshes["core_cladding_cell_type"],
        cladding_cell_type,
        ["cell"],
    )
    reset_scalar_record(
        iteration.meshes["core_reflectivity"],
        reflectivity,
        ["cell", "interface"],
    )
    reset_scalar_record(
        iteration.meshes["core_point_beta"],
        point_beta,
        ["point", "level"],
    )
    reset_scalar_record(iteration.meshes["core_refractive_index"], refractive_index, ["interface"])
    reset_scalar_record(iteration.meshes["core_lambda_absorption"], lambda_absorption, ["wavelength"])
    reset_scalar_record(iteration.meshes["core_lambda_emission"], lambda_emission, ["wavelength"])
    reset_scalar_record(iteration.meshes["core_sigma_absorption"], sigma_absorption, ["wavelength"])
    reset_scalar_record(iteration.meshes["core_sigma_emission"], sigma_emission, ["wavelength"])

    iteration.close()
    series.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stream", nargs="?", default="hase_input.sst")
    args = parser.parse_args()

    if "sst" not in io.file_extensions:
        print("SST engine not available in this openpmd_api/ADIOS2 build.")
        return 0

    write_random_core_mesh(args.stream)
    return 0


if __name__ == "__main__":
    sys.exit(main())
