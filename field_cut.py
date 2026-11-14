"""Electric field cuts from the VTK files of the Monte Carlo runs.

The runs write the field as legacy ASCII VTK unstructured grids (VisIt/EField_vtk_files/efieldmag_<i>.vtk,
one per subhistory) on a tetrahedral mesh, with one constant vector per tetrahedron (cell data).
A cut through cell data is a staircase, so the field is first moved to the mesh points as ParaView's
"Cell Data to Point Data" does (average of the cells sharing each point) and then sampled along the line
by linear interpolation inside the tetrahedra, as ParaView's "Plot Over Line".
"""
import glob
import os
import re

import numpy as np
from scipy.spatial import cKDTree

VTK_TETRA = 10


def read_vtk_cell_vectors(path):
    """Points (N, 3), tetrahedra (M, 4) and cell vectors (M, 3) of a legacy ASCII VTK unstructured grid.

    The vectors are the first VECTORS array of the CELL_DATA section.
    """
    with open(path) as f:
        tokens = f.read().split()

    def after(keyword, start=0):
        return tokens.index(keyword, start)

    i = after("POINTS")
    n_points = int(tokens[i + 1])
    points = np.array(tokens[i + 3:i + 3 + 3 * n_points], dtype=float).reshape(n_points, 3)

    i = after("CELLS", i)
    n_cells, size = int(tokens[i + 1]), int(tokens[i + 2])
    cells = np.array(tokens[i + 3:i + 3 + size], dtype=np.int64)
    if size != 5 * n_cells or np.any(cells[::5] != 4):
        raise ValueError(f"{path}: only tetrahedral meshes are supported")
    tets = cells.reshape(n_cells, 5)[:, 1:]

    i = after("CELL_TYPES", i)
    types = np.array(tokens[i + 2:i + 2 + n_cells], dtype=int)
    if np.any(types != VTK_TETRA):
        raise ValueError(f"{path}: only tetrahedral meshes are supported")

    i = after("VECTORS", after("CELL_DATA", i))
    vectors = np.array(tokens[i + 3:i + 3 + 3 * n_cells], dtype=float).reshape(n_cells, 3)
    return points, tets, vectors


def last_snapshot(run_dir, pattern="efieldmag_*.vtk"):
    """(path, index) of the field file of the last subhistory in <run_dir>/VisIt/EField_vtk_files."""
    files = glob.glob(os.path.join(run_dir, "VisIt", "EField_vtk_files", pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {run_dir}/VisIt/EField_vtk_files")
    index = {f: int(re.search(r"_(\d+)\.vtk$", f).group(1)) for f in files}
    path = max(files, key=index.get)
    return path, index[path]


def cell_to_point(tets, cell_values, n_points):
    """Point values = mean of the values of the cells sharing each point (ParaView's Cell Data to Point Data)."""
    sums = np.zeros((n_points,) + cell_values.shape[1:])
    counts = np.zeros(n_points)
    np.add.at(sums, tets.ravel(), np.repeat(cell_values, tets.shape[1], axis=0))
    np.add.at(counts, tets.ravel(), 1)
    with np.errstate(invalid="ignore"):
        return sums / counts.reshape((-1,) + (1,) * (cell_values.ndim - 1))


def sample_line(points, tets, point_values, start, end, n_samples=1000, n_candidates=64):
    """Point data interpolated linearly along the segment start-end (ParaView's Plot Over Line).

    Returns (distance from start, values); values are nan where the line is outside the mesh.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    s = np.linspace(0, 1, n_samples)
    samples = start + s[:, None] * (end - start)

    v0 = points[tets[:, 0]]
    edges = np.stack([points[tets[:, k]] - v0 for k in (1, 2, 3)], axis=-1)  # (M, 3, 3), edge vectors as columns
    inv_edges = np.linalg.inv(edges)

    def barycentric(sample, cells):
        lam = np.einsum("...ij,...j->...i", inv_edges[cells], sample - v0[cells])
        return np.concatenate([1 - lam.sum(axis=-1, keepdims=True), lam], axis=-1)

    tree = cKDTree(points[tets].mean(axis=1))
    _, nearest = tree.query(samples, k=min(n_candidates, len(tets)))
    values = np.full((n_samples,) + point_values.shape[1:], np.nan)
    tol = -1e-9
    for j, sample in enumerate(samples):
        weights = barycentric(sample, nearest[j])
        inside = np.flatnonzero(np.all(weights >= tol, axis=1))
        if len(inside):
            cell, w = nearest[j][inside[0]], weights[inside[0]]
        else:  # not among the nearest cells: search all of them
            weights = barycentric(sample, np.arange(len(tets)))
            inside = np.flatnonzero(np.all(weights >= tol, axis=1))
            if not len(inside):
                continue
            cell, w = inside[0], weights[inside[0]]
        values[j] = np.tensordot(w, point_values[tets[cell]], axes=1)
    return s * np.linalg.norm(end - start), values


def field_magnitude_cut(vtk_file, start, end, n_samples=1000):
    """(distance from start, |E| smoothed with Cell Data to Point Data) along the segment start-end."""
    points, tets, field = read_vtk_cell_vectors(vtk_file)
    point_field = cell_to_point(tets, field, len(points))
    distance, values = sample_line(points, tets, point_field, start, end, n_samples)
    return distance, np.linalg.norm(values, axis=1)
