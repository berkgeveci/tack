"""Helpers shared by the tack-vis algorithm tests.

Analytic scalar fields on uniform grids, plus mesh-topology measures that
turn "is this isosurface right?" into golden numbers: watertightness,
Euler characteristic, and enclosed volume.
"""

from collections import Counter

import numpy as np

import tack
from tack.algorithms.flying_edges import UniformGrid


def make_grid(n, lo=-1.5, hi=1.5, nz=None):
    """Cubic uniform grid of n×n×nz cells spanning [lo, hi] in x and y.

    z spans the same spacing but only nz cells, so a domain can be split
    into stacked blocks that share their interface node plane.
    """
    nz = n if nz is None else nz
    h = (hi - lo) / n
    return UniformGrid(n, n, nz, lo, lo, lo, h, h, h)


def node_coords(grid):
    """(n_nodes, 3) node coordinates in flying-edges node order.

    Node index is k*nxy_p1 + j*nx_p1 + i, so k varies slowest.
    """
    i = np.arange(grid.nx + 1)
    j = np.arange(grid.ny + 1)
    k = np.arange(grid.nz + 1)
    kk, jj, ii = np.meshgrid(k, j, i, indexing="ij")
    return np.stack([
        (grid.x0 + ii * grid.dx).ravel(),
        (grid.y0 + jj * grid.dy).ravel(),
        (grid.z0 + kk * grid.dz).ravel(),
    ], axis=1)


def upload(values, dtype=tack.f32, np_dtype=np.float32):
    """Copy a numpy array into a fresh 1-D tack field."""
    f = tack.field(dtype=dtype, shape=(values.size,))
    f.from_numpy(np.ascontiguousarray(values, dtype=np_dtype))
    return f


def sphere_field(grid, radius=1.0, inside_positive=True):
    """Scalar field whose isovalue-0 surface is a sphere at the origin.

    With inside_positive=True the field is radius² − |p|², so the region
    flying edges treats as "above" the isovalue is the sphere's interior
    and the emitted triangles wind counter-clockwise seen from outside.
    """
    c = node_coords(grid)
    r2 = (c ** 2).sum(axis=1)
    return upload(radius ** 2 - r2 if inside_positive else r2 - radius ** 2)


def plane_field(grid):
    """Scalar field equal to the node's z coordinate (exactly linear)."""
    return upload(node_coords(grid)[:, 2])


def undirected_edge_counts(conn):
    """How many triangles use each undirected edge."""
    counts = Counter()
    for v0, v1, v2 in conn:
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            counts[(min(a, b), max(a, b))] += 1
    return counts


def directed_edge_counts(conn):
    """How many triangles use each directed edge (2 → inconsistent winding)."""
    counts = Counter()
    for v0, v1, v2 in conn:
        counts[(v0, v1)] += 1
        counts[(v1, v2)] += 1
        counts[(v2, v0)] += 1
    return counts


def enclosed_volume(result):
    """Signed volume enclosed by a closed triangle mesh (divergence theorem).

    Positive when triangles wind counter-clockwise seen from outside.
    """
    p = result["points"].astype(np.float64)
    c = result["conn"]
    tets = np.einsum("ij,ij->i", p[c[:, 0]], np.cross(p[c[:, 1]], p[c[:, 2]]))
    return tets.sum() / 6.0


def euler_characteristic(result):
    """V − E + F; 2 for any closed surface of genus 0."""
    v = result["total_points"]
    e = len(undirected_edge_counts(result["conn"]))
    f = result["total_tris"]
    return v - e + f
