# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Mesh metrics kept with the mesh: computed once a mesh, again when it changes."""

import numpy as np
import pytest

import tack
import tack.dataset as td


def _solid():
    tets = td.tetrahedralize(td.uniform_grid((4, 3, 3), spacing=(0.5, 0.5, 0.5)))
    return td.DataSet(tets.topology, tets.geometry.values.numpy())


def test_cell_and_face_geometry_once_a_mesh(backend):
    solid = _solid()
    measure = td.cell_geometry(solid).measure
    normal = td.face_geometry(solid).normal
    solid.fields["u"] = td.Field(td.Values(solid, "cells"),
                                 tack.array(np.ones(solid.num_cells, np.float32)))
    assert td.cell_geometry(solid).measure is measure
    assert td.face_geometry(solid).normal is normal
    stretched = solid.geometry.values.numpy() * np.float32(2.0)
    tack.copyto(solid.geometry.values, stretched)
    again = td.cell_geometry(solid).measure
    assert again is not measure
    np.testing.assert_allclose(again.values.numpy().sum(), 8 * measure.values.numpy().sum(),
                               rtol=1e-5)
    assert td.face_geometry(solid).normal is not normal


def test_the_boundary_of_one_mesh_has_its_normals_once(backend):
    solid = _solid()
    first = td.extract_surface(solid)
    solid.fields["u"] = td.Field(td.Values(solid, "points"),
                                 tack.array(np.ones(solid.num_points, np.float32)))
    again = td.extract_surface(solid)
    assert again.geometry is first.geometry
    assert "u" in again.fields
    normals = td.surface_normals(first).fields["cell_normals"].values
    assert td.surface_normals(again).fields["cell_normals"].values is normals
    tack.copyto(solid.geometry.values, solid.geometry.values.numpy() * np.float32(2.0))
    moved = td.extract_surface(solid)
    assert moved.geometry is not first.geometry
    assert td.surface_normals(moved).fields["cell_normals"].values is not normals
    np.testing.assert_allclose(td.integrate(moved).measure, 4 * td.integrate(first).measure,
                               rtol=1e-5)


def test_surface_normals_are_the_same_kept_or_not(backend):
    surface = td.extract_surface(_solid())
    kept = td.surface_normals(surface)
    fresh = td.surface_normals(td.DataSet(surface.topology, surface.geometry.values.numpy()))
    for name in ("normals", "cell_normals"):
        np.testing.assert_allclose(kept.fields[name].values.numpy(),
                                   fresh.fields[name].values.numpy(), atol=1e-6)
    assert td.surface_normals(surface, weights="area").fields["normals"].values is not \
        kept.fields["normals"].values
    with pytest.raises(ValueError):
        td.surface_normals(_solid())
