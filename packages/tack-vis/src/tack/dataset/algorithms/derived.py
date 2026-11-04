# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The fields a dataset derives from its own geometry where it stores none of their
names (``register_derived_field``): ``normals``, ``measures`` and ``centroids``,
each at the locations it has."""

from tack.dataset.algorithms._common import register_derived_field


def _surface_normals(name, on):
    def compute(data):
        from tack.dataset.algorithms.surface_normals import surface_normals
        from tack.dataset.dataset import Field
        from tack.dataset.spaces import Values

        made = surface_normals(data).fields[name]
        return Field(Values(data, on), made.values, made.unit)
    return compute


def _face_geometry(part):
    def compute(data):
        from tack.dataset.algorithms.face_geometry import face_geometry

        return getattr(face_geometry(data), part)
    return compute


def _cell_geometry(part):
    def compute(data):
        from tack.dataset.algorithms.cell_geometry import cell_geometry

        return getattr(cell_geometry(data), part)
    return compute


register_derived_field("normals", _surface_normals("normals", "points"), at="points")
register_derived_field("normals", _surface_normals("cell_normals", "cells"), at="cells")
register_derived_field("normals", _face_geometry("normal"), at="faces")
register_derived_field("measures", _cell_geometry("measure"), at="cells")
register_derived_field("measures", _face_geometry("area"), at="faces")
register_derived_field("centroids", _cell_geometry("centroid"), at="cells")
