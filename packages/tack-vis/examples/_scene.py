# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The VTK examples' pictures (47-53, ``--png``), drawn offscreen.

An example describes its picture as layers: what to draw, in what color and
opacity, or colored by which array over which range. ``render`` draws them with
VTK's views and representations where this VTK has them (``vtkViewsScivis``,
VTK master: a ``vtkScivisView`` given a representation per layer by
``view.show``, its color maps shared by array name through the view's lookup
table manager, its scalar bars drawn for them), else with a renderer, mappers
and actors, as VTK 9.7.1 draws. The data and the camera are the same either
way; a layer colored by an array takes each way's own color map (the view's
color scheme, or blue to red), and only the view draws scalar bars and the
orientation axes.
"""

from dataclasses import dataclass

BACKGROUND = (0.12, 0.12, 0.16)


def has_views():
    """Whether this VTK has views and representations (VTK master)."""
    try:
        import vtkmodules.vtkViewsScivis  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class Layer:
    """Something to draw: a VTK dataset or tree of them, in ``color`` (r, g, b), or
    colored by the point (or with ``cells``, cell) array ``array``, a vector by
    its magnitude, over ``value_range`` (by default the data's); ``point_size``
    draws its points as spheres of that many pixels."""

    data: object
    color: tuple = (0.9, 0.9, 0.9)
    opacity: float = 1.0
    array: str = None
    cells: bool = False
    value_range: tuple = None
    point_size: float = None


def render(path, layers, size=(800, 600), azimuth=0.0, elevation=0.0, zoom=1.0, focus=None,
           background=BACKGROUND):
    """Draw ``layers`` and write the picture to the PNG file ``path``: the camera
    reset to everything (or to the bounds ``focus``), then turned by ``azimuth``
    and ``elevation`` (degrees) and zoomed by ``zoom``. Returns which way it drew,
    ``"views"`` or ``"actors"``."""
    import vtkmodules.vtkInteractionStyle
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

    camera = {"azimuth": azimuth, "elevation": elevation, "zoom": zoom, "focus": focus}
    if has_views():
        _with_views(path, layers, size, background, **camera)
        return "views"
    _with_actors(path, layers, size, background, **camera)
    return "actors"


def _aim(renderer, camera, azimuth, elevation, zoom, focus):
    if focus is None:
        renderer.ResetCamera()
    else:
        renderer.ResetCamera(*focus)
    camera.Azimuth(azimuth)
    camera.Elevation(elevation)
    camera.OrthogonalizeViewUp()
    camera.Zoom(zoom)


def _with_views(path, layers, size, background, **camera):
    from vtkmodules.vtkCommonCore import vtkScalarsToColors
    from vtkmodules.vtkViewsScivis import vtkScivisView

    view = vtkScivisView(size=size, background=background)
    view.render_window.SetOffScreenRendering(True)
    for layer in layers:
        representation = view.show(layer.data, color=layer.color, opacity=layer.opacity)
        if layer.array is None:
            # A representation colors by the active scalars until told otherwise.
            representation.ColorBySolidColor()
        else:
            # The view keeps one color map per array name; asked for before the
            # first render it is the layer's to set, range and all.
            table = view.lookup_table_manager[layer.array]
            table.vector_mode = vtkScalarsToColors.MAGNITUDE
            if layer.value_range is not None:
                table.SetRange(*layer.value_range)
            if layer.cells:
                representation.ColorByCellArray(layer.array)
            else:
                representation.ColorByPointArray(layer.array)
        if layer.point_size is not None:
            representation.representation = "points"
            representation.property.point_size = layer.point_size
            representation.property.render_points_as_spheres = True
    # A view resets its camera to the scene on its first render: aim it after.
    view.Render()
    _aim(view.renderer, view.camera, **camera)
    view.Render()
    if not view.exporter.SaveScreenshot(str(path)):
        raise RuntimeError(f"could not write {path}")


def _with_actors(path, layers, size, background, **camera):
    import vtkmodules.vtkCommonDataModel as dm
    from vtkmodules.vtkCommonCore import vtkLookupTable, vtkScalarsToColors
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkCompositePolyDataMapper,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkWindowToImageFilter,
    )

    renderer = vtkRenderer(background=background)
    tables = {}
    for layer in layers:
        # Polygonal data as it is, a tree block by block, any other dataset's surface.
        tree = isinstance(layer.data, dm.vtkCompositeDataSet)
        mapper = (vtkCompositePolyDataMapper if tree else vtkPolyDataMapper)(
            scalar_visibility=layer.array is not None)
        if isinstance(layer.data, dm.vtkPolyData):
            mapper.SetInputDataObject(layer.data)
        else:
            layer.data >> vtkGeometryFilter() >> mapper
        if layer.array is not None:
            if layer.array not in tables:      # one map per array, as a view keeps
                table = tables[layer.array] = vtkLookupTable(hue_range=(0.667, 0.0))
                table.vector_mode = vtkScalarsToColors.MAGNITUDE
            mapper.lookup_table = tables[layer.array]
            if layer.cells:
                mapper.SetScalarModeToUseCellFieldData()
            else:
                mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray(layer.array)
            mapper.scalar_range = (layer.value_range if layer.value_range is not None
                                   else _range(layer.data, layer.array, layer.cells))
        actor = vtkActor(mapper=mapper)
        actor.property.color = layer.color
        actor.property.opacity = layer.opacity
        if layer.point_size is not None:
            actor.property.representation = 0          # points
            actor.property.point_size = layer.point_size
            actor.property.render_points_as_spheres = True
        renderer.AddActor(actor)
    window = vtkRenderWindow(off_screen_rendering=True, size=size)
    window.AddRenderer(renderer)
    _aim(renderer, renderer.active_camera, **camera)
    window.Render()
    # A PNG writer called on its input writes raw pixels (VTK 9.7.1): connect it
    # and Write().
    png = vtkPNGWriter(file_name=str(path))
    vtkWindowToImageFilter(input=window) >> png
    png.Write()


def _range(data, name, cells):
    """The range of the array ``name`` (a vector's magnitudes) over a dataset or
    every dataset of a tree."""
    import numpy as np
    import vtkmodules.vtkCommonDataModel as dm

    leaves = [data]
    if isinstance(data, dm.vtkCompositeDataSet):
        leaves, iterator = [], data.NewIterator()
        iterator.InitTraversal()
        while not iterator.IsDoneWithTraversal():
            leaves.append(iterator.GetCurrentDataObject())
            iterator.GoToNextItem()
    low, high = np.inf, -np.inf
    for leaf in leaves:
        attributes = leaf.cell_data if cells else leaf.point_data
        if name not in attributes.keys():
            continue
        values = np.asarray(attributes[name], np.float64)
        if values.size:
            values = np.linalg.norm(values, axis=1) if values.ndim > 1 else values
            low, high = min(low, values.min()), max(high, values.max())
    return (float(low), float(high)) if low <= high else (0.0, 1.0)
