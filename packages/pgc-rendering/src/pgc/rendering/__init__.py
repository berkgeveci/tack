"""pgc.rendering — Path tracing and volume rendering for in situ visualization.

Provides GPU-accelerated renderers that operate directly on pgc fields.
Geometry from algorithms like flying_edges can be rendered without host copies.

Public API
----------
PerspectiveCamera
    Perspective projection camera.

Canvas
    Framebuffer for rendered images.

Scene, Actor, PointLight
    Scene graph objects.

ColorTable
    Maps scalar fields to RGB colors via sampled lookup tables.

Volume, TransferFunction
    Uniform-grid volume with RGBA transfer function for ray casting.

render(canvas, scene, camera, ...)
    Unified renderer — dispatches to path tracing for surface actors
    and ray casting for volumes based on scene contents.

render_volume(canvas, volume, camera, ...)
    Ray-cast a single volume directly (without a Scene).
"""

from pgc.rendering.camera import PerspectiveCamera
from pgc.rendering.canvas import Canvas
from pgc.rendering.colortable import ColorTable
from pgc.rendering.scene import Scene, Actor, PointLight, compute_normals
from pgc.rendering.render import render
from pgc.rendering.volume import Volume, TransferFunction, render_volume

__all__ = [
    "PerspectiveCamera",
    "Canvas",
    "ColorTable",
    "Scene",
    "Actor",
    "PointLight",
    "compute_normals",
    "render",
    "Volume",
    "TransferFunction",
    "render_volume",
]
