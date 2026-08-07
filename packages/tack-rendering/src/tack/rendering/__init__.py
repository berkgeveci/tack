"""tack.rendering — Path tracing and volume rendering for in situ visualization.

Provides GPU-accelerated renderers that operate directly on tack fields.
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

ColorBar, AxisIndicator, TextOverlay, annotate
    Post-render annotations drawn on numpy images.

render(canvas, scene, camera, ...)
    Unified renderer — dispatches to path tracing for surface actors
    and ray casting for volumes based on scene contents.

render_volume(canvas, volume, camera, ...)
    Ray-cast a single volume directly (without a Scene).
"""

from tack.rendering.annotate import (
    AxisIndicator,
    ColorBar,
    TextOverlay,
    annotate,
)
from tack.rendering.camera import OrthographicCamera, PerspectiveCamera
from tack.rendering.canvas import Canvas
from tack.rendering.colortable import ColorTable
from tack.rendering.render import render
from tack.rendering.scene import (
    Actor,
    DirectionalLight,
    Material,
    PointLight,
    Scene,
    compute_normals,
)
from tack.rendering.volume import TransferFunction, Volume, render_volume

__all__ = [
    "Actor",
    "AxisIndicator",
    "Canvas",
    "ColorBar",
    "ColorTable",
    "DirectionalLight",
    "Material",
    "OrthographicCamera",
    "PerspectiveCamera",
    "PointLight",
    "Scene",
    "TextOverlay",
    "TransferFunction",
    "Volume",
    "annotate",
    "compute_normals",
    "render",
    "render_volume",
]
