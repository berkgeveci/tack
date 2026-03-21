"""pgc.rendering — Path tracing renderer for in situ visualization.

Provides a GPU-accelerated path tracer that operates directly on pgc fields.
Geometry from algorithms like flying_edges can be rendered without host copies.

Public API
----------
PerspectiveCamera
    Perspective projection camera.

Canvas
    Framebuffer for rendered images.

Scene, Actor, PointLight
    Scene graph objects.

render(canvas, scene, camera, ...)
    Path trace the scene into a canvas.
"""

from pgc.rendering.camera import PerspectiveCamera
from pgc.rendering.canvas import Canvas
from pgc.rendering.scene import Scene, Actor, PointLight
from pgc.rendering.pathtrace import render

__all__ = [
    "PerspectiveCamera",
    "Canvas",
    "Scene",
    "Actor",
    "PointLight",
    "render",
]
