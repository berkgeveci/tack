"""Unified render dispatcher.

Routes rendering to the appropriate backend based on scene contents:
surface actors go through the path tracer, volumes through the ray caster.
"""


def render(canvas, scene, camera, samples=1, max_bounces=3,
           light_position=None, light_intensity=100.0,
           background=(0.05, 0.05, 0.1)):
    """Render the scene into the canvas.

    Dispatches to path tracing for surface actors and ray casting for
    volumes.  If both are present, surfaces are rendered first and
    volumes are composited on top.

    Args:
        canvas: Canvas to render into.
        scene: Scene containing actors, volumes, and lights.
        camera: PerspectiveCamera.
        samples: Samples per pixel for path tracing (ignored for volumes).
        max_bounces: Maximum path depth for path tracing.
        light_position: Light override for path tracing.
        light_intensity: Light intensity override for path tracing.
        background: RGB background color in [0, 1].
    """
    has_surfaces = len(scene.actors) > 0
    has_volumes = len(scene.volumes) > 0

    if not has_surfaces and not has_volumes:
        return

    if has_surfaces:
        from pgc.rendering.pathtrace import render as _render_pathtrace
        _render_pathtrace(canvas, scene, camera,
                          samples=samples, max_bounces=max_bounces,
                          light_position=light_position,
                          light_intensity=light_intensity,
                          background=background)

    if has_volumes:
        from pgc.rendering.volume import render_volume
        # If surfaces were rendered, volumes composite over them;
        # otherwise volumes render against the background.
        vol_bg = background if not has_surfaces else (0.0, 0.0, 0.0)
        for vol in scene.volumes:
            render_volume(canvas, vol, camera, background=vol_bg)
