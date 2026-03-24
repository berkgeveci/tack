"""Unified render dispatcher.

Routes rendering to the appropriate backend based on scene contents.
When both surfaces and volumes are present, they are rendered together
in a single path-trace pass with correct depth compositing.
"""


def render(canvas, scene, camera, samples=1, max_bounces=3,
           light_position=None, light_intensity=100.0,
           background=(0.05, 0.05, 0.1)):
    """Render the scene into the canvas.

    Surfaces and volumes are rendered together in a single pass — volume
    samples are correctly depth-composited against surface geometry.
    Volume-only scenes (no surface actors) use the standalone volume
    ray caster.

    Args:
        canvas: Canvas to render into.
        scene: Scene containing actors, volumes, and lights.
        camera: PerspectiveCamera.
        samples: Samples per pixel for path tracing (ignored for volume-only).
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
        # Path tracer handles both surfaces and volumes in one pass
        from pgc.rendering.pathtrace import render as _render_pathtrace
        _render_pathtrace(canvas, scene, camera,
                          samples=samples, max_bounces=max_bounces,
                          light_position=light_position,
                          light_intensity=light_intensity,
                          background=background)
    elif has_volumes:
        # Volume-only: use standalone ray caster (no BVH needed)
        from pgc.rendering.volume import render_volume
        for vol in scene.volumes:
            render_volume(canvas, vol, camera, background=background)
