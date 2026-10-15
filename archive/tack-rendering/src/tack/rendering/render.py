"""Unified render dispatcher.

Routes rendering to the appropriate backend based on scene contents
and actor render modes.
"""


def render(canvas, scene, camera, samples=1, max_bounces=3,
           light_position=None, light_intensity=100.0,
           background=(0.05, 0.05, 0.1), point_size=3.0):
    """Render the scene into the canvas.

    Dispatches to the path tracer for solid surface actors, the rasterizer
    for wireframe/point actors, and the volume ray caster for volumes.

    Args:
        canvas: Canvas to render into.
        scene: Scene containing actors, volumes, and lights.
        camera: PerspectiveCamera or OrthographicCamera.
        samples: Samples per pixel for path tracing.
        max_bounces: Maximum path depth for path tracing.
        light_position: Light override for path tracing.
        light_intensity: Light intensity override for path tracing.
        background: RGB background color in [0, 1].
        point_size: Pixel radius for point rendering.
    """
    has_volumes = len(scene.volumes) > 0

    if not scene.actors and not has_volumes:
        return

    # Classify actors by render mode
    solid_actors = [a for a in scene.actors
                    if getattr(a, 'render_mode', 'solid') == 'solid']
    raster_actors = [a for a in scene.actors
                     if getattr(a, 'render_mode', 'solid') in
                     ('wireframe', 'points')]

    if solid_actors:
        # Path tracer handles solid surfaces and volumes
        from tack.rendering.pathtrace import render as _render_pathtrace
        _render_pathtrace(canvas, _solid_scene(scene, solid_actors), camera,
                          samples=samples, max_bounces=max_bounces,
                          light_position=light_position,
                          light_intensity=light_intensity,
                          background=background)
    elif has_volumes:
        # No solid surfaces: standalone ray caster
        from tack.rendering.volume import render_volume
        for vol in scene.volumes:
            render_volume(canvas, vol, camera, background=background)

    if raster_actors:
        # Rasterize wireframe/point actors
        from tack.rendering.rasterize import render_raster

        # Build a temporary scene with only raster actors
        from tack.rendering.scene import Scene
        raster_scene = Scene()
        for a in raster_actors:
            raster_scene.add(a)
        # Over a path-traced or ray-cast image the rasterizer composites,
        # depth-tested against the solid surfaces.  A ray-cast volume has no
        # depth, so rasterized actors are drawn over it.
        render_raster(canvas, raster_scene, camera,
                      background=background,
                      point_size=point_size,
                      composite=bool(solid_actors) or has_volumes,
                      surface_depth=bool(solid_actors))


def _solid_scene(scene, solid_actors):
    """The scene the path tracer sees: solid actors, volumes and lights.

    Wireframe and point actors are rasterized, so they must not also be
    traced as surfaces.  The sub-scene is kept on the parent so its merged
    geometry and BVH stay cached across frames.
    """
    if len(solid_actors) == len(scene.actors):
        return scene
    key = (scene._version, tuple(id(a) for a in solid_actors))
    cached = getattr(scene, '_solid_subscene', None)
    if cached is None or cached[0] != key:
        from tack.rendering.scene import Scene
        sub = Scene()
        sub.actors = list(solid_actors)
        cached = (key, sub)
        scene._solid_subscene = cached
    sub = cached[1]
    sub.volumes = scene.volumes
    sub.lights = scene.lights
    return sub
