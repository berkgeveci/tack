# tack-rendering design notes (archived)

These sections were in the repository's CLAUDE.md while tack-rendering was a
maintained package. They describe the code as it was archived (2026-10-15)
and are kept with it for whoever revives it.

## Rasterization passes

Rasterized points and wireframes use three completed passes per actor:
atomic depth reduction, atomic lowest-ID selection at the completed depth,
and one color writer per pixel. Never combine atomic depth updates with
ordinary depth reads and competing RGB stores in one geometry dispatch.
`rendering/rasterize.py` uses cached i32 winner scratch; equal-depth
primitives select the lowest index, while equal-depth actors overwrite in
scene order. See `test_raster_winners.py` for contention across workgroups,
forced CPU workers, ties, repeated frames and compositing.

## ColorTable (tack.rendering)

`ColorTable` maps per-vertex scalar fields to RGB colors via a sampled lookup table. Presets: `viridis`, `cool_to_warm`, `inferno`, `plasma`, `grayscale`, `rainbow`. The `Actor` class accepts `scalars` (tack.field or numpy) + `color_table` (ColorTable) to enable scalar field coloring. During `Scene._prepare()`, scalars are mapped to per-vertex colors on GPU via linear interpolation into the lookup table. The pathtrace kernel's existing per-vertex color interpolation handles the rest — no kernel changes needed.

## Multiple lights (tack.rendering)

Multiple `PointLight` instances can be added to a `Scene`. Each light contributes independently with its own shadow ray. Light data is packed into a flat field (7 floats per light: x, y, z, intensity, r, g, b) and passed to the pathtrace kernel, which loops over all lights in the direct-illumination section. Light color (previously ignored) is now applied to each light's contribution. The `render()` function's `light_position` kwarg still works as a single-light override for backward compatibility.

## Volume rendering (tack.rendering)

`Volume` represents a uniform-grid scalar field for ray-casting volume rendering. `TransferFunction` maps scalars to RGBA (reuses ColorTable preset colors + user-defined opacity function). `render_volume(canvas, volume, camera)` ray marches through the volume with front-to-back compositing, trilinear interpolation via `tack.texture3d`, and opacity-corrected compositing.

## Unified render dispatcher (tack.rendering)

`render()` in `render.py` is the unified entry point. When a scene has both surfaces and volumes, the path tracer integrates volume ray marching directly — for each primary ray, after BVH traversal finds the closest surface hit, the volume is marched from the ray origin to the hit distance with front-to-back compositing. Volume opacity attenuates the throughput so surfaces are correctly visible through semi-transparent volumes. Volume-only scenes use the standalone ray caster. `render_volume()` remains available for direct single-volume rendering without a Scene.

## Annotations (tack.rendering)

`annotate(image, annotations)` draws overlays on `(H, W, 4) uint8` numpy arrays (from `Canvas.to_numpy()`). Three annotation types: `ColorBar` (gradient strip + tick labels from ColorTable/TransferFunction), `AxisIndicator` (projected XYZ axes from camera orientation), `TextOverlay` (arbitrary text). Lines and rectangles use pure numpy; text requires Pillow (soft dependency, gracefully skipped if absent). Camera stores `_right`, `_up`, `_forward` basis vectors for the axis indicator.

## Orthographic camera (tack.rendering)

`OrthographicCamera` generates parallel rays — all rays have the same direction, but origins vary per pixel. Uses the same scalar attribute pattern as `PerspectiveCamera` with added `odx_x/y/z` and `ody_x/y/z` origin-per-pixel deltas. `PerspectiveCamera` sets these to zero for backward compatibility. Works with both path tracer and volume renderer. The `view_height` parameter controls the world-space height of the view rectangle.

## Material system (tack.rendering)

`Material` class with three types: `MATTE` (Lambertian diffuse, default), `SPECULAR` (perfect mirror reflection), `TRANSPARENT` (glass with Snell's law refraction + Schlick Fresnel). Set per-actor via `Actor(..., material=Material(Material.SPECULAR))`. Per-triangle material IDs (`mat_ids` i32 field) index into a flat material table (`mat_table` f32 field, stride 4: type, ior, reserved, reserved). Direct lighting is only computed for matte surfaces. Specular bounces reflect perfectly; transparent surfaces refract or reflect based on Fresnel probability using Halton random numbers.

## Wireframe and point rendering (tack.rendering)

`Actor` accepts `render_mode="solid"` (default, path traced), `"wireframe"` (triangle edges), or `"points"` (vertex discs). Wireframe and point actors are dispatched to GPU rasterization kernels in `rasterize.py` instead of the path tracer. Uses an MVP projection matrix passed as a flat f32 field, Bresenham line rasterization for wireframe, disc rasterization for points, and `tack.atomic_min` depth testing. Supports perspective and orthographic cameras, per-actor colors, scalar coloring, and configurable point size.

In a scene that also has solid actors or volumes, `render()` path traces a solid-only sub-scene (cached on the parent so the BVH survives across frames) and then calls `render_raster(..., composite=True)`: wireframe and point actors are drawn into scratch buffers and laid over the existing image instead of clearing it. With solids, the path tracer's primary-ray distances are converted to the rasterizer's clip-space depth (rebuilding each pixel's jittered first-sample ray) so surfaces hide rasterized geometry behind them; `_SURFACE_DEPTH_BIAS` pushes surfaces back 0.5% so a wireframe lying on its own surface is drawn. A ray-cast volume has no depth, so rasterized actors are drawn over it. `canvas.depth` keeps the path tracer's ray distances in a mixed scene.
