"""Public-API raster differential probe (CX8 / TA-23).

Renders the same point and wireframe scenes through the public rendering
API only -- ``Actor``, ``Scene``, ``Canvas``, ``OrthographicCamera`` and
``render_raster`` -- and checks every frame against an oracle that does not
come from the run under test:

* **Reference composition.** Each candidate actor is rendered alone, which
  cannot race, and the expected frame is composed per pixel from those
  references: the candidate with the smallest depth wins, its colour and
  depth are expected, and candidates at exactly equal depth form a set any
  of whose colours is acceptable. Uncovered pixels must match an empty
  render, including depth.
* **Candidate-set colours.** Where every primitive of one actor contends
  for the same pixels (the canonical 4097 coincident points), each point
  carries a distinct, index-coded colour, so a pixel's RGB must equal one
  primitive's colour exactly. A mixed pixel -- red from one point, green
  from another -- is the race signature CX8 fixed, and it cannot hide in a
  checksum.

Because the probe uses only interfaces that exist in every tree since the
rasterizer gained points and wireframes, the identical bytes run unmodified
against the pre-fix parent, the fix and the current head. It prints the
module it actually imported and a fingerprint of that module's source so a
run cannot silently test the wrong tree.

usage:
    python benchmarks/raster_differential.py --arch cuda [--repeats 20]
                                             [--json out.json] [--label name]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
import time

import numpy as np

SIZE = 32          # pixels, square
VIEW = 4.0         # world units across the view; world x in [-2, 2] -> 32 px
POINT_SIZE = 3     # disc radius in pixels -> 29 pixels per point
COUNT = 4097       # canonical coincident-point count: 16 full groups + 1
BG = (0.0, 0.0, 0.0)

_ARCHES = ("cpu", "metal", "cuda", "hip", "level_zero")


# ── setup ────────────────────────────────────────────────────────────────

def init_backend(arch: str):
    """Initialize the requested backend and refuse to continue otherwise."""
    import tack
    from tack.runtime.dispatch import get_backend
    tack.init(arch=getattr(tack, arch))
    backend = get_backend()
    if backend.name != arch:
        raise SystemExit(f"requested {arch}, got {backend.name}")
    print(f"Required backend: {backend.name}", flush=True)
    return backend


def source_identity() -> dict:
    """Which tree is under test: import paths and a source fingerprint."""
    import tack
    from tack.rendering import rasterize
    path = rasterize.__file__
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return {"tack": tack.__file__, "rasterize": path,
            "rasterize_sha256": digest[:16]}


def _rendering():
    from tack.rendering import Actor, Canvas, OrthographicCamera, Scene
    from tack.rendering.rasterize import render_raster
    return Actor, Canvas, OrthographicCamera, Scene, render_raster


def _field(values, dtype=None):
    import tack
    dtype = dtype or tack.f32
    values = np.asarray(values, dtype=dtype.numpy_dtype).ravel()
    field = tack.field(dtype, values.shape)
    field.from_numpy(values)
    return field


def _camera():
    _, _, OrthographicCamera, _, _ = _rendering()
    return OrthographicCamera(position=(0, 0, 5), look_at=(0, 0, 0),
                              view_height=VIEW, width=SIZE, height=SIZE)


def _planes(canvas):
    rgb = np.column_stack([canvas.color_r.to_numpy(), canvas.color_g.to_numpy(),
                           canvas.color_b.to_numpy()])
    return rgb, canvas.depth.to_numpy().copy()


def _point_actor(xyz, colors):
    import tack
    Actor, *_ = _rendering()
    pts = np.asarray(xyz, np.float32).reshape(-1, 3)
    return Actor(_field(pts), _field([0, 0, 0], tack.i32),
                 point_colors=np.asarray(colors, np.float32).reshape(-1, 3),
                 render_mode="points")


def _line_actor(p0, p1, color):
    """A degenerate triangle whose three edges all lie on the segment p0-p1."""
    import tack
    Actor, *_ = _rendering()
    mid = (np.asarray(p0) + np.asarray(p1)) / 2.0
    pts = np.asarray([p0, mid, p1], np.float32)
    return Actor(_field(pts), _field([0, 1, 2], tack.i32), color=color,
                 render_mode="wireframe")


def _render(actors, canvas=None, **kw):
    _, Canvas, _, Scene, render_raster = _rendering()
    scene = Scene()
    for a in actors:
        scene.add(a)
    canvas = canvas or Canvas(SIZE, SIZE)
    with contextlib.redirect_stdout(io.StringIO()):     # the renderer prints timings
        render_raster(canvas, scene, _camera(), background=BG,
                      point_size=POINT_SIZE, **kw)
    return canvas


# ── oracle ───────────────────────────────────────────────────────────────

class Expected:
    """Per-pixel expectation composed from reference renders."""

    def __init__(self, references, empty):
        # references: list of (rgb, depth, covered) for each candidate alone
        self.empty_rgb, self.empty_depth = empty
        n = SIZE * SIZE
        self.depth = self.empty_depth.copy()
        self.candidates = [set() for _ in range(n)]
        self.covered = np.zeros(n, bool)
        depths = np.full((max(len(references), 1), n), np.inf)
        for k, (rgb, depth, cov) in enumerate(references):
            depths[k, cov] = depth[cov]
        best = depths.min(axis=0)
        for pid in range(n):
            if np.isinf(best[pid]):
                continue
            self.covered[pid] = True
            self.depth[pid] = best[pid]
            for k, (rgb, depth, cov) in enumerate(references):
                if cov[pid] and depths[k, pid] == best[pid]:
                    self.candidates[pid].add(tuple(rgb[pid]))

    def check(self, rgb, depth, label):
        """Return None when the frame matches, else a description."""
        for pid in range(SIZE * SIZE):
            got = tuple(rgb[pid])
            if self.covered[pid]:
                if got not in self.candidates[pid]:
                    return (f"{label}: pixel {pid} colour {got} not among "
                            f"{sorted(self.candidates[pid])}")
                if depth[pid] != self.depth[pid]:
                    return (f"{label}: pixel {pid} depth {depth[pid]!r} "
                            f"expected {self.depth[pid]!r}")
            else:
                if got != tuple(self.empty_rgb[pid]):
                    return f"{label}: uncovered pixel {pid} coloured {got}"
                if depth[pid] != self.empty_depth[pid]:
                    return f"{label}: uncovered pixel {pid} depth {depth[pid]!r}"
        return None


def _reference(actor):
    """Render one candidate alone. Coverage comes from the depth plane, so a
    primitive whose colour happens to equal the background still counts."""
    rgb, depth = _planes(_render([actor]))
    _, empty_depth = _empty()
    covered = depth != empty_depth
    return rgb, depth, covered


def _empty():
    return _planes(_render([]))


def _index_colors(n):
    """Distinct f32 colours, one per primitive, with the blue channel a hash
    of the other two: a pixel whose red came from one primitive and whose
    green came from another almost never forms a valid triple, so mixed
    channels -- the race signature -- cannot pass as a legitimate colour.
    Channel values are multiples of 1/64, exactly representable."""
    i = np.arange(n)
    r = i % 64
    g = (i // 64) % 64
    b = (3 * r + 5 * g + 7 * (i // 4096)) % 64
    # +1 keeps every colour away from the black background.
    return ((np.column_stack([r, g, b]) + 1) / 64.0).astype(np.float32)


_gamma_cache: dict = {}


def _as_rendered(value: float) -> float:
    """What the renderer stores for a channel value, measured once per value
    by rendering a single point of that colour: colour correction happens on
    the device, so the oracle learns the mapping instead of assuming it."""
    key = float(value)
    if key not in _gamma_cache:
        # Green and blue at full so the disc is visible even when the value
        # under test is the background's own 0.0; red carries the value.
        rgb, depth = _planes(_render([_point_actor([[0, 0, 0]], [[key, 1.0, 1.0]])]))
        _, empty_depth = _empty()
        covered = depth != empty_depth
        vals = np.unique(rgb[covered, 0])
        assert len(vals) == 1, vals
        _gamma_cache[key] = float(vals[0])
    return _gamma_cache[key]


def _rendered_palette(colors):
    """The set of colour triples the renderer stores for this palette."""
    channel = {v: _as_rendered(v) for v in np.unique(colors)}
    return {tuple(channel[float(c)] for c in row) for row in colors}


# ── scenarios ────────────────────────────────────────────────────────────
# Each returns (frames_checked, failures) where failures is a list of strings.

def scenario_coincident_points(repeats, winner_at):
    """4097 coincident points, one nearer; its colour must own all 29 pixels."""
    z = np.full(COUNT, -1.0, np.float32)
    z[winner_at] = 0.0
    xyz = np.zeros((COUNT, 3), np.float32)
    xyz[:, 2] = z
    colors = _index_colors(COUNT)
    actor = _point_actor(xyz, colors)
    # Reference: the winner alone (cannot race) and the losers alone.
    win_ref = _reference(_point_actor(xyz[[winner_at]], colors[[winner_at]]))
    lose_idx = np.setdiff1d(np.arange(COUNT), [winner_at])
    lose_ref = _reference(_point_actor(xyz[lose_idx], colors[lose_idx]))
    expected = Expected([win_ref, lose_ref], _empty())
    assert expected.covered.sum() == 29, expected.covered.sum()
    failures = []
    for f in range(repeats):
        rgb, depth = _planes(_render([actor]))
        err = expected.check(rgb, depth, f"frame {f}")
        if err:
            failures.append(err)
    return repeats, failures


def scenario_equal_depth_ties(repeats):
    """4097 coincident points at one depth: every pixel must carry exactly
    one primitive's colour (no channel mixing), and 29 pixels are covered."""
    xyz = np.zeros((COUNT, 3), np.float32)
    colors = _index_colors(COUNT)
    allowed = _rendered_palette(colors)
    index0 = tuple(_as_rendered(float(c)) for c in colors[0])
    actor = _point_actor(xyz, colors)
    any_ref = _reference(_point_actor(xyz[[0]], colors[[0]]))
    empty_rgb, empty_depth = _empty()
    failures = []
    lowest_index_wins = 0
    for f in range(repeats):
        rgb, depth = _planes(_render([actor]))
        covered = depth != empty_depth
        if covered.sum() != 29:
            failures.append(f"frame {f}: {covered.sum()} pixels covered, expected 29")
            continue
        if not np.array_equal(covered, any_ref[2]):
            failures.append(f"frame {f}: covered mask differs from the reference disc")
            continue
        bad = [pid for pid in np.flatnonzero(covered) if tuple(rgb[pid]) not in allowed]
        if bad:
            failures.append(f"frame {f}: {len(bad)} mixed-colour pixel(s), first {bad[0]} "
                            f"{tuple(rgb[bad[0]])}")
            continue
        if not np.all(depth[covered] == any_ref[1][covered]):
            failures.append(f"frame {f}: tie depth differs from the reference depth")
            continue
        if all(tuple(px) == index0 for px in rgb[covered]):
            lowest_index_wins += 1
    return repeats, failures, {"frames_where_index0_won_everywhere": lowest_index_wins}


def scenario_overlapping_wireframes(repeats):
    """Two overlapping horizontal lines, one tilted in depth: the winner
    changes along the line and is decided per pixel by interpolated depth."""
    flat = _line_actor((-1.5, 0.0, 0.0), (1.5, 0.0, 0.0), (0, 1, 0))
    tilt = _line_actor((-1.5, 0.0, 0.5), (1.5, 0.0, -0.5), (1, 0, 0))
    expected = Expected([_reference(flat), _reference(tilt)], _empty())
    covered = expected.covered.sum()
    assert covered > 10, covered
    failures = []
    for f in range(repeats):
        for order in ((flat, tilt), (tilt, flat)):
            rgb, depth = _planes(_render(list(order)))
            err = expected.check(rgb, depth, f"frame {f}, order {[a.color for a in order]}")
            if err:
                failures.append(err)
    return 2 * repeats, failures, {"covered_pixels": int(covered)}


def scenario_changed_inputs(repeats):
    """One actor, geometry and colours changed between frames; the frame
    must reflect the current inputs, never a stale winner."""
    actor = _point_actor([[0, 0, 0], [0, 0, -1]], [[0, 1, 0], [1, 0, 0]])
    _, Canvas, *_ = _rendering()
    canvas = Canvas(SIZE, SIZE)
    failures = []
    frames = 0
    for f in range(repeats):
        winner = f % 2
        shift = 0.5 if (f // 2) % 2 else 0.0          # move the disc every two frames
        pts = np.array([[shift, 0, -1], [shift, 0, -1]], np.float32)
        pts[winner, 2] = 0.0
        colors = np.array([[0, 1, 0], [1, 0, 0]], np.float32)
        if (f // 4) % 2:
            colors = colors[::-1].copy()                 # swap colours every four
        actor.points.from_numpy(pts.ravel())
        actor.point_colors.from_numpy(colors.ravel())
        ref = _reference(_point_actor(pts[[winner]], colors[[winner]]))
        expected = Expected([ref], _empty())
        _render([actor], canvas)
        rgb, depth = _planes(canvas)
        err = expected.check(rgb, depth, f"frame {f} (winner {winner}, shift {shift})")
        if err:
            failures.append(err)
        frames += 1
    # An empty frame afterwards must not keep the previous winner.
    actor.points.from_numpy(np.array([[9, 9, 0], [9, 9, -1]], np.float32).ravel())
    _render([actor], canvas)
    rgb, depth = _planes(canvas)
    err = Expected([], _empty()).check(rgb, depth, "empty frame")
    if err:
        failures.append(err)
    return frames + 1, failures


def scenario_actor_order(repeats):
    """Two actors at different depths in both scene orders: depth decides;
    at equal depth both colours are acceptable and the chosen one is reported."""
    results = {}
    failures = []
    frames = 0
    for mode in ("points", "wireframe"):
        def make(z, color):
            if mode == "points":
                return _point_actor([[0, 0, z]], [color])
            return _line_actor((-1.0, 0.0, z), (1.0, 0.0, z), color)
        for tie in (False, True):
            front = make(0.0, (0, 1, 0))
            back = make(0.0 if tie else -1.0, (1, 0, 0))
            expected = Expected([_reference(front), _reference(back)], _empty())
            for order_name, order in (("front_first", (front, back)), ("back_first", (back, front))):
                picks = set()
                for f in range(repeats):
                    rgb, depth = _planes(_render(list(order)))
                    err = expected.check(rgb, depth, f"{mode} tie={tie} {order_name} frame {f}")
                    if err:
                        failures.append(err)
                    picks.add(tuple(np.unique(rgb[expected.covered], axis=0).tolist()[0])
                              if len(np.unique(rgb[expected.covered], axis=0)) == 1 else "mixed")
                    frames += 1
                results[f"{mode}/tie={tie}/{order_name}"] = sorted(map(str, picks))
    return frames, failures, results


def scenario_compositing(repeats):
    """Composite over an existing image: uncovered colour and the path
    tracer's depth plane are preserved, drawn pixels are gamma-corrected,
    and a nearer surface hides the raster behind it."""
    _, Canvas, *_ = _rendering()
    xyz = np.zeros((COUNT, 3), np.float32)
    colors = np.tile([0.0, 1.0, 0.0], (COUNT, 1)).astype(np.float32)
    actor = _point_actor(xyz, colors)
    ref_rgb, _, covered = _reference(_point_actor(xyz[[0]], colors[[0]]))
    base = np.array([0.25, 0.5, 0.75], np.float32)
    drawn = np.array([0.0, 1.0, 0.0], np.float32) ** np.float32(0.4545)   # exact for 0/1
    failures = []
    frames = 0
    n = SIZE * SIZE
    left_half = (np.arange(n) % SIZE) < SIZE // 2
    for f in range(repeats):
        for hidden_left in (False, True):
            canvas = Canvas(SIZE, SIZE)
            canvas.color_r.fill(float(base[0])); canvas.color_g.fill(float(base[1]))
            canvas.color_b.fill(float(base[2]))
            surface = np.full(n, -1.0, np.float32)            # path-traced miss
            if hidden_left:
                surface[left_half] = 1.0                      # a surface 1 unit out, nearer than z=0
            canvas.depth.from_numpy(surface)
            _render([actor], canvas, composite=True, surface_depth=True)
            rgb, depth = _planes(canvas)
            label = f"frame {f} hidden_left={hidden_left}"
            visible = covered & ~(left_half if hidden_left else np.zeros(n, bool))
            if not np.all(rgb[visible] == drawn):
                failures.append(f"{label}: drawn pixels are not the gamma-corrected colour")
            elif not np.all(rgb[~visible] == base):
                failures.append(f"{label}: an uncovered or hidden pixel changed colour")
            elif not np.array_equal(depth, surface):
                failures.append(f"{label}: the canvas depth plane was modified")
            frames += 1
    return frames, failures


# ── driver ───────────────────────────────────────────────────────────────

def run_all(repeats: int) -> dict:
    cases = {}

    def record(name, result):
        frames, failures, *extra = result
        cases[name] = {"frames": frames, "failures": len(failures),
                       "first_failure": failures[0] if failures else None,
                       **(extra[0] if extra else {})}
        status = "PASS" if not failures else "FAIL"
        print(f"  {status:4s} {name:48s} {frames - len(failures):4d}/{frames}"
              + (f"  {failures[0]}" if failures else ""), flush=True)

    for where, at in (("first", 0), ("middle", COUNT // 2), ("last", COUNT - 1)):
        record(f"coincident_points/winner_{where}", scenario_coincident_points(repeats, at))
    record("equal_depth_ties/4097_points", scenario_equal_depth_ties(repeats))
    record("overlapping_wireframes/varying_depth", scenario_overlapping_wireframes(repeats))
    record("changed_inputs/repeated_frames", scenario_changed_inputs(repeats))
    record("actor_order/depth_and_ties", scenario_actor_order(repeats))
    record("compositing/uncovered_colour_and_depth", scenario_compositing(repeats))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arch", required=True, choices=_ARCHES)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--json", default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    backend = init_backend(args.arch)
    ident = source_identity()
    print(f"tack module:      {ident['tack']}")
    print(f"rasterize module: {ident['rasterize']}  sha256 {ident['rasterize_sha256']}")
    import llvmlite
    meta = {"label": args.label, "arch": backend.name, "repeats": args.repeats,
            "python": platform.python_version(), "numpy": np.__version__,
            "llvmlite": llvmlite.__version__, "platform": platform.platform(),
            "size": SIZE, "point_size": POINT_SIZE, "count": COUNT, **ident}
    t0 = time.perf_counter()
    cases = run_all(args.repeats)
    meta["seconds"] = round(time.perf_counter() - t0, 1)
    failed = sum(1 for c in cases.values() if c["failures"])
    print(f"{len(cases) - failed}/{len(cases)} cases clean on {backend.name} "
          f"({meta['seconds']} s)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "cases": cases}, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
