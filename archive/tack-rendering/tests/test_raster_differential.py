"""The public-API raster differential probe, run as a test (CX8 / TA-23).

`benchmarks/raster_differential.py` checks raster frames against an oracle
composed from reference renders of each candidate alone, plus an
index-coded palette whose third channel hashes the other two so a pixel
with channels from two different primitives is usually rejected (some
mixtures collide with valid colours). Here it runs on every available
backend, and its oracle is itself exercised with planted defects so a
passing frame means the check could have failed.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

_PROBE = pathlib.Path(__file__).resolve().parents[3] / "benchmarks" / "raster_differential.py"
_spec = importlib.util.spec_from_file_location("raster_differential_probe", _PROBE)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

REPEATS = 3

SCENARIOS = {
    "coincident_points/winner_first": lambda: probe.scenario_coincident_points(REPEATS, 0),
    "coincident_points/winner_middle": lambda: probe.scenario_coincident_points(REPEATS, probe.COUNT // 2),
    "coincident_points/winner_last": lambda: probe.scenario_coincident_points(REPEATS, probe.COUNT - 1),
    "equal_depth_ties": lambda: probe.scenario_equal_depth_ties(REPEATS),
    "overlapping_wireframes": lambda: probe.scenario_overlapping_wireframes(REPEATS),
    "changed_inputs": lambda: probe.scenario_changed_inputs(REPEATS),
    "actor_order": lambda: probe.scenario_actor_order(REPEATS),
    "compositing": lambda: probe.scenario_compositing(REPEATS),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_probe_scenario_matches_its_independent_oracle(backend, scenario):
    frames, failures, *_ = SCENARIOS[scenario]()
    assert frames > 0
    assert not failures, failures[0]


# ── the oracle itself must be able to fail ───────────────────────────────

def test_oracle_rejects_a_planted_wrong_colour_depth_and_stray_pixel(backend):
    actor = probe._point_actor([[0, 0, 0]], [[0, 1, 0]])
    expected = probe.Expected([probe._reference(actor)], probe._empty())
    rgb, depth = probe._planes(probe._render([actor]))
    assert expected.check(rgb, depth, "clean") is None
    assert expected.covered.sum() == 29

    covered = np.flatnonzero(expected.covered)
    bad = rgb.copy()
    bad[covered[0]] = [1, 0, 0]                       # a loser's colour on a covered pixel
    assert "not among" in expected.check(bad, depth, "colour")

    bad_depth = depth.copy()
    bad_depth[covered[1]] = np.float32(bad_depth[covered[1]]) + np.float32(1.0)
    assert "depth" in expected.check(rgb, bad_depth, "depth")

    stray = rgb.copy()
    uncovered = np.flatnonzero(~expected.covered)[0]
    stray[uncovered] = [0, 0, 1]                       # paint outside the disc
    assert "uncovered" in expected.check(stray, depth, "stray")


def test_oracle_prefers_the_nearer_reference_and_accepts_exact_ties(backend):
    front = probe._point_actor([[0, 0, 0]], [[0, 1, 0]])
    back = probe._point_actor([[0, 0, -1]], [[1, 0, 0]])
    tie = probe._point_actor([[0, 0, 0]], [[0, 0, 1]])
    front_ref, back_ref, tie_ref = map(probe._reference, (front, back, tie))
    nearer = probe.Expected([front_ref, back_ref], probe._empty())
    pid = np.flatnonzero(nearer.covered)[0]
    assert nearer.candidates[pid] == {tuple(front_ref[0][pid])}
    assert nearer.depth[pid] == front_ref[1][pid] < back_ref[1][pid]
    tied = probe.Expected([front_ref, tie_ref], probe._empty())
    assert tied.candidates[pid] == {tuple(front_ref[0][pid]), tuple(tie_ref[0][pid])}


def test_palette_exposes_mixed_channels():
    colors = probe._index_colors(probe.COUNT)
    assert len({tuple(c) for c in colors}) == probe.COUNT, "colours must be distinct"
    assert not np.any(np.all(colors == 0, axis=1)), "no colour may equal the background"
    palette = {tuple(c) for c in colors}
    rng = np.random.default_rng(7)
    i = rng.integers(0, probe.COUNT, 2000)
    j = rng.integers(0, probe.COUNT, 2000)
    keep = (i % 64) != (j % 64)                        # red actually differs
    mixed = np.column_stack([colors[i, 0], colors[j, 1], colors[i, 2]])[keep]
    valid = sum(tuple(m) in palette for m in mixed)
    # Blue hashes red and green, so a red/green mix matches a real colour
    # only by coincidence: at most a few per cent of the time.
    assert valid < 0.05 * len(mixed), f"{valid} of {len(mixed)} mixed pixels looked valid"


def test_rendered_channel_mapping_is_measured_not_assumed(backend):
    probe._gamma_cache.clear()
    assert probe._as_rendered(0.0) == 0.0
    assert probe._as_rendered(1.0) == 1.0
    mid = probe._as_rendered(0.25)
    assert 0.25 <= mid <= 1.0                          # colour correction only brightens
    assert probe._as_rendered(0.5) > mid               # and is monotone


def test_channel_mapping_follows_backend_switches_and_reinitialization(monkeypatch):
    """Two instances of the same architecture may measure different values.

    Use distinct calibration results to expose stale reuse on any host,
    including hosts where the real backends happen to round identically.
    """
    from tack.runtime import dispatch

    class CalibrationBackend:
        name = "calibration"

        def __init__(self, measured):
            self.measured = measured

    first = CalibrationBackend(0.5)
    second = CalibrationBackend(0.75)
    active = [first]
    measurements = []

    def calibrate(actors):
        assert actors[0][0][0] == 0.25
        measured = active[0].measured
        measurements.append(measured)
        return np.full((probe.SIZE ** 2, 3), measured, np.float32), np.zeros(
            probe.SIZE ** 2, np.float32
        )

    monkeypatch.setattr(dispatch, "get_backend", lambda: active[0])
    monkeypatch.setattr(probe, "_point_actor", lambda xyz, colors: colors)
    monkeypatch.setattr(probe, "_render", calibrate)
    monkeypatch.setattr(probe, "_planes", lambda canvas: canvas)
    monkeypatch.setattr(probe, "_empty", lambda: (
        np.zeros((probe.SIZE ** 2, 3), np.float32),
        np.full(probe.SIZE ** 2, -1.0, np.float32),
    ))
    probe._gamma_cache.clear()
    assert probe._as_rendered(0.25) == 0.5
    assert probe._as_rendered(0.25) == 0.5
    active[0] = second
    assert probe._as_rendered(0.25) == 0.75
    active[0] = first
    assert probe._as_rendered(0.25) == 0.5
    assert measurements == [0.5, 0.75]


def test_tie_oracle_rejects_colour_outside_the_disc(backend, monkeypatch):
    render = probe._render

    def render_with_stray_pixel(actors, *args, **kwargs):
        canvas = render(actors, *args, **kwargs)
        # Corrupt only the contended frame, leaving its depth unchanged and
        # the single-point references/calibration measurements untouched.
        if len(actors) == 1 and actors[0].points.shape == (probe.COUNT * 3,):
            red = canvas.color_r.to_numpy().copy()
            red[0] = 1.0                           # outside the central disc
            canvas.color_r.from_numpy(red)
        return canvas

    monkeypatch.setattr(probe, "_render", render_with_stray_pixel)
    frames, failures, *_ = probe.scenario_equal_depth_ties(1)
    assert frames == 1
    assert len(failures) == 1
    assert "uncovered pixel 0 changed colour" in failures[0]
