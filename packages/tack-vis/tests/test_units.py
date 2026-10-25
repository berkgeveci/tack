"""Physical units (phase U0 of the vault's units proposal): units on fields, the
spatial unit on the geometry, kept by every filter that keeps a quantity,
unknown (``None``) from the rest, mismatched spatial units refused, ``convert``
on the device, and VTK's ``UNITS_LABEL`` both ways."""

import numpy as np
import pytest
from test_dataset_devices import CALLS, _hexes, _SecondCPU

import tack
import tack.dataset as td
from tack.dataset import units
from tack.runtime import dispatch
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkCommonCore import vtkDataArray, vtkFloatArray, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtkUnstructuredGrid = None

needs_vtk = pytest.mark.skipif(vtkUnstructuredGrid is None, reason="needs VTK")

pytestmark = pytest.mark.quick


def _labelled(data, spatial="m"):
    """``data`` with a spatial unit, ``s`` in kelvin and ``v`` in metres per second."""
    fields = {"s": td.Field(data.fields["s"].space, data.fields["s"].values, "K"),
              "v": td.Field(data.fields["v"].space, data.fields["v"].values, "m/s")}
    return td.DataSet(data.topology, data.geometry, fields=fields, sets=dict(data.sets),
                      unit=spatial)


# -- the units themselves ----------------------------------------------------------------

def test_units_are_canonical_strings():
    assert units.canonical("m/s") == units.canonical("m*s**-1") == units.canonical("m/s^1")
    assert units.canonical("kg*m^2/s^2") == units.canonical("kg*m**2/s**2")
    assert units.canonical("1") == units.canonical("dimensionless") == "dimensionless"
    assert units.canonical(None) is None
    assert units.canonical("3.0857e21*cm") == "3085700000000000000000*cm"   # code units
    with pytest.raises(ValueError, match="empty unit"):
        units.canonical(" ")
    with pytest.raises(ValueError, match="is not a unit"):
        units.canonical("furlongs per fortnight")
    with pytest.raises(TypeError, match="a unit is a string"):
        units.canonical(3.0)


def test_conversions_and_sameness():
    factor, offset = units.conversion("ftUS", "m")
    assert factor == 1200 / 3937 and offset == 0.0
    assert units.conversion("ft", "m") == (0.3048, 0.0)
    factor, offset = units.conversion("degC", "K")
    assert (factor, offset) == (1.0, 273.15)
    factor, offset = units.conversion("degF", "degC")
    np.testing.assert_allclose(factor * 212.0 + offset, 100.0)
    with pytest.raises(ValueError, match="cannot convert 'm' to 's'"):
        units.conversion("m", "s")
    with pytest.raises(ValueError, match="unit is unknown"):
        units.conversion(None, "m")
    assert units.same("m", units.canonical("100*cm")) and not units.same("m", "ft")
    assert units.same(None, None) and not units.same(None, "m")
    assert units.compatible(None, "m") and not units.compatible("m", "ftUS")
    assert units.common("x", ["m", "m"]) == "m"
    assert units.common("x", ["m", None]) is None
    with pytest.raises(ValueError, match="x is in 'm' and 'ftUS'"):
        units.common("x", ["m", None, "ftUS"])


# -- fields and datasets -----------------------------------------------------------------

def test_fields_and_datasets_carry_units():
    data = td.uniform_grid((3, 3, 3), spacing=(0.5, 0.5, 0.5), unit="km")
    assert data.unit == "km" and data.geometry.unit == "km"
    values = tack.array(np.arange(27, dtype=np.float32))
    t = td.Field(td.H1(data), values, unit="degC")
    assert t.unit == "°C" and "in °C" in repr(t)
    assert td.Field(td.H1(data), values).unit is None
    assert td.rectilinear_grid([0.0, 1.0], [0.0, 1.0], unit="m").unit == "m"
    cloud = td.point_cloud(np.zeros((4, 3)), fields={"i": np.arange(4)}, unit="ftUS",
                           units={"i": "1"})
    assert cloud.unit == "ftUS" and cloud.fields["i"].unit == "dimensionless"
    with pytest.raises(KeyError, match="units for fields not given"):
        td.point_cloud(np.zeros((4, 3)), units={"i": "1"})
    # A geometry Field brings its unit; a second, different one is refused.
    geometry = td.Field(td.H1(data), data.geometry.values, "m")
    assert td.DataSet(data.topology, geometry).unit == "m"
    assert td.DataSet(data.topology, td.Field(td.H1(data), data.geometry.values),
                      unit="m").unit == "m"
    with pytest.raises(ValueError, match="the geometry is in 'm', not 'ft'"):
        td.DataSet(data.topology, geometry, unit="ft")
    # Positions given as values stay in the spatial unit; a Field brings its own.
    data.geometry = tack.array(data.positions(), tack.vec3())
    assert data.unit == "km"
    data.geometry = td.Field(td.H1(data), data.geometry.values)
    assert data.unit is None


#: What a filter that gives a field (rather than a dataset) gives it, from the
#: point field ``s`` in kelvin, ``v`` in metres per second and positions in
#: metres. Everything else gives ``None``: a derived unit is phase U1's, and
#: until then it is unknown, not guessed.
FIELD_UNITS = {
    "cell_centers": "m",
    "face_centers": "m",
    "values_at_centers": "K",
    "to_cells": "K",
    "to_points": "K",
    "discontinuous": "K",
    "project": "K",
    "traces": "K",
    "magnitude": "m/s",
}

#: Filters whose output's geometry is not in the input's spatial unit:
#: cylindrical and spherical coordinates mix an angle with lengths, and probe
#: gives back its target, a dataset of no unit here.
UNKNOWN_SPACE = {"cylindrical", "spherical", "probe"}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_every_algorithm_keeps_a_quantitys_unit_and_forgets_no_other(name):
    data = _labelled(_hexes(tack.device("cpu")))
    out = CALLS[name](data)
    if isinstance(out, td.DataSet):
        assert out.unit == (None if name in UNKNOWN_SPACE else "m")
        for field_name, field in out.fields.items():
            if field_name in data.fields:
                assert field.unit == data.fields[field_name].unit, field_name
    elif isinstance(out, td.Field):
        assert out.unit == FIELD_UNITS.get(name)
    else:
        assert name not in FIELD_UNITS


def test_contour_values_are_in_the_fields_unit():
    data = _labelled(_hexes(tack.device("cpu")))
    out = td.contour(data, "s", [2.0, 3.0])
    assert out.fields["contour_value"].unit == "K" and out.unit == "m"


def test_pieces_and_devices_keep_units():
    data = _labelled(_hexes(tack.device("cpu")))
    second = dispatch.as_device(_SecondCPU())
    moved = data.to_device(second)
    assert moved.unit == "m" and moved.fields["v"].unit == "m/s"
    grid = td.uniform_grid((5, 4, 3), unit="m")
    grid.fields["c"] = td.Field(td.Constant(grid), tack.ones((grid.num_cells,)), "kg")
    pieces = td.split(grid, (2, 1, 1), ghost_layers=1)
    assert all(p.unit == "m" and p.fields["c"].unit == "kg" for p in pieces)
    merged = td.merge_pieces(pieces)
    assert merged.unit == "m" and merged.fields["c"].unit == "kg"
    unstructured = td.PartitionedDataSet([td.tetrahedralize(p) for p in td.split(grid, (2, 1, 1))])
    ghosted = td.ghost_cells(unstructured)
    assert all(p.unit == "m" and p.fields["c"].unit == "kg" for p in ghosted)


def test_mismatched_spatial_units_are_refused():
    data = _labelled(_hexes(tack.device("cpu")))
    feet = td.point_cloud(np.array([[0.5, 0.5, 0.5]]), unit="ftUS")
    with pytest.raises(ValueError, match="'ftUS' and 'm' differ"):
        td.probe(data, feet)
    unknown = td.point_cloud(np.array([[0.5, 0.5, 0.5]]))
    assert td.probe(data, unknown).fields["s"].unit == "K"       # unknown is not checked
    assert td.probe(data, np.array([[0.5, 0.5, 0.5]])).unit == "m"
    grid = td.uniform_grid((5, 4, 3), unit="m")
    pieces = list(td.split(grid, (2, 1, 1)))
    pieces[1] = td.DataSet(pieces[1].topology, td.Field(pieces[1].geometry.space,
                                                        pieces[1].geometry.values, "ft"))
    with pytest.raises(ValueError, match="pieces' geometry is in '(m|ft)' and '(ft|m)'"):
        td.ghost_cells(td.PartitionedDataSet(pieces))
    import test_amr

    amr = test_amr.hierarchy()
    levels = [[td.DataSet(b.topology, b.geometry.values, fields=dict(b.fields), dtype=b.dtype,
                          unit="m" if level == 0 else "ft") for b in boxes]
              for level, boxes in enumerate(amr.levels)]
    with pytest.raises(ValueError, match="boxes' geometry is in '(m|ft)' and '(ft|m)'"):
        td.AMRDataSet(levels, amr.ratios)


# -- convert ---------------------------------------------------------------------------

def test_convert_the_geometry_keeps_survey_coordinates_and_images():
    corner = (1_400_000.0, 600_000.0, 50.0)            # State Plane, in US survey feet
    image = td.uniform_grid((4, 3, 2), origin=(1.0, 2.0, 0.0), spacing=(3.0, 3.0, 3.0),
                            unit="ftUS")
    image.origin = corner
    metres = td.convert(image, "m")
    assert isinstance(metres.geometry.values, tack.arrays.UniformCoordinates)
    assert metres.unit == "m"
    np.testing.assert_allclose(metres.world_positions(), image.world_positions() * 1200 / 3937,
                               rtol=0, atol=1e-6)
    np.testing.assert_array_equal(metres.origin, np.asarray(corner) * (1200 / 3937))
    lines = td.rectilinear_grid([0.0, 1.0, 3.0], [0.0, 2.0], [5.0, 6.0], unit="km")
    lines_m = td.convert(lines, "m")
    assert isinstance(lines_m.geometry.values, tack.arrays.CartesianProduct)
    np.testing.assert_array_equal(lines_m.world_positions(), lines.world_positions() * 1000)
    data = _labelled(_hexes(tack.device("cpu")))
    cm = td.convert(data, "cm")
    np.testing.assert_allclose(cm.positions(), data.positions() * 100)
    assert cm.unit == "cm" and cm.fields["s"].unit == "K"
    assert cm.sets["corner"].ids.numpy().tolist() == [0]
    with pytest.raises(ValueError, match="cannot convert 'm' to 's'"):
        td.convert(data, "s")
    with pytest.raises(ValueError, match="unit is unknown"):
        td.convert(_hexes(tack.device("cpu")), "m")


def test_convert_a_field_on_the_device():
    data = _labelled(_hexes(tack.device("cpu")))
    data.ghosts = td.Ghosts(cells=np.array([0] * 7 + [1], np.uint8))
    celsius = td.DataSet(data.topology, data.geometry,
                         fields={**data.fields, "t": td.Field(td.H1(data), tack.array(
                             np.linspace(0, 100, data.num_points).astype(np.float32)), "degC")},
                         unit="m")
    kelvin = td.convert(celsius, "K", field="t")
    assert kelvin.fields["t"].unit == "K" and kelvin.fields["s"] is celsius.fields["s"]
    np.testing.assert_allclose(kelvin.fields["t"].values.numpy(),
                               np.linspace(0, 100, data.num_points) + 273.15, rtol=1e-6)
    fast = td.convert(data, "km/hr", field="v")
    np.testing.assert_allclose(fast.fields["v"].values.numpy(),
                               data.fields["v"].values.numpy() * 3.6, rtol=1e-6)
    marked = td.convert(td.DataSet(data.topology, data.geometry, fields=dict(data.fields),
                                   ghosts=data.ghosts, unit="m"), "mK", field="s")
    assert marked.ghosts is data.ghosts
    counts = td.DataSet(data.topology, data.geometry, unit="m", fields={
        "n": td.Field(td.Constant(data), tack.array(np.arange(8, dtype=np.int32)), "1000*m")})
    relabelled = td.convert(counts, "km", field="n")
    assert relabelled.fields["n"].unit == "km"
    assert relabelled.fields["n"].values is counts.fields["n"].values
    with pytest.raises(TypeError, match="holds integers"):
        td.convert(counts, "m", field="n")
    with pytest.raises(KeyError, match="no field 'w'"):
        td.convert(data, "m", field="w")


# -- the octree ------------------------------------------------------------------------

def test_octree_units_reach_its_leaves_and_probes():
    import test_octrees

    roots, level, centre, rho, temperature = test_octrees.ramses_like(3)
    roots = td.uniform_grid(tuple(n for n in roots.topology.point_dims), unit="kpc")
    octree = td.OctreeDataSet.from_cells(roots, level, centre,
                                         fields={"rho": rho, "T": temperature},
                                         units={"rho": "Msun/kpc**3", "T": "K"})
    assert octree.unit == "kpc" and octree.units == {"rho": "Msun/kpc**3", "T": "K"}
    leaves = octree.leaves()
    assert leaves.unit == "kpc" and leaves.fields["T"].unit == "K"
    probed = td.probe(octree, np.array([[0.5, 0.5, 0.5]]))
    assert probed.unit == "kpc" and probed.fields["rho"].unit == "Msun/kpc**3"
    with pytest.raises(ValueError, match="differ"):
        td.probe(octree, td.point_cloud(np.array([[0.5, 0.5, 0.5]]), unit="m"))
    with pytest.raises(KeyError, match="units for fields not given"):
        td.OctreeDataSet.from_cells(roots, level, centre, units={"T": "K"})


# -- VTK ---------------------------------------------------------------------------------

@needs_vtk
def test_vtk_units_label_both_ways():
    data = _labelled(_hexes(tack.device("cpu")))
    data.fields["c"] = td.Field(td.Constant(data), tack.ones((data.num_cells,)), "kg")
    grid = dataset_to_vtk(data)
    key = vtkDataArray.UNITS_LABEL()
    assert grid.points.data.GetInformation().Get(key) == "m"
    assert grid.point_data["s"].GetInformation().Get(key) == "K"
    assert grid.cell_data["c"].GetInformation().Get(key) == "kg"
    back = vtk_to_dataset(grid)
    assert back.unit == "m"
    assert {n: f.unit for n, f in back.fields.items()} == {"s": "K", "v": "m/s", "c": "kg"}
    lines = vtk_to_dataset(dataset_to_vtk(td.rectilinear_grid([0.0, 1.0], [0.0, 2.0],
                                                              unit="ftUS")))
    assert lines.unit == "ftUS"
    # VTK's labels are free text, and LLNL's powers are '^'.
    raw = vtkUnstructuredGrid()
    points = vtkFloatArray(np.zeros((1, 3), np.float32))
    points.GetInformation().Set(key, "kg*m^2/s^2")
    raw.points = vtkPoints(data=points)
    named = vtkFloatArray(np.zeros(1, np.float32))
    named.name = "e"
    named.GetInformation().Set(key, "joules, roughly")
    raw.GetPointData().AddArray(named)     # point_data[...] = drops its information
    with pytest.warns(UserWarning, match="no unit Tack reads"):
        read = vtk_to_dataset(raw)
    assert read.unit == "kg*m**2/s**2" and read.fields["e"].unit is None


@needs_vtk
def test_hypertreegrid_units_both_ways():
    import test_octrees

    roots, level, centre, rho, temperature = test_octrees.ramses_like(2)
    roots = td.uniform_grid(tuple(n for n in roots.topology.point_dims), unit="kpc")
    octree = td.OctreeDataSet.from_cells(roots, level, centre, fields={"T": temperature},
                                         units={"T": "K"})
    back = vtk_to_dataset(dataset_to_vtk(octree))
    assert back.unit == "kpc" and back.units == {"T": "K"}
