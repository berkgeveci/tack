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


@pytest.fixture(autouse=True)
def cpu():
    """Units are the host's: datasets here are on the CPU, whatever the default
    device of the run (a dataset and a point cloud on two devices would be
    refused before their units are compared)."""
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


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
#: metres. Labels, ids and flags give ``None``, as does a result computed from
#: a bare number (upwind_flux's velocity, perot's faces of ones here).
FIELD_UNITS = {
    "cell_centers": "m",
    "face_centers": "m",
    "values_at_centers": "K",
    "gradients": "K/m",
    "gradients_at_points": "K/m",
    "to_cells": "K",
    "to_points": "K",
    "discontinuous": "K",
    "project": "K",
    "traces": "K",
    "jump": "K",
    "edge_lengths": "m",
    "magnitude": "m/s",
    "dot": "m**2/s**2",
    "cross": "m**2/s**2",
    "composite": "K",
    "cell_geometry": "m**3",
    "cell_quality": "dimensionless",
    "face_geometry": "m**2",
    "flow_quantities": "1/s",
    "net_flux": "dimensionless",            # of the faces' unit normals
    "deposit": "dimensionless",             # a count
    "deposit_cic": "K",                     # a mean of s
}

#: Fields a filter that gives a dataset computes, and their units; every other
#: field it has from the input keeps the input's unit. SPH interpolation here
#: takes its masses and densities as numbers, so its sums' units are unknown.
DATASET_UNITS = {
    "sph_interpolate": {"s": None, "v": None, "shepard": None},
    "sph_density": {"density": None},
    "outliers_statistical": {"mean_distance": "m"},
    "point_normals": {"normals": "dimensionless"},
    "point_normals_radius": {"normals": "dimensionless"},
    "point_normals_periodic": {"normals": "dimensionless"},
    "advect": {"time": "s"},
    "streamlines": {"time": "s"},
    "halo_catalog": {"com": "m", "radius": "m", "mass": None, "velocity": "m/s",
                     "dispersion": "m/s"},
    "halo_catalog_periodic": {"com": "m", "radius": "m", "mass": None},
}

#: Filters a unit refuses: the logarithm of kelvin.
REFUSED = {"log_values": "takes a pure number"}

#: Filters whose output's geometry is not in the input's spatial unit:
#: cylindrical and spherical coordinates mix an angle with lengths, and probe
#: gives back its target, a dataset of no unit here.
UNKNOWN_SPACE = {"cylindrical", "spherical", "probe"}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_every_algorithm_keeps_or_derives_units(name):
    data = _labelled(_hexes(tack.device("cpu")))
    if name in REFUSED:
        with pytest.raises(ValueError, match=REFUSED[name]):
            CALLS[name](data)
        return
    out = CALLS[name](data)
    if isinstance(out, td.DataSet):
        assert out.unit == (None if name in UNKNOWN_SPACE else "m")
        computed = DATASET_UNITS.get(name, {})
        for field_name, field in out.fields.items():
            if field_name in computed:
                assert field.unit == computed[field_name], field_name
            elif field_name in data.fields:
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


# -- derived units (U1) ------------------------------------------------------------------

def test_derived_units_deltas_and_names():
    assert units.derived(("K", 1), ("m", -1)) == "K/m"
    assert units.derived("kg/m**3", "m/s", ("m", 2)) == "kg/s"
    assert units.derived("dimensionless", ("m", -3)) == units.canonical("1/m**3")
    assert units.derived(None, "m") is None
    with pytest.raises(ValueError, match="offset zero"):
        units.derived("degC", "m")
    # A difference of degrees Celsius is in kelvin, of Fahrenheit in Rankine.
    assert units.delta("degC") == "K" and units.delta("degF") == "R"
    assert units.delta("m") == "m" and units.delta(None) is None
    # Only the dimensions and the size decide a name.
    assert units.named(units.derived("Pa", ("m", 3))) == "J"
    assert units.named(units.derived("m", "N")) == "J"
    assert units.named("kg*m/s**2") == "N"
    assert units.named("dyn*cm") == "dyn*cm"           # 1e-7 J: not renamed
    assert units.named("1/s") == "1/s"                 # Hz and Bq: left out
    assert units.dimensionless_or_none("x", "1") == "dimensionless"
    assert units.dimensionless_or_none("x", None) is None
    with pytest.raises(ValueError, match="x takes a pure number"):
        units.dimensionless_or_none("x", "km/m")


def _celsius(data):
    t = td.Field(data.fields["s"].space, data.fields["s"].values, "degC")
    return td.DataSet(data.topology, data.geometry, fields={**data.fields, "t": t},
                      unit=data.unit)


def test_differences_of_celsius_are_in_kelvin():
    data = _celsius(_labelled(_hexes(tack.device("cpu"))))
    assert td.gradients(data, "t").unit == "K/m"
    assert td.jump(data, td.project(data, "t", td.L2(data))).unit == "K"
    stats = td.statistics(data, "t")
    assert stats.unit == "°C" and stats.unit_of("mean") == "°C"
    assert stats.unit_of("population_variance") == "K**2"
    assert stats.unit_of("sample_stddev") == "K"
    with pytest.raises(ValueError, match="offset zero"):
        td.dot(data, "v", td.composite(data, "t", "t", "t"))


def test_statistics_give_each_number_its_unit():
    data = _labelled(_hexes(tack.device("cpu")))
    plain = td.statistics(data, "s")
    assert (plain.unit, plain.weight_unit) == ("K", "dimensionless")
    assert plain.unit_of("sum") == "K" and plain.unit_of("m2") == "K**2"
    assert plain.unit_of("skewness") == plain.unit_of("n") == "dimensionless"
    cells = td.project(data, "s", td.Constant(data))
    data.fields["c"] = cells
    data.fields["rho"] = td.Field(td.Constant(data), tack.ones((data.num_cells,)), "kg/m**3")
    by_volume = td.statistics(data, "c", weights="volume")
    assert by_volume.weight_unit == "m**3" and by_volume.unit_of("weight") == "m**3"
    assert by_volume.unit_of("sum") == "K*m**3" and by_volume.unit_of("mean") == "K"
    by_density = td.statistics(data, "c", weights="rho")
    assert by_density.unit_of("sum") == units.canonical("K*kg/m**3")
    with pytest.raises(KeyError, match="no statistic"):
        plain.unit_of("median")
    import test_octrees

    roots, level, centre, rho, temperature = test_octrees.ramses_like(3)
    roots = td.uniform_grid(tuple(n for n in roots.topology.point_dims), unit="kpc")
    octree = td.OctreeDataSet.from_cells(roots, level, centre, fields={"T": temperature},
                                         units={"T": "K"})
    stats = td.statistics(octree, "T", weights="volume")
    assert stats.unit == "K" and stats.weight_unit == "kpc**3"


def test_products_composites_and_logarithms():
    data = _labelled(_hexes(tack.device("cpu")))
    data.fields["F"] = td.Field(data.fields["v"].space, data.fields["v"].values, "N")
    data.fields["d"] = td.Field(data.fields["v"].space, data.fields["v"].values, "m")
    assert td.dot(data, "F", "d").unit == "J"            # N m folds to its name
    assert td.cross(data, "d", "F").unit == "J"
    data.fields["r"] = td.Field(data.fields["s"].space, data.fields["s"].values, "1")
    assert td.log_values(data, "r").unit == "dimensionless"
    with pytest.raises(ValueError, match="takes a pure number"):
        td.log_values(data, "s")
    with pytest.raises(ValueError, match="composite's fields is in"):
        td.composite(data, "s", "r", "s")


def test_geometry_and_fluxes():
    data = _labelled(_hexes(tack.device("cpu")))
    assert td.cell_geometry(data).centroid.unit == "m"
    normal, area = td.face_geometry(data)
    assert normal.unit == "dimensionless" and area.unit == "m**2"
    flux = td.Field(td.Values(data, "faces", oriented=True), tack.ones(
        (data.topology.faces().num_faces,)), "kg/s")
    assert td.net_flux(data, flux).unit == "kg/s"
    assert td.divergence(data, flux).unit == units.canonical("kg/(m**3*s)")
    velocity = td.Field(td.Values(data, "faces"), tack.ones(
        (data.topology.faces().num_faces,)), "m/s")
    assert td.perot(data, velocity).unit == "m/s"
    surface = td.uniform_grid((3, 3), unit="km")
    assert td.cell_geometry(surface).measure.unit == "km**2"
    assert td.face_geometry(surface).area.unit == "km"   # a 2D mesh's faces are edges
    gradient = td.gradients(data, "v")
    quantities = td.flow_quantities(data, gradient)
    assert gradient.unit == "1/s" and quantities.q_criterion.unit == "s**(-2)"


def test_deposits_and_particles():
    rng = np.random.default_rng(0)
    cloud = td.point_cloud(rng.uniform(0.0, 2.0, (200, 3)),
                           fields={"m": np.full(200, 2.0), "rho": np.full(200, 4.0),
                                   "h": np.full(200, 0.5), "T": rng.random(200),
                                   "v": rng.random((200, 3))},
                           unit="m", units={"m": "kg", "rho": "kg/m**3", "h": "m",
                                            "T": "K", "v": "m/s"}, origin=(0, 0, 0))
    image = td.uniform_grid((5, 5, 5), spacing=(0.5, 0.5, 0.5), unit="m")
    assert td.deposit(cloud, image).unit == "dimensionless"
    assert td.deposit(cloud, image, per_volume=True).unit == units.canonical("1/m**3")
    assert td.deposit(cloud, image, "m", per_volume=True).unit == "kg/m**3"
    assert td.deposit(cloud, image, "T", "mean").unit == "K"
    feet = td.uniform_grid((5, 5, 5), spacing=(0.5, 0.5, 0.5), unit="ft")
    with pytest.raises(ValueError, match="spatial unit is in"):
        td.deposit(cloud, feet)
    density = td.sph_density(cloud, smoothing_length="h", mass="m")
    assert density.fields["density"].unit == "kg/m**3"
    sampled = td.sph_interpolate(cloud, image, ["T"], smoothing_length="h", mass="m",
                                 density="rho")
    assert sampled.fields["T"].unit == "K" and sampled.fields["shepard"].unit == "dimensionless"
    normalized = td.sph_interpolate(cloud, image, ["T"], smoothing_length="h", mass=1.0,
                                    density=1.0, normalize=True)
    assert normalized.fields["T"].unit == "K"
    labelled = td.connected_components(cloud, radius=0.4)
    catalog = td.halo_catalog(labelled, mass="m", velocity="v")
    assert catalog.fields["mass"].unit == "kg" and catalog.fields["radius"].unit == "m"
    grid = _labelled(_hexes(tack.device("cpu")))
    moved = td.advect(grid, "v", np.array([[0.5, 0.5, 0.5]], np.float32), 0.01, 3)
    assert moved.fields["time"].unit == "s"


@tack.func
def _height(x):
    return x[2]


def test_interpolate_takes_the_unit_it_cannot_infer():
    data = _labelled(_hexes(tack.device("cpu")))
    assert td.interpolate(data, _height, td.H1(data), unit="m").unit == "m"
    assert td.interpolate(data, _height, td.H1(data)).unit is None


# -- quantities as parameters (U2) -------------------------------------------------------

def test_quantities_parse_and_convert():
    assert units.quantity("10 cm") == (10.0, "cm")
    assert units.quantity((2, "ft")) == (2, "ft")
    assert units.quantity("h") is None and units.quantity(3.0) is None
    assert units.quantity("4") == (4.0, "dimensionless")
    from unyt import unyt_quantity

    assert units.quantity(unyt_quantity(5.0, "kpc")) == (5.0, "kpc")
    assert units.magnitude("10 cm", "m", "x") == pytest.approx(0.1)
    assert units.magnitude("20 degC", "K", "x") == pytest.approx(293.15)
    assert units.magnitude(0.3, "m", "x") == 0.3                      # as given
    np.testing.assert_allclose(units.magnitude(([1, 2, 3], "km"), "m", "x"), [1e3, 2e3, 3e3])
    np.testing.assert_allclose(units.magnitude(["300 K", "310 K"], "degC", "x"),
                               [26.85, 36.85])
    with pytest.raises(ValueError, match="radius is given in 'm', but"):
        units.magnitude("1 m", None, "radius")


def _cloud(unit="m", n=300, seed=0):
    rng = np.random.default_rng(seed)
    return td.point_cloud(rng.uniform(0.0, 4.0, (n, 3)),
                          fields={"m": np.full(n, 2.0), "rho": np.full(n, 4.0),
                                  "h": np.full(n, 0.8), "T": 280 + 30 * rng.random(n),
                                  "v": rng.random((n, 3))},
                          unit=unit, units={"m": "kg", "rho": "kg/m**3", "h": "m", "T": "K",
                                            "v": "m/s"}, origin=(0.0, 0.0, 0.0))


def test_lengths_take_quantities():
    cloud = _cloud()
    same = lambda a, b: np.testing.assert_array_equal(a.positions(), b.positions())
    same(td.voxel_downsample(cloud, "50 cm"), td.voxel_downsample(cloud, 0.5))
    a = td.outliers(cloud, radius="600 mm", min_neighbors=3)
    b = td.outliers(cloud, radius=0.6, min_neighbors=3)
    np.testing.assert_array_equal(a.fields["outlier"].values.numpy(),
                                  b.fields["outlier"].values.numpy())
    a = td.connected_components(cloud, radius=(0.5, "m"), periodic="4 m")
    b = td.connected_components(cloud, radius=0.5, periodic=4.0)
    np.testing.assert_array_equal(a.fields["region"].values.numpy(),
                                  b.fields["region"].values.numpy())
    a = td.point_normals(cloud, radius="1 m")
    np.testing.assert_array_equal(a.fields["normals"].values.numpy(),
                                  td.point_normals(cloud, radius=1.0).fields["normals"]
                                  .values.numpy())
    locator = td.PointLocator(cloud, bin_size="1 m")
    offsets, ids = locator.neighbors("50 cm")
    expected = td.PointLocator(cloud, bin_size=1.0).neighbors(0.5)
    np.testing.assert_array_equal(ids.numpy(), expected[1].numpy())
    pieces = td.split(td.uniform_grid((6, 5, 4), unit="m"), (2, 1, 1))
    clouds = td.PartitionedDataSet([td.point_cloud(p.world_positions(), unit="m",
                                                   origin=(0, 0, 0)) for p in pieces])
    ghosted = td.ghost_points(clouds, radius="1.5 m")
    assert [p.num_points for p in ghosted] == [p.num_points for p in
                                               td.ghost_points(clouds, radius=1.5)]
    # A length needs a spatial unit to convert to.
    with pytest.raises(ValueError, match="no unit"):
        td.voxel_downsample(td.point_cloud(np.zeros((4, 3))), "1 m")


def test_field_values_take_quantities():
    data = _labelled(_hexes(tack.device("cpu")))           # s in kelvin
    a = td.threshold(data, "s", "-273.15 degC", "-270.15 degC")
    b = td.threshold(data, "s", 0.0, 3.0)
    assert a.num_cells == b.num_cells
    assert td.threshold_points(data, "s", 0.0, "3000 mK").num_points == \
        td.threshold_points(data, "s", 0.0, 3.0).num_points
    np.testing.assert_array_equal(td.contour(data, "s", ["2 K", "3 K"]).positions(),
                                  td.contour(data, "s", [2.0, 3.0]).positions())
    assert td.clip(data, "s", "3000 mK").num_cells == td.clip(data, "s", 3.0).num_cells
    with pytest.raises(ValueError, match="no unit"):
        td.clip(data, td.Sphere((0, 0, 0), 1.0), "1 m")   # a function's values have none
    np.testing.assert_allclose(
        td.slice_plane(data, ((50.0, 0.0, 0.0), "cm"), (1, 0, 0)).positions(),
        td.slice_plane(data, (0.5, 0.0, 0.0), (1, 0, 0)).positions())


def test_parameters_that_bring_their_own_units():
    cloud = _cloud(unit="cm")
    density = td.sph_density(cloud, smoothing_length="80 cm", mass=(2.0, "kg"))
    assert density.fields["density"].unit == "kg/cm**3"
    np.testing.assert_array_equal(density.fields["density"].values.numpy(),
                                  td.sph_density(cloud, smoothing_length=80.0, mass=2.0)
                                  .fields["density"].values.numpy())
    labelled = td.connected_components(cloud, radius="40 cm")
    catalog = td.halo_catalog(labelled, mass="1e10 Msun", softening="1 cm")
    assert catalog.fields["mass"].unit == "Msun"
    data = _labelled(_hexes(tack.device("cpu")))
    flux = td.upwind_flux(data, "s", ((1.0, 0.5, 0.0), "m/s"))
    assert flux.unit == units.canonical("K*m**3/s")
    assert td.upwind_flux(data, "s", (1.0, 0.5, 0.0)).unit is None
    seeds = np.array([[0.5, 0.5, 0.5]], np.float32)
    a = td.advect(data, "v", seeds, "10 ms", 3)
    b = td.advect(data, "v", seeds, 0.01, 3)
    np.testing.assert_array_equal(a.positions(), b.positions())
    assert a.fields["time"].unit == "s"
