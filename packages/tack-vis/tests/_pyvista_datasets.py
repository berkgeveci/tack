"""PyVista's example datasets (``pyvista.examples.downloads``), read with VTK alone.

The files are those of the ``pyvista/vtk-data`` repository, in its layout (a
zip unpacked beside itself as ``<zip>.unzip/``, as PyVista caches them), under
``TACK_PYVISTA_DATA`` (default ``~/.cache/tack/pyvista``); a PyVista cache
(``PYVISTA_USERDATA_PATH``) is such a directory. ``python _pyvista_datasets.py``
fetches what is missing and checks it against ``SHA256``. Composite data
(Exodus, OpenFOAM, EnSight) is merged into one unstructured grid."""

import hashlib
import os
import pathlib
import sys
import urllib.request
import zipfile

DATA = pathlib.Path(os.environ.get("TACK_PYVISTA_DATA",
                                   pathlib.Path.home() / ".cache" / "tack" / "pyvista"))
SOURCE = "https://github.com/pyvista/vtk-data/raw/master/Data/{path}"

_FEA = "fea/hertzian_contact_cylinder/Hertzian_cylinder_on_plate.zip"
_TURBO = "fvm/turbo_incompressible/Turbo-Incompressible_3-Run_1-SOLUTION_FIELDS.zip"
_CYLINDER = "EnSight/CylinderCrossflow/cylinder_Re35"
_VESSELS = "pvtu_blood_vessels/blood_vessels.zip"

#: By PyVista's name, ``(file, sources)``: the file a reader opens, and the
#: files of ``pyvista/vtk-data`` it comes from (the file itself if none).
DATASETS = {
    # Linear cells.
    "tetrahedron": ("Tetrahedron.vtu", ()),
    "can_crushed_vtu": ("can.vtu", ()),
    "angular_sector": ("AngularSector.vtk", ()),
    "notch_stress": ("notch_stress_fixed.vtk", ()),
    "unstructured_grid": ("uGridEx.vtk", ()),
    "exodus": ("mesh_fs8.exo", ()),
    "cavity": ("OpenFOAM.zip.unzip/OpenFOAM/cavity/case.foam", ("OpenFOAM.zip",)),
    "backward_facing_step": ("EnSight.zip.unzip/EnSight/foam_case_0_0_0_0.case",
                             ("EnSight.zip",)),
    "cylinder_crossflow": (f"{_CYLINDER}.case",
                           tuple(f"{_CYLINDER}.{e}" for e in ("case", "geo", "scl1", "scl2",
                                                              "vel"))),
    "prostar": ("prostar.vrt", ("prostar.vrt", "prostar.cel")),
    "blood_vessels": (f"{_VESSELS}.unzip/blood_vessels/T0000000500.pvtu", (_VESSELS,)),
    "reservoir": ("reservoir/UNISIM-II-D.zip.unzip/UNISIM-II-D.vtu",
                  ("reservoir/UNISIM-II-D.zip",)),
    "fea_hertzian_contact_cylinder": (
        f"{_FEA}.unzip/bfac9fd1-e982-4825-9a95-9e5d8c5b4d3e_result_1.pvtu", (_FEA,)),
    "openfoam_tubes": (f"{_TURBO}.unzip/case.foam", (_TURBO,)),
    # Quadratic cells.
    "aero_bracket": ("fea/aero_bracket/aero_bracket.vtu", ()),
    "fea_bracket": ("fea/kiefer/dataset.vtu", ()),
    "pump_bracket": ("fea/pump_bracket/pump_bracket.zip.unzip/pump_bracket.vtk",
                     ("fea/pump_bracket/pump_bracket.zip",)),
    "notch_displacement": ("notch_disp.vtu", ()),
    "disc_quads": ("Disc_BiQuadraticQuads_0_0.vtu", ()),
    "quadratic_pyramid": ("QuadraticPyramid.vtu", ()),
    "tri_quadratic_hexahedron": ("TriQuadraticHexahedron.vtu", ()),
    # Structured grids and images.
    "structured_grid": ("StructuredGrid.vts", ()),
    "structured_grid_two": ("SampleStructGrid.vtk", ()),
    "kitchen": ("kitchen.vtk", ()),
    "office": ("office.binary.vtk", ()),
    "rectilinear_grid": ("RectilinearGrid.vtr", ()),
    "carotid": ("carotid.vtk", ()),
    "coil_magnetic_field": ("magpylib/coil_field.vti", ()),
    # Point clouds.
    "particles_lethe": ("lethe/result_particles.20000.0000.vtu", ()),
    "stars_cloud_hyg": ("point-clouds/hyg-database/stars.vtp", ()),
}

SHA256 = {
    "Tetrahedron.vtu": "2c7a3e2dd6d161e4dba87fb390851edea5308e8c0ea35fd3fb0de5f02df76629",
    "can.vtu": "14fc2b44ea5d35f505a96306ff470b0b699e795a13cf841a21c38cb0e2eba78d",
    "AngularSector.vtk": "a204a535171643230228f7c35e2be0f5a3c496d8493c9e3cbde7cabd9f06c7fe",
    "notch_stress_fixed.vtk":
        "56402f77fa5fcb946fd4ff1868d68ffd7556bc85abc22b2836e734d3cd4bd1f4",
    "uGridEx.vtk": "b77a0d17a463687f2ed876e4fc76123054f013ee9b16859191e4fc3af1b31a66",
    "mesh_fs8.exo": "13f9b1966900e625849eb20072abff887f98eccd5391454532a1e008ed04ed15",
    "OpenFOAM.zip": "63318a35a3ef11a908a5997fc7e7564cf92558c6c98adbfe3d710f87b2d950dd",
    "EnSight.zip": "b060030f7715ab5a57f6f196bc08eef08240feea24fb92fff7c8d36e62cbd88a",
    f"{_CYLINDER}.case": "683956c15c786730bf07596a2a8480eb1de091ba81c57fde6d656f63f1520a32",
    f"{_CYLINDER}.geo": "5c58108fdb00d5ff306f9b594c423f1634e700197ac0cde9312deab7d8a58dda",
    f"{_CYLINDER}.scl1": "c489f3af25ce0df23e24933da47aecd3bb64a964f364856e1efe18a54f03f2da",
    f"{_CYLINDER}.scl2": "c5713b63c643c0f9de20c6a1135de9f383740f8eb33fd430db552b85a9aaa2f5",
    f"{_CYLINDER}.vel": "aa21f382ea6b189f2b18bf871d6f1352c64b772ad18584b87828ba0ab81cf9b2",
    "prostar.vrt": "23ab0e9c2a014f88e6157b99c98b067898fd98293dccca48f5d84500631d24a8",
    "prostar.cel": "ff212a54ceab75e800322662b204978a4f94d374acc54691d44c449f53a1a88d",
    _VESSELS: "832edbd67a5ab104f99580c27c8f17ed55795dfc836a1151f8ebec1aaad54878",
    "reservoir/UNISIM-II-D.zip":
        "9f817df86aafbf5f5d2f807842aa8ad2be4ec4512071ccc6cf5660deeee7fea5",
    _FEA: "57a4faa0cc73c91e1411c17e5e11ec8a74a33f3c36a80630bee21f3fc0776256",
    _TURBO: "4f615c2d7644e7ee504b9c548c8ef9095bfaaf19ac2a7071516f8978cbcc5385",
    "fea/aero_bracket/aero_bracket.vtu":
        "932bb12ed7a590b2e5314809695ebe33deadb6f12f2d7c889d92794942abd5ea",
    "fea/kiefer/dataset.vtu": "bc4856de9425095972022e79de8f4145df3ba99b09053f3315d7131f7e4e1324",
    "fea/pump_bracket/pump_bracket.zip":
        "d1ad75b0b74be1fb585be3d1621cef30b57f1550658f09554b0245e5ad2f2cf7",
    "notch_disp.vtu": "bd8ff6e0aa691c2503ac7ad994496849b2696d3d459fb5ad52ec6d092ba307c2",
    "Disc_BiQuadraticQuads_0_0.vtu":
        "2a14e4416fa751033253c0f78f58d04901bf0cd2bbaca369816e69bcd352d9f9",
    "QuadraticPyramid.vtu": "6e14d5c0999b71fb008e45fa6d065766678ba92b97de0e012bba4380505d21c7",
    "TriQuadraticHexahedron.vtu":
        "af6cd96d3095dcc49ba63c790ac69c176c2c4f1783e4c5de72b7b3ed6f6b0d4f",
    "StructuredGrid.vts": "1e21185bebebef969ac93ebc77c66dd3fd5502e3afe589a9f94b89d6917f6fbb",
    "SampleStructGrid.vtk": "7488c79cd0e6f49225fe66e2ff6a44eceebfe1b05541f0593add2b1ca93f3828",
    "kitchen.vtk": "cf50c31bd065aedbfca8074522a3aa278ab1007273e95a8cc4e5d279834b9a25",
    "office.binary.vtk": "b71ac0334e92829a431fe7aaff54cec1add42f4d50e42a56fa4917560c9c8de5",
    "RectilinearGrid.vtr": "f61e18645d1fb22ea6d9e6b0f46034b608b4ca8ffe96149c0e74a708279ce9a3",
    "carotid.vtk": "3f5dac7baa8ed60fa094f67b918d55add4d55f1d498fad587d1850915f60032d",
    "magpylib/coil_field.vti": "5b3833aaf6f227d63b967dd8ba3d094146e19658d5430b77e80e69c52756e56a",
    "lethe/result_particles.20000.0000.vtu":
        "b7bf4ff0fee5af4f948e8838c4f9429585e5e894094440b32c9cef780ab59b21",
    "point-clouds/hyg-database/stars.vtp":
        "09584c832abcfb04a01e027a9454c528beeee50ecb0c5b580d23454039925888",
}


def path(name):
    """The file dataset ``name`` is read from."""
    return DATA / DATASETS[name][0]


def present(name):
    return path(name).exists()


def read(name):
    """Dataset ``name`` as one VTK dataset."""
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (the pythonic constructors)
    from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet

    file = str(path(name))
    suffix = pathlib.Path(file).suffix
    if suffix == ".vtk":
        from vtkmodules.vtkIOLegacy import vtkDataSetReader

        out = vtkDataSetReader(file_name=file, read_all_scalars=True, read_all_vectors=True,
                               read_all_tensors=True, read_all_fields=True)()
    elif suffix in (".vtu", ".vts", ".vtr", ".vti", ".vtp", ".pvtu"):
        from vtkmodules.vtkIOXML import vtkXMLGenericDataObjectReader

        out = vtkXMLGenericDataObjectReader(file_name=file)()
    elif suffix == ".exo":
        from vtkmodules.vtkIOExodus import vtkExodusIIReader

        reader = vtkExodusIIReader(file_name=file)
        reader.UpdateInformation()
        for kind in (vtkExodusIIReader.ELEM_BLOCK, vtkExodusIIReader.NODAL):
            reader.SetAllArrayStatus(kind, 1)
        out = reader()
    elif suffix == ".foam":
        from vtkmodules.vtkIOGeometry import vtkOpenFOAMReader

        reader = vtkOpenFOAMReader(file_name=file)
        reader.UpdateInformation()
        reader.SetPatchArrayStatus("internalMesh", 1)
        out = reader()
    elif suffix == ".case":
        from vtkmodules.vtkIOEnSight import vtkGenericEnSightReader

        out = vtkGenericEnSightReader(case_file_name=file)()
    elif suffix == ".vrt":
        from vtkmodules.vtkIOGeometry import vtkProStarReader

        out = vtkProStarReader(file_name=file[:-len(suffix)])()
    else:
        raise ValueError(f"no reader for {file}")
    if isinstance(out, vtkCompositeDataSet):
        from vtkmodules.vtkFiltersParallel import vtkMergeBlocks

        out = vtkMergeBlocks()(out)
    return out


def _sha256(file):
    digest = hashlib.sha256()
    with open(file, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(verbose=True):
    """Download what is missing, check it and unpack zips; the sources that failed."""
    failed = []
    for name, (file, sources) in DATASETS.items():
        if present(name):
            continue
        for source in sources or (file,):
            out = DATA / source
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    urllib.request.urlretrieve(SOURCE.format(path=source), f"{out}.part")
                except OSError as e:
                    failed.append(source)
                    if verbose:
                        print(f"FAILED {source}: {e}")
                    continue
                if _sha256(f"{out}.part") != SHA256[source]:
                    failed.append(source)
                    if verbose:
                        print(f"FAILED {source}: its SHA-256 differs")
                    continue
                os.replace(f"{out}.part", out)
            if source.endswith(".zip"):
                with zipfile.ZipFile(out) as z:
                    z.extractall(f"{out}.unzip")
        if verbose:
            print(f"{name}: {'ok' if present(name) else 'missing'}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if fetch() else 0)
