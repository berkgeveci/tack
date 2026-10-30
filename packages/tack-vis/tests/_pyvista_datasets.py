"""PyVista's example datasets (``pyvista.examples.downloads``), read with VTK alone.

The files are those of the ``pyvista/vtk-data`` repository, in its layout (a
zip unpacked beside itself as ``<zip>.unzip/``, as PyVista caches them), under
``TACK_PYVISTA_DATA`` (default ``~/.cache/tack/pyvista``); a PyVista cache
(``PYVISTA_USERDATA_PATH``) is such a directory. ``python _pyvista_datasets.py``
fetches what is missing and checks it against ``SHA256``. Composite data
(Exodus, OpenFOAM, EnSight, CGNS, ...) is merged into one unstructured grid,
or read as one block (``_BLOCK``); large datasets are read smaller
(``_STRIDE``, ``_CROP``). A series (PVD) is its first step.

Not here, for want of a VTK reader: ``cloud_dark_matter`` and
``gpr_data_array`` (NumPy ``.npy``), ``dolfin`` (DOLFIN XML) and ``frd``
(CalculiX). Textures alone (the doorman's PNGs) are not datasets."""

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
_COOLING = "fvm/cooling_electronics/datasets.zip"
_CT_FEMALE = "whole_body_ct/s1380_resampled.zip"
_CT_MALE = "whole_body_ct/s1397_resampled.zip"
_NEK = ("nek5000/eddy_uv.nek5000", *(f"nek5000/eddy_uv0.f{i:05}" for i in range(1, 12)))
_HEADSQ = ("headsq/quarter.nhdr", *(f"headsq/quarter.{i}" for i in range(1, 94)))

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
    # More cells: a mesh as its reader gives it.
    "blow": ("blow.vtk", ()),
    "can_crushed_hdf": ("hdf/can-vtu.hdf", ()),
    "cells_nd": ("cellsnd.ascii.inp", ()),
    "electronics_cooling": (f"{_COOLING}.unzip/air.vtu", (_COOLING,)),
    "letter_a": ("a_grid.vtk", ()),
    "lshape": ("EnSight/LShape.case", tuple(f"EnSight/LShape{e}" for e in (
        ".case", "_geometry.geo", "_displacement.var"))),
    "meshio_xdmf": ("meshio/out.xdmf", ("meshio/out.xdmf", "meshio/out.h5")),
    "model_with_variance": ("model_with_variance.vtu", ()),
    "naca": ("EnSight/naca.bin.case", ("EnSight/naca.bin.case", *(
        f"EnSight/naca.gold.bin.{e}" for e in ("DENS_1", "DENS_3", "geo")))),
    "nek5000": ("nek5000/eddy_uv.nek5000", _NEK),
    "parallel_exodus": ("ParallelExodus/can.e.4.0",
                        tuple(f"ParallelExodus/can.e.4.{i}" for i in range(4))),
    "prism": ("prism.neu", ()),
    "room_cff": ("FLUENTCFF/room.cas.h5", ("FLUENTCFF/room.cas.h5", "FLUENTCFF/room.dat.h5")),
    "tecplot_ascii": ("tecplot_ascii.dat", ()),
    "tensors": ("tensors.vtk", ()),
    "tetra_dc_mesh": ("dc-inversion.zip.unzip/dc-inversion/mesh-inverse.vtu",
                      ("dc-inversion.zip",)),
    "warping_spheres": ("warping_spheres/warping_spheres.vtkhdf", ()),
    "cgns_multi": ("cgns/multi.cgns", ()),
    "cgns_structured": ("cgns/sqnz_s.adf.cgns", ()),
    "wavy": ("PVD/wavy.zip.unzip/wavy/wavy00.vts", ("PVD/wavy.zip",)),
    # Surfaces.
    "action_figure": ("tigerfighter_decimated.obj", ()),
    "antarctica_velocity": ("antarctica_velocity.vtp", ()),
    "armadillo": ("Armadillo.ply", ()),
    "bunny": ("bunny.ply", ()),
    "bunny_coarse": ("Bunny.vtp", ()),
    "cad_model": ("42400-IDGH.stl", ()),
    "cad_model_case": ("cad/4947746/Vented_Rear_Case_With_Pi_Supports.vtp", ()),
    "carburetor": ("carburetor.ply", ()),
    "cow": ("cow.vtp", ()),
    "crater_topo": ("Ruapehu_mag_dem_15m_NZTM.vtk", ()),
    "delaunay_example": ("250.vtk", ()),
    "doorman": ("doorman/doorman.obj", ()),
    "dragon": ("dragon.ply", ()),
    "face": ("fran_cut.vtk", ()),
    "faults": ("faults.vtk", ()),
    "foot_bones": ("fsu/footbones.ply", ()),
    "gears": ("gears.stl", ()),
    "horse": ("horse.vtp", ()),
    "louis_louvre": ("louis.ply", ()),
    "lucy": ("lucy.ply", ()),
    "pine_roots": ("pine_root.tri", ()),
    "room_surface_mesh": ("room_surface_mesh.obj", ()),
    "saddle_surface": ("InterpolatingOnSTL_final.stl", ()),
    "sextant": ("vrml/sextant.wrl", ()),
    "teapot": ("teapot.g", ()),
    "topo_global": ("EarthModels/ETOPO_10min_Ice.vtp", ()),
    "topo_land": ("EarthModels/ETOPO_10min_Ice_only-land.vtp", ()),
    "turbine_blade": ("turbineblade.ply", ()),
    "washington_bust": ("washington_bust/washington_bust.obj", ()),
    # Images, 2D and 3D.
    "beach": ("beach.nrrd", ()),
    "bird": ("Pileated.jpg", ()),
    "gourds": ("Gourds.png", ()),
    "moonlanding_image": ("moonlanding.png", ()),
    "knee": ("DICOM_KNEE.dcm", ()),
    "st_helens": ("SainteHelens.dem", ()),
    "t3_grid_0": ("t3_grid_0.mnc", ()),
    "bolt_nut": ("bolt.slc", ()),
    "brain": ("brain.vtk", ()),
    "brain_atlas_with_sides": ("avg152T1_RL_nifti.nii.gz", ()),
    "chest": ("MetaIO/ChestCT-SHORT.mha", ()),
    "damavand_volcano": ("damavand-volcano.vtk", ()),
    "dicom_stack": ("DICOM_Stack/data.zip.unzip/data", ("DICOM_Stack/data.zip",)),
    "embryo": ("embryo.slc", ()),
    "frog": ("froggy/frog.mhd", ("froggy/frog.mhd", "froggy/frog.zraw")),
    "head": ("HeadMRVolume.mhd", ("HeadMRVolume.mhd", "HeadMRVolume.raw")),
    "headsq": ("headsq/quarter.nhdr", _HEADSQ),
    "knee_full": ("vw_knee.slc", ()),
    "m4_total_density": ("m4_TotalDensity.cube", ()),
    "whole_body_ct_female": (f"{_CT_FEMALE}.unzip/s1380_resampled.vtm", (_CT_FEMALE,)),
    "whole_body_ct_male": (f"{_CT_MALE}.unzip/s1397_resampled.vtm", (_CT_MALE,)),
    # More point clouds.
    "gpr_path": ("gpr-example/path.txt", ()),
    "horse_points": ("horsePoints.vtp", ()),
    "lidar": ("kafadar-lidar-interp.vtp", ()),
    "particles": ("Particles.raw", ()),
    "sparse_points": ("sparsePoints.txt", ()),
    "thermal_probes": ("probes.vtp", ()),
}

#: headsq's 93 slices and Nek5000's 11 steps, as "file SHA-256" lines.
_SLICES = """
nek5000/eddy_uv.nek5000 b281069b4a4ec92cdc45ca9a8fd0d447e6ed4174225d24c754583c9d5d7b3089
nek5000/eddy_uv0.f00001 55b2f102021f4909b3179f09b7b4be58fec96dcf0f6688228f5c60979a8c59fc
nek5000/eddy_uv0.f00002 2fb8b8ccb33be87a541b35f018b3ab6911c639504c472865b2807fb754261b06
nek5000/eddy_uv0.f00003 22e983c0d560c65485dd30b5f6831f789c5f92800acee6445d546469c86d1fa9
nek5000/eddy_uv0.f00004 332a7569e207d9c6665a9c97fc95a4a76509a59641c8544b4b695b091859483f
nek5000/eddy_uv0.f00005 af250cb64627d4b851da923f876ace45e9dcfcf5b5c4d9ac4216a587601e33ce
nek5000/eddy_uv0.f00006 718baff164d2eb2e25d653ae8f0756e72baa2cdfb0bb5efb745b7f8e119afb93
nek5000/eddy_uv0.f00007 084ef2f4ca05292b549027df535762355aaf5075a28d86ac78e0d086bf814099
nek5000/eddy_uv0.f00008 3fca28d2ccad1611ef0c272121a9cd1f0b825ea4d38a7e056f9be591f458de92
nek5000/eddy_uv0.f00009 2a1ea0292f19e54be21b63be1e85c6b50b99c602a2e39c9b017f5603f1cd1e47
nek5000/eddy_uv0.f00010 7bbabf93b18b4f17ffab18aec5f36da0bbfe7914873f51e17f631d6b44f12b64
nek5000/eddy_uv0.f00011 72f4dc1a8754962a0c764b51372b764478e73868bc4bc592673c3bbade314ee7
headsq/quarter.nhdr 43430dd3b3db25bd535a2583be16110ca31c1dee4310e3c3e2134cc270739afb
headsq/quarter.1 d274caa86f844ccefc99ed4ee4050b89d0807831675554e2ce3d6a1115af04fe
headsq/quarter.2 ea4c76e8a1314536e5d5d88e83a83b7137e9e40607b3b1bb691697f2f5cfa00a
headsq/quarter.3 16ae310283e10eab4bbf629fc7d092c5d8b4b169925126c12d2943fca6bb9f96
headsq/quarter.4 d796e067300deeb0517e60c29a85b00636d9468c21ed5123c7758003ddae2528
headsq/quarter.5 11fb825ef8efc5be44c0fb89570db4387f75549d44710b914356494e275fba49
headsq/quarter.6 cd5e5b58213abb95f31c62159e5ca68509048dd9e099ab026bdfc7ef089a26f6
headsq/quarter.7 287fdff445218727aa9a2fea56f9af1e5635e6fa997ed6d0b46fe9fe8367a232
headsq/quarter.8 63239131d977710b49d21c234260dd09182a18ae535d31a1d0893e57342218e8
headsq/quarter.9 774f06f8b43774f4ca591ae11922e63294ddec7dc56d112b5b0d850ab3f3b80d
headsq/quarter.10 62b4399c186359d0d030bd4d5955fe4ca2fdeef4eba7697db9f21c8ac7cc050d
headsq/quarter.11 dca25e46f7df3be09bab385445c4f97449a0b37e2fdd6e1e2dbc15abb3bae6b2
headsq/quarter.12 e6910a26104f7187786827afa12e00161013c7d55c318267871e7de499ab180e
headsq/quarter.13 02f0fe8ffba357b6a8c3aa3dd08f946803addc3c6f0124ae23a4859dbc81c40d
headsq/quarter.14 9fc26aa61ff3fe9fb07b88a8381b2ac8acce83ade5c04cd94280ee68617bb1b3
headsq/quarter.15 b2fef0d491d1219e3d74bbd35527a991f019652bff4e493a8372ebe8c9b9c873
headsq/quarter.16 db7a49e2d56e8917558448f4836155381d493506312cf4ceb0417fb98c0164d4
headsq/quarter.17 1292be33c3b9c4a344fa80ae55911bc0889c45ac212064d829cd17d9346119c9
headsq/quarter.18 1f1f413d12f7f9bd7ec2d124a2e4cbe512c4a912ac2fc82812e34b2f801e2944
headsq/quarter.19 6d609442cbcaf1bb6b46a8367ff7fa1e78f5a73c08900ff9c94538d59637d179
headsq/quarter.20 21b975b6a287e4bc614ced1a37cefe2e132995b10feb91746fe85c284d7a206f
headsq/quarter.21 aacab7d0190a8049a7eef07be1a0c9c892e1df71c5e5583ab576ac8d3f10c4c8
headsq/quarter.22 053f1fdd3906caf7ee24284f9f3898fe53557249be01fe4704d2f77b187356be
headsq/quarter.23 4eebc3f19525be62225e4a16788ddabc6609f054e47f332bb6626935b3ee2276
headsq/quarter.24 95cea53665e9fb81a63b159009d6bddc97631291bc5f985a705c2312735bba83
headsq/quarter.25 fa6f322d95c0080138db16feb9ced5fbff9f992134b9efb82bf9219f5b336be5
headsq/quarter.26 4eb7f41176da50f750283fd83948d3def21de2fbe4061e1402b5e9d93eb08a59
headsq/quarter.27 8a6f30fd392a85ab7ce4384bddaa90cfc2df2cc2c2e2c95127375a38e1cccf2e
headsq/quarter.28 74570b8863ecc6a293cd452295555157dbdf083e1d916ea3d7fd9e6d8a7fb044
headsq/quarter.29 c7358075aa85d78187d810b28a60eaf671e92cd84609ee136cb7479cb5b8fafb
headsq/quarter.30 7546ea14aaff628653580dae9419703222daa27ba80106decb514a92e325775d
headsq/quarter.31 e7205a43df7b0dd2972c84a04f74a71313d9b447593a29b5ec054f03eae894f3
headsq/quarter.32 048e678048e7b1882ba40829371a94e686279e2cbbf3f0bb754deb00861b7cc0
headsq/quarter.33 130bec4de0668a765559329bc33f67330220ab0b2c3388db984d23d5f721d7ca
headsq/quarter.34 d15ac4fc3171c8e88c4224f056736e797a54424b43776a6e1278421a2fed9dbd
headsq/quarter.35 9ab7e10998dc9899d65e8186d7a4afc1eb2fbee302f53e7c78ba8134821fe6f2
headsq/quarter.36 211e82381d939331adb3da46a46b453efd630470f990bd88c3a926bc223c6883
headsq/quarter.37 0f3b302e76a40ceed395ce2fdc99b0cf9311a33569449e21a740ad885eef2d0a
headsq/quarter.38 e91c75787fe38d1d63e6147fe1fc0e672738066ba5f98fb138be2f125beea392
headsq/quarter.39 7caaa822f6ae12e2f57ad261f031e3f7fdd2a295033c5cd6ca6cb6f8783b3e38
headsq/quarter.40 bb96d9e78cc4d4576722d8608b1c536af7d4532aebc4fa95925b4678a17591c0
headsq/quarter.41 a696eb462997cba0f21f26618ed163e6d8c6270a68d64d97ee937874ecfca851
headsq/quarter.42 7c8c4556b2fe8d6f1def51d7405aaa1638a137bdf11dc63aa4d7eb3b8556df8a
headsq/quarter.43 0e60aacb6035c4541ece96fcdaf9b250f639e6c5174a4a6f536ad6fcbb57b421
headsq/quarter.44 7a5d2282b8521a8bcc17756c01e1595d9e99ea0deb1416a594500c242834363e
headsq/quarter.45 4b085eda2da6a8f724f88feb7ce2fdbbf17a8b3f7f8cb0c455ff25f345b699e2
headsq/quarter.46 356ac390b8b11471e89f226072923ef56bb120ec96f113a1237a022a534b8003
headsq/quarter.47 fe6a82787746189f7daf5ad9fdbd6d38ca3525e695bc992fb4e93e9646585fce
headsq/quarter.48 9123cd85dd2188a53e2bae8325a66c0160fdfc3054cb8442973389de076faf4d
headsq/quarter.49 9887f7fa336dd68336cef02cab9b5e1c4ce5726d28a8779e12e04ae06f9d178e
headsq/quarter.50 3a386c86de73689b15ae54fdddd7a29c161c7a813cc9a62e34e60db81adb0961
headsq/quarter.51 371c547b4c465bec310206739024cf9ebbf4320da7bc2ff9a935a4dd9bf25a4e
headsq/quarter.52 eef94b636c69971657ff3ccd085aa63c0f148e8f143939dcf9a5dce0763c9c84
headsq/quarter.53 4d9eacf7103fe42b50edfc7c57f4c16860c9a4d783744983019ab9fee4e9a439
headsq/quarter.54 be5ad0efbe8cb448d2315799c2a45f48efad301827d516be6072ae83ab963275
headsq/quarter.55 98e3745ee9633cc72e4966fd058fb6298ad833bbd609669b6ad19bfcf54ef1a2
headsq/quarter.56 c6b92b4b645348ea9b7f460588e6eae11679936cfa3305d2fb51a82af18fe249
headsq/quarter.57 472f525dfea1a9de2b5a1c0dc263ea25d80c3042cc8195eb950df2bcceae8f2e
headsq/quarter.58 b576f1060abda164ea3f7960bf902e102789939576dc90fbc5bb98abf580cbe5
headsq/quarter.59 470f06d6a23049d5cb858cfe9ce2929aab38339021e105ac57756ed3b11ed2eb
headsq/quarter.60 d50f874c22229cd302cfd289e064f1bdd7946970755c6789f26a059c2298549f
headsq/quarter.61 6f3949bb764163daab8468f51452116435b5499668eb3c7cd4a7ce6fb0af67ec
headsq/quarter.62 f8cf78335a73e344f435bdb56f0871bdc02d4b346edaf9111ddc04b5853481f1
headsq/quarter.63 aec7ce078f496626a93a98dcbec9abbdd7ac5dc436351c185335d6794f58f0eb
headsq/quarter.64 ae929c41d24172e94ff4135acf2b74d4fe2ccbc0386d5340dde3e4cfec2ae78d
headsq/quarter.65 2f15a49165bac411d20ead6d43cb29ea25d55bf077698cb6645d4b4b7841c7ff
headsq/quarter.66 6f15f6e96a1396e67951894941fec903143bf019567546609274b24b20639df1
headsq/quarter.67 827084a776017b3c6438bad15f2af33084c1d8b133ca7728c38ccffa14849313
headsq/quarter.68 97f2bf73645d8a906d5c7c8888f16c15edbd3ee8b1fa9c4594a3729856ac3b7c
headsq/quarter.69 61b80c64d0418a3f6c119689648739385464422fca932e5071f284e4b223d5a1
headsq/quarter.70 640f09eaab7c1ddedc031ed1de38005b92ce10f0158ffdb92f0c010f361cb3c8
headsq/quarter.71 97e9c61268c0181a0e8576f938bf4251840f0f862ea5c4f4ceff061bd101c33d
headsq/quarter.72 e96bd308c3ebca9472cf0d4567fb347bd9cec39a4937fadd90d5260c513f976f
headsq/quarter.73 ff93a58c8ed023e574521e21ac19cc610af2fe560b118cafab8a9809df81b463
headsq/quarter.74 1d6949704deb5d1fdc22304654044d23b49fdb1988b809513e18c595d1eb40ba
headsq/quarter.75 f5c8ff8e90791499f837b3de88fbb0c6c8eb40431dfd4dec1503bd0d32bc98dd
headsq/quarter.76 137f61e136695d933114129475307265e495942f106c21c7acd596ac2cf08dfd
headsq/quarter.77 424e6533fb3fbda2a181340369164f57255a1e31d0aba071c79a7625e494bd92
headsq/quarter.78 c23dea7c5e7b5f5010f024e83c4b8e37190445f76881796baeda80796c187f2c
headsq/quarter.79 8299fd00ee0c370686195dc17f519715542483def359cda5777e2dff7ac94cf7
headsq/quarter.80 d2fdfda3a3f62a2c025f19948945f3248b25abe94ea1f0132dac6d92d7b88af2
headsq/quarter.81 41117899f4c078b4d1a0950389e6bb452f61389a6ec78c9f2f0ceeb45d8ae1e7
headsq/quarter.82 adaafd1a69de193eaf5eb5e8ddf5c9529cb52525cccdfc3dc58ffb1d4369fd7c
headsq/quarter.83 d31e9cbf2a9cdfc11faad2ded94aac7d37fe3168550ee3f1879f340494181f1f
headsq/quarter.84 458e1d6a09b5054b95540084328ed60164ce7e47e79a494b922c81348b61349b
headsq/quarter.85 ec142457ef1cb4126a3cc5b0dada59b45e836974b9ca3e7186f6ac53910a1fe7
headsq/quarter.86 eec124d9ca3e3cc5c3a22d4d2e3887e29aa689e6316b06160161788bb8eb01e0
headsq/quarter.87 909423c4e93cc17aaa50412bbfc4e063925dbe64cc812e472f40df2437196084
headsq/quarter.88 2b5e4ce9d06b4fc5e393dbf7b70a6db8bab127f61c7c01a0a4216e175f8ba456
headsq/quarter.89 eaeb2ab76398608c1797e6d63fb0500449f7772c001a67538c8ca08897d5e6b2
headsq/quarter.90 7f5ffbb232c2baff406c3a82d8b291dc2bc6d1745296d9c4e4c3fc42d29df04a
headsq/quarter.91 ee8a2ca033ed36086df1b86e4dae4a759d8a5a16d99858f48d2bb4f9d472f912
headsq/quarter.92 08a1a8950b66a76ae9247181566943aa0319bac8211c816c2c0b988a1db98128
headsq/quarter.93 ae82ca217a9bf1fae260b5787360b664e5f8189c5ff13e604db6a2b18ea2ee8f
"""

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
    "blow.vtk": "a9a661052e214153c69c5a2aba62ac7900d625da46476488adcacc461e2174d5",
    "hdf/can-vtu.hdf": "2cbd2e06b45827f4dc763cd01a2f27aad18709a0aca58bfab9cbf7ea1cec1d4c",
    "cellsnd.ascii.inp": "7adc7aea8765ec6778cc2b70fccf6148e9e81afb56c9002bc813c376529c7277",
    _COOLING: "c144fcf2a2c9203198677014bec6067d7926ce4aa16315919e3d2a8e3886cc82",
    "a_grid.vtk": "4e6e4f5c1972e42c6d4102863ded8c01371f3d0854c49eb0406c02e8bc464b19",
    "EnSight/LShape.case": "5ecf71daf8aed179a55d534d457aa563be949654158e4ca238cc467f9a85dc94",
    "EnSight/LShape_geometry.geo":
        "74bd8d9d7682d3eb6c20401b6c6faea4d7aa962d5b82f549611fe70bbd4b7c40",
    "EnSight/LShape_displacement.var":
        "c9a5c4c138e0883fff404a1181ce02d3ddc5a496b63b57493d3bd6227075e3ba",
    "meshio/out.xdmf": "08b4d2b3fa6b7ede90942985f7e0d71e929866130848c472a0ac99ad73d88876",
    "meshio/out.h5": "17c0c198a4a1b32f50c64cfb7f69ed15bf2d99cc72ab6d874ab6f1a396fa00f8",
    "model_with_variance.vtu": "bf1947386b8a9313efcbb28e7748fed6226477e5b8cb14b6db3af7b805a948de",
    "EnSight/naca.bin.case": "3bd8ce362381d3074e0024a03fde1b527c510b55a9fe57d202177f0aabeca081",
    "EnSight/naca.gold.bin.DENS_1":
        "a446a966c393ad034c5140130109bc57816435283ff614b7b242c021eb6d0eea",
    "EnSight/naca.gold.bin.DENS_3":
        "9ca638601d5aea081bc3e5c24119c433bef902eb8c15312491002697e08b460f",
    "EnSight/naca.gold.bin.geo":
        "5779efa303eab4a578d1a884b3da680eacc3d425dd92bfdd2a571109a692cad7",
    "ParallelExodus/can.e.4.0": "4be7c6cea254ce839b2e4dabab77d329ec87ca3a656dd5a8b2220d1ebc5ea805",
    "ParallelExodus/can.e.4.1": "6d408dcd64bde26493dd59633c1dbf3ec6e235b21e8e7abe9b24de44207f9858",
    "ParallelExodus/can.e.4.2": "759607b72d15bf1876041af0632344f88a6ad3b6a975e63ea62e9c97d5edc715",
    "ParallelExodus/can.e.4.3": "a31ec7dc4f9d3db1fb8302ce49850ce9e9cf94716f3ac3fc13e72fe2f24b9fbd",
    "prism.neu": "4b50c31671d36bb4c85478f20766173312354de5a5740ef39a8c0e540f95885b",
    "FLUENTCFF/room.cas.h5": "cbfbb263036462392a5d53c2d8466d65596595513417a664962d482fb9b8bc83",
    "FLUENTCFF/room.dat.h5": "037f0fd543510cdb1b27a47f9b2a7e8fa9c6d6205661426df55c65321eefc276",
    "tecplot_ascii.dat": "266e4df03349a3c905038b3e371995b9ecf3c0ce6096ddeff1e3f359b6849741",
    "tensors.vtk": "cb4722a965253a8e5c0580bc791fd9b61030f144c0fbfb54e95d5caac9b9da83",
    "dc-inversion.zip": "47c1a073ce68ad72e62092761c91d1f1190eff09408596379af13d78c9fe4619",
    "warping_spheres/warping_spheres.vtkhdf":
        "11aeb0593838672f9615804377a9434486173c16121c5b42210727d8cac726af",
    "cgns/multi.cgns": "615bf8ef276261b4fa3af5ac75c8a0e5b200aca6ab5576206948f31fa335035b",
    "cgns/sqnz_s.adf.cgns": "e4ff8c84a3f1ba2d110902b9c4c011a33b4e351ae39714e9e17a982036de0bcf",
    "PVD/wavy.zip": "54b44a60b6fa92e222c4fcf61d947fbd897b72cc0d34f024716075ec08a6d352",
    "tigerfighter_decimated.obj":
        "be30d1b5ce8ed7149be9d925743f3ebaa8ef8151e052d4b331f3071333ba1b06",
    "antarctica_velocity.vtp": "675f5ab69eb387ae7398d187d2ffbf4509414c3935c5a6878c499096a53998f2",
    "Armadillo.ply": "102ccc712069c91d471fd003373bdb2c83c7f037bd85ca75b160f69c35cdd282",
    "bunny.ply": "442bdd8f35bf107005424bbd60100358cd57fa4711640766a5802a17d58f98ad",
    "Bunny.vtp": "aa0b1f7fc5210769ba707af4584b71b92dea365a8a6c71d8d6567c0d0be6d421",
    "42400-IDGH.stl": "943a3a869b92ed884c43c8f16d184468e24c69c197069adf73082ea5cb3df448",
    "cad/4947746/Vented_Rear_Case_With_Pi_Supports.vtp":
        "3445b4644d938f270442a3c5aa7baa68bbed46132c344bd7fae1bf557e7801bc",
    "carburetor.ply": "f0b4b240f91c271c763b5ac95c47bcfa7e1f9410e77fa359ca07092144e946e6",
    "cow.vtp": "d5aa4901d186902f90e17bf3b5917541cb6cb8cf223bfeea736631df4c047652",
    "Ruapehu_mag_dem_15m_NZTM.vtk":
        "1d8d85c97cccf598e85a29169ba8eb58b3187be230be4ed9e1e17451d2358d84",
    "250.vtk": "b2b37078a8a416c5275e60ce0a135cb3724e5e384f593318529816b8e39655e8",
    "doorman/doorman.obj": "ed9c6240ca823bb2cdf25f5806601ebe531bddfd93538af24fd1fee500e51dd0",
    "dragon.ply": "fea87ff48f2aba22fb53e7b67c3ff3f7b8c2a3b3a0653af62c48bba67c6d5744",
    "fran_cut.vtk": "499c170dd475725f47de4e436f080bd18d1a2cc315411f9c490ba42b6dffbc6d",
    "faults.vtk": "356633902699d73b0a29310397efe0bec3407fccde4b573d7d4403d0fb1fbff9",
    "fsu/footbones.ply": "9f5112422a4f73c3d7b4bc94741798c5ef732d9249d05d56282d0eb744c0c1bb",
    "gears.stl": "efd5c90522e787ed72926033b98796f71ddb1c3bf99d21e379e7456deb0e0138",
    "horse.vtp": "26f5e01d3abbe74bdfb76e517f196c352a5cba2d9af72e3d61fe28457d8b6adb",
    "louis.ply": "ecf27059148378a8f2b3a5e0b6f200690d51694865c9c3ab105de1cf0c6b14ab",
    "lucy.ply": "c57a95174c3c24d87c1ce8b44f48cbf0a5fcab47ce938880ec2acd497a478eb3",
    "pine_root.tri": "1d0430fbdcaf740ba988b72029f806b273c51c648fcdc907302711e78adc438e",
    "room_surface_mesh.obj": "1933cfaa63a9746b0799f869de3b11bb810ba5f1443da8f03e7f647d378226d9",
    "InterpolatingOnSTL_final.stl":
        "fe1d2a27455aff9984d05c3137eb4923b76332d7bd3142916c907804b7b88f2a",
    "vrml/sextant.wrl": "8d61c9dc02d027b29fe6a0a77a8f9b099005de971ae1915125eb2bbae11863ba",
    "teapot.g": "f46204ed8bf8bac829232c79791566a47c6e9ae7c784340192388ded8d58ad47",
    "EarthModels/ETOPO_10min_Ice.vtp":
        "89fff96333d351ad207bce0599c8a1df011853be8aaaa6face1f0c472d6676bf",
    "EarthModels/ETOPO_10min_Ice_only-land.vtp":
        "e59b77278a9c0449d6350ab1604da913aad9bff0440635091e5a782f15818cf3",
    "turbineblade.ply": "c2850452c86e3076e0d70f77859bee942c096153f687c66b85549f3e52d809cb",
    "washington_bust/washington_bust.obj":
        "b97b76991d612158ee1945a74cb9dd86c0d65e001c7912d862041f36b0850893",
    "beach.nrrd": "cbd88cff9964795ab97af018fcb9245f265a4b56f30edf003274fdc28a0c0e49",
    "Pileated.jpg": "d4200ebe479ac0b4cd92fa2dc050ebec19ca1cf503e2fb7161e03aa1dac3a889",
    "Gourds.png": "27577400c1d57a7bfd5fed0409073b6ae1a77880300decfd42acf09ade09be6a",
    "moonlanding.png": "160f1f46d416a2c30726629c87a01ab3d6af2655f6d83759e996774f7caaca2d",
    "DICOM_KNEE.dcm": "21d1d79e9174266d86df787c67d68987d86ebc11cf6eb2d5b1e2a08d837c503e",
    "SainteHelens.dem": "4b516d82778e6d8d10ffada64d2da41b3b3aad7e427af56de5a2a1666d3c746f",
    "t3_grid_0.mnc": "45efd966219cfc361aa7a09768ddf96d9e115449311e6b094c7c4869a45089a5",
    "bolt.slc": "499e03eb8f420f11dfa41c98d10524789c11c2751338734753d1248f2db3f95e",
    "brain.vtk": "afd61a7e4cd7ccf64d467e92bd576cf8e98fd5c2273cb5419eb6b27569db9b2e",
    "avg152T1_RL_nifti.nii.gz": "8a731128dac4de46ccb2cc60d972b98f75a52f21fb63ddb040ca96f0aed8b51a",
    "MetaIO/ChestCT-SHORT.mha": "0e43b3a5b6a618d8bbb856c0bef85abb1781870c4c756b63032f24fa4653f451",
    "damavand-volcano.vtk": "fca868b3242c62c00051520a40df5e69af459c519815c60b7b056c2b7dde05b1",
    "DICOM_Stack/data.zip": "4dfe8abdad0c60531061628c1fb0fa33e669451b68a6f3d9f6907df48aecd108",
    "embryo.slc": "512cd1216ed49b3feee783ad3c7ec3f80f59f814d03f65b50f74a97b3adb5d71",
    "froggy/frog.mhd": "bfa117c99504e92cc62a4c8e68da99db2f807a93ab332f1934f6a874cff0291e",
    "froggy/frog.zraw": "73c21b67f7e8f4f915eb262cb31b313c10e9d680b0eb537f002140902a84c7b8",
    "HeadMRVolume.mhd": "a2eecb88f8468b6091347723a54066e6635f830f80224ce904bb65170a2f39e6",
    "HeadMRVolume.raw": "714ff5b2db59d3867675d0f2419c24a71ed234985b39dc1ea83ee7d72110de4b",
    "vw_knee.slc": "ed64d1e536f4617ff990b6024f1c892c9c74058604fea73846d1f27bb2d0d435",
    "m4_TotalDensity.cube": "2e1839f23ca61709bd2407cfdccf2d3434bbe09cb11f56145ecaa24783156b34",
    _CT_FEMALE: "8114e0e210652c790790242f6d6a15a5afea64e60d4c356d5d85c1f2d2b65be6",
    _CT_MALE: "fd8ade8dd658d6c1db372cb7276f735483d3dfaa8292d13ed87f271a4620e7e0",
    "gpr-example/path.txt": "76cb8c14955605990ece3c1f284c0b7aca98c0771131b61a2123569ecb17ff78",
    "horsePoints.vtp": "08d1ec09d66ff854aa22905d0c8d97e8643aa360de216e39cc4c1f449eb64a3f",
    "kafadar-lidar-interp.vtp": "1e3ed11ad976866302287d289c3058c4d2417d791079c4b1a7a60f27a8353849",
    "Particles.raw": "73742bfa3b738aa4d1eaa33c7318d42604f4c835f7c49e6169662c51d4231f95",
    "sparsePoints.txt": "ddfaf9bd6b6ae8016a665683e620b3c715d9230d7ab8d70354f95d4ee4c99cf9",
    "probes.vtp": "df21f4f4e750188c93a90860d014b1a11b39756595ac901c2763b599d0ef64b2",
    **dict(line.split() for line in _SLICES.strip().splitlines()),
}


def path(name):
    """The file dataset ``name`` is read from."""
    return DATA / DATASETS[name][0]


def present(name):
    return path(name).exists()


#: Composites read as one of their blocks rather than merged: the whole-body
#: CTs are the CT (block 0) and a block of 117 segmentation masks on its grid;
#: meshio's XDMF a time series (its first step read) and the mesh again.
_BLOCK = {"whole_body_ct_female": 0, "whole_body_ct_male": 0, "meshio_xdmf": 0}

#: Datasets too large to run every algorithm on whole (in minutes) are read
#: smaller, to about 150,000 points or fewer. Images keep every n-th point
#: along each axis (``vtkExtractVOI``'s sample rate)...
_STRIDE = {
    "crater_topo": (4, 4, 1), "gourds": (2, 2, 1), "moonlanding_image": (2, 2, 1),
    "knee": (2, 2, 1), "dicom_stack": (4, 4, 1), "chest": (4, 4, 2), "bolt_nut": (2, 2, 2),
    "brain_atlas_with_sides": (2, 2, 2), "headsq": (2, 2, 2), "brain": (4, 4, 4),
    "whole_body_ct_female": (4, 4, 4), "whole_body_ct_male": (4, 4, 4),
    "knee_full": (5, 5, 5), "damavand_volcano": (5, 5, 5), "frog": (8, 8, 4),
    # Not 5: embryo.slc's last voxel of each plane is uninitialized memory
    # (PyVista zeroes it), and every 6th point of 256 never reaches it.
    "embryo": (6, 6, 6),
}

#: ...and other datasets keep the cells within a box about their bounds'
#: centre, this fraction of their extent along each axis (the Earth's
#: topography a band about its equator).
_CROP = {
    "electronics_cooling": 0.5, "model_with_variance": 0.5, "antarctica_velocity": 0.5,
    "carburetor": 0.6, "dragon": 0.5, "louis_louvre": 0.6, "pine_roots": 0.7,
    "washington_bust": 0.5, "armadillo": 0.7, "lidar": 0.4,
    "topo_global": (1, 1, 0.1), "topo_land": (1, 1, 0.3),
}


def _vtk_reader(file, suffix):
    """VTK's reader of ``file`` (by its suffix), run."""
    if suffix == ".vtk":
        from vtkmodules.vtkIOLegacy import vtkDataSetReader

        return vtkDataSetReader(file_name=file, read_all_scalars=True, read_all_vectors=True,
                                read_all_tensors=True, read_all_fields=True)()
    if suffix in (".vtu", ".vts", ".vtr", ".vti", ".vtp", ".pvtu", ".vtm"):
        from vtkmodules.vtkIOXML import vtkXMLGenericDataObjectReader

        return vtkXMLGenericDataObjectReader(file_name=file)()
    if suffix in (".exo", ".0"):                   # ".0": the first of parallel Exodus's
        from vtkmodules.vtkIOExodus import vtkExodusIIReader

        if suffix == ".0":
            from vtkmodules.vtkIOParallelExodus import vtkPExodusIIReader
            from vtkmodules.vtkParallelCore import vtkDummyController

            # Any one file of the series reads them all; it needs a controller.
            reader = vtkPExodusIIReader(file_name=file, controller=vtkDummyController())
        else:
            reader = vtkExodusIIReader(file_name=file)
        reader.UpdateInformation()
        for kind in (vtkExodusIIReader.ELEM_BLOCK, vtkExodusIIReader.NODAL):
            reader.SetAllArrayStatus(kind, 1)
        return reader()
    if suffix == ".foam":
        from vtkmodules.vtkIOGeometry import vtkOpenFOAMReader

        reader = vtkOpenFOAMReader(file_name=file)
        reader.UpdateInformation()
        reader.SetPatchArrayStatus("internalMesh", 1)
        return reader()
    if suffix == ".case":
        from vtkmodules.vtkIOEnSight import vtkGenericEnSightReader

        return vtkGenericEnSightReader(case_file_name=file)()
    if suffix == ".vrt":
        from vtkmodules.vtkIOGeometry import vtkProStarReader

        return vtkProStarReader(file_name=file[:-len(suffix)])()
    if suffix in (".hdf", ".vtkhdf"):
        from vtkmodules.vtkIOHDF import vtkHDFReader

        return vtkHDFReader(file_name=file)()
    if suffix == ".cgns":
        from vtkmodules.vtkIOCGNSReader import vtkCGNSReader

        reader = vtkCGNSReader(file_name=file)
        reader.UpdateInformation()
        reader.EnableAllPointArrays()
        reader.EnableAllCellArrays()
        return reader()
    if suffix == ".xdmf":
        from vtkmodules.vtkIOXdmf2 import vtkXdmfReader

        return vtkXdmfReader(file_name=file)()
    if suffix == ".nek5000":
        from vtkmodules.vtkIOParallel import vtkNek5000Reader

        return vtkNek5000Reader(file_name=file)()
    if suffix == ".h5":                             # FLUENT's CFF case (and data beside it)
        from vtkmodules.vtkIOFLUENTCFF import vtkFLUENTCFFReader

        return vtkFLUENTCFFReader(file_name=file)()
    if suffix == ".inp":
        from vtkmodules.vtkIOGeometry import vtkAVSucdReader

        return vtkAVSucdReader(file_name=file)()
    if suffix == ".neu":
        from vtkmodules.vtkIOGeometry import vtkGAMBITReader

        return vtkGAMBITReader(file_name=file)()
    if suffix == ".dat":
        from vtkmodules.vtkIOGeometry import vtkTecplotReader

        return vtkTecplotReader(file_name=file)()
    if suffix == ".obj":
        from vtkmodules.vtkIOGeometry import vtkOBJReader

        return vtkOBJReader(file_name=file)()
    if suffix == ".ply":
        from vtkmodules.vtkIOPLY import vtkPLYReader

        return vtkPLYReader(file_name=file)()
    if suffix == ".stl":
        from vtkmodules.vtkIOGeometry import vtkSTLReader

        return vtkSTLReader(file_name=file)()
    if suffix == ".g":
        from vtkmodules.vtkIOGeometry import vtkBYUReader

        return vtkBYUReader(geometry_file_name=file)()
    if suffix == ".tri":
        from vtkmodules.vtkIOGeometry import vtkMCubesReader

        return vtkMCubesReader(file_name=file)()
    if suffix == ".wrl":
        return _vrml(file)
    if suffix == ".raw":                            # vtkParticleReader's x, y, z, value
        from vtkmodules.vtkIOGeometry import vtkParticleReader

        reader = vtkParticleReader(file_name=file)
        reader.SetDataByteOrderToBigEndian()
        return reader()
    if suffix == ".txt":
        return _delimited_points(file)
    if suffix in (".nrrd", ".nhdr"):
        from vtkmodules.vtkIOImage import vtkNrrdReader

        return vtkNrrdReader(file_name=file)()
    if suffix in (".mha", ".mhd"):
        from vtkmodules.vtkIOImage import vtkMetaImageReader

        return vtkMetaImageReader(file_name=file)()
    if suffix == ".gz":                             # .nii.gz
        from vtkmodules.vtkIOImage import vtkNIFTIImageReader

        return vtkNIFTIImageReader(file_name=file)()
    if suffix == ".jpg":
        from vtkmodules.vtkIOImage import vtkJPEGReader

        return vtkJPEGReader(file_name=file)()
    if suffix == ".png":
        from vtkmodules.vtkIOImage import vtkPNGReader

        return vtkPNGReader(file_name=file)()
    if suffix == ".slc":
        from vtkmodules.vtkIOImage import vtkSLCReader

        return vtkSLCReader(file_name=file)()
    if suffix == ".dem":
        from vtkmodules.vtkIOImage import vtkDEMReader

        return vtkDEMReader(file_name=file)()
    if suffix == ".mnc":
        from vtkmodules.vtkIOMINC import vtkMINCImageReader

        return vtkMINCImageReader(file_name=file)()
    if suffix in (".dcm", ""):                      # a file, or a directory of slices
        from vtkmodules.vtkIOImage import vtkDICOMImageReader

        if suffix:
            return vtkDICOMImageReader(file_name=file)()
        return vtkDICOMImageReader(directory_name=file)()
    if suffix == ".cube":                           # the grid, not the molecule
        from vtkmodules.vtkIOChemistry import vtkGaussianCubeReader2

        reader = vtkGaussianCubeReader2(file_name=file)
        reader.Update()
        return reader.GetGridOutput()
    raise ValueError(f"no reader for {file}")


def _delimited_points(file):
    """A table of points (its first three columns x, y and z, the rest point
    data), delimited by tabs or spaces, as ``vtkTableToPolyData`` makes it."""
    from vtkmodules.vtkFiltersGeneral import vtkTableToPolyData
    from vtkmodules.vtkIOInfovis import vtkDelimitedTextReader

    with open(file) as f:
        delimiter = "\t" if "\t" in f.readline() else " "
    table = vtkDelimitedTextReader(file_name=file, have_headers=True,
                                   detect_numeric_columns=True,
                                   field_delimiter_characters=delimiter)()
    points = vtkTableToPolyData()
    points.SetXColumnIndex(0)
    points.SetYColumnIndex(1)
    points.SetZColumnIndex(2)
    return points(table)


def _vrml(file):
    """A VRML scene's shapes as one polydata: ``vtkVRMLImporter`` makes an
    actor of each, its mapper's input placed by the actor's matrix."""
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData
    from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
    from vtkmodules.vtkIOImport import vtkVRMLImporter
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow

    window = vtkRenderWindow()
    window.AddRenderer(vtkRenderer())
    importer = vtkVRMLImporter(file_name=file, render_window=window)
    importer.Update()
    append = vtkAppendPolyData()
    actors = importer.GetRenderer().GetActors()
    actors.InitTraversal()
    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        mapper = actor.GetMapper()
        mapper.GetInputAlgorithm().Update()
        transform = vtkTransform()
        transform.SetMatrix(actor.GetMatrix())
        append.AddInputData(vtkTransformPolyDataFilter(transform=transform)(
            mapper.GetInputDataObject(0, 0)))
    append.Update()                                 # (a call wants input connections)
    return append.GetOutput()


def read(name):
    """Dataset ``name`` as one VTK dataset."""
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (the pythonic constructors)
    from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet, vtkImageData

    file = str(path(name))
    out = _vtk_reader(file, pathlib.Path(file).suffix)
    if name in _BLOCK:
        out = out.GetBlock(_BLOCK[name])
    if isinstance(out, vtkCompositeDataSet):
        from vtkmodules.vtkFiltersParallel import vtkMergeBlocks

        out = vtkMergeBlocks()(out)
    if name in _STRIDE:
        from vtkmodules.vtkImagingCore import vtkExtractVOI

        assert isinstance(out, vtkImageData), name
        out = vtkExtractVOI(voi=out.extent, sample_rate=_STRIDE[name])(out)
    if name in _CROP:
        out = _cropped(out, _CROP[name])
    return out


def _cropped(data, fraction):
    """The cells of ``data`` within a box about its bounds' centre, ``fraction``
    of its extent along each axis (one, or one per axis; polydata stays polydata)."""
    from vtkmodules.vtkCommonDataModel import vtkBox, vtkPolyData

    low, high = data.bounds[::2], data.bounds[1::2]
    fraction = fraction if isinstance(fraction, tuple) else (fraction,) * 3
    centre = [(a + b) / 2 for a, b in zip(low, high)]
    half = [f * (b - a) / 2 for f, a, b in zip(fraction, low, high)]
    box = vtkBox()
    box.SetBounds(*[c + s * h for c, h in zip(centre, half) for s in (-1, 1)])
    if isinstance(data, vtkPolyData):
        from vtkmodules.vtkFiltersExtraction import vtkExtractPolyDataGeometry

        return vtkExtractPolyDataGeometry(implicit_function=box)(data)
    from vtkmodules.vtkFiltersExtraction import vtkExtractGeometry

    return vtkExtractGeometry(implicit_function=box)(data)


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
