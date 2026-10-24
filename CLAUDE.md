# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Monorepo Structure

Tack is split into 2 packages under `packages/`:

| Package | Path | Contents |
|---------|------|----------|
| `tack-core` | `packages/tack-core/` | Kernels, arrays (and `tack.arrays`' implicit arrays), types, IR, codegen, backends, scan/sort/copy |
| `tack-vis` | `packages/tack-vis/` | `tack.dataset` (data model and algorithms), VTK/MFEM interop |

The path tracer that was `tack-rendering` is archived in `archive/tack-rendering/`
(outside the workspace, CI and docs; its design notes are in its `NOTES.md`)
until the dataset API settles. Ruff excludes `archive/`.

Both share the `tack` namespace via `pkgutil.extend_path` in tack-core's `tack/__init__.py`; tack-vis's `tack/` has no `__init__.py` (a namespace portion), so tack-core's runs whichever comes first on the path (`test_namespace.py`; CI's `wheels` job checks the built wheels). `tack` declares `__all__` and is marked typed (`py.typed`; tack-vis's in `dataset/` and `interop/`): a public name re-exported from `tack/__init__.py` must be in `__all__`, or type checkers read it as private. Public submodules (`tack.arrays`, `tack.algorithms`, `tack.math`, `tack.random`, `tack.cache`, `tack.dataset`, `tack.interop`) load on first attribute access (`__getattr__`).

**One catch worth knowing.** `extend_path` merges the *contents* of same-named directories, but only one `__init__.py` runs — the first found on the path. So only one package may ship a given `tack/<name>/` subpackage: `tack/algorithms/` is tack-core's alone (scan, sort, stats, copy). tack-vis's visualization algorithms are `tack.dataset.algorithms`. `packages/tack-vis/tests/test_namespace.py` checks tack-vis ships no `tack/algorithms/` and that the retired `tack.data`, `tack.deprecated` and `tack.fe` stay gone.

## Build & Test Commands

```bash
uv sync --extra cpu --extra dev                             # install everything the tests need
uv sync --extra cpu --extra dev --extra interop             # ...and VTK and MFEM, which the dataset tests compare against
uv run pytest packages/tack-core/tests/test_cpu_jit.py        # the files your change touches
uv run pytest -m quick -n 4                                   # the inner-loop set, 80 s on 4 workers
uv run pytest -n auto                                         # the whole suite across processes
uv run pytest                                                 # the whole suite, serially (CI's way)
uv run pytest packages/tack-core/tests/test_hip.py            # run HIP backend tests
uv run pytest -k "test_saxpy"                                # run tests matching a pattern
uv run python packages/tack-core/examples/validate_all.py     # validation suite
uv run python packages/tack-core/examples/01_hello_tack.py --arch hip  # example on backend
```

### Tests, in three tiers

Run the smallest tier that covers the change; the whole suite is for a
validation round or a merge gate, once per machine, not for iteration.

1. **The files the change touches.** Seconds to a minute.
2. **`-m quick`**: a module per area of the code (dispatch, the variant
   cache, each backend's main file, templates, the dataset API and its
   algorithms), marked `pytestmark = pytest.mark.quick`
   at module level; about 1900 tests, ~300 s of test time, 80 s on four
   workers (a 16-thread Xeon with an RTX 4060 Ti). When a new area gets a
   test module, mark it quick if it runs in under ten seconds; the heavy
   modules (polyhedra, implicit, MFEM interop, flow, clip, the sort, the
   kernel-cache recycle test) stay out so the tier stays quick.
   `test_dataset_api.py` is the one exception, 180 s of the tier's 300:
   it is the dataset API's main test and its cost is spread over 169
   tests at a second each, so it only costs a second or so of wall time
   per worker.
3. **The whole suite**, with `-n auto` (pytest-xdist, in the `dev` extra):
   6000 tests, 33 minutes of test time, 4:30 on eight workers here, with
   the same counts as the serial run. Tests are handed out one at a time
   (`--dist load`, xdist's default); no module depends on its tests'
   order. Every xdist worker is a process with its own backends and CUDA
   context, so GPU memory scales with the worker count. The `timing`
   tests skip themselves on a busy host, so a parallel run skips them;
   run `-m timing` alone when they matter. A test whose assertion is about
   what the scheduler or the OS did (timings, wake-ups, NUMA placement)
   calls the root conftest's `require_idle` fixture, or better, asserts
   with a tolerance a loaded host passes.

Do not edit a `.py` file while a run that imported it is still going:
`linecache` reloads the changed file and every source lookup after that
reads the wrong line, in pytest (`???` in the traceback) and in Tack
(`@tack.func must decorate a function, got <class 'ast.Assign'>`), which
looks like order dependence and is not.

Examples accept `--arch` to select the backend. `09_shared_memory.py`
requires an explicit GPU choice (`metal|cuda|hip|level_zero`); CPU has no
workgroup execution model.

On this Mac, sandboxed processes cannot discover the Apple M1 Max GPU:
`MTLCreateSystemDefaultDevice()` returns `None` even with the bindings installed.
Run Metal hardware validation with GPU access outside that sandbox and explicitly
make the Metal device (`tack.set_default_device("metal")`) first. A sandboxed CPU-only pytest collection does not
validate Metal. Keep `TACK_NO_REINIT` unset for multi-backend runs.

No build step — pure Python with JIT compilation at runtime.

## Architecture

Tack is a Python-first GPU compute framework inspired by Taichi. Kernels are decorated Python functions that compile at call time through this pipeline:

```
@tack.kernel Python function
    → Source validation (source_validation.py)
    → AST transform (ast_transform.py, with its parts in lang/lowering/)
    → Tack IR (ir.py)
    → IR passes: resolve (ir_resolve.py) → type inference → optimize (ir_optimize.py)
    → Backend-specific codegen + dispatch
```

### Codegen backends from Tack IR

| Backend | Codegen | Runtime compilation | Dispatch |
|---------|---------|-------------------|----------|
| CPU | `llvm_gen.py` → LLVM IR | llvmlite JIT | ctypes function call |
| Metal | `msl_gen.py` → MSL source | Metal API (pyobjc) | compute pipeline |
| CUDA | `cuda_gen.py` → CUDA C source | NVRTC → PTX | cuLaunchKernel |
| HIP | `hip_gen.py` → HIP C source (extends CUDA) | hipRTC → code object | hipModuleLaunchKernel |
| Level Zero | `opencl_gen.py` → OpenCL C source (extends CUDA) | libocloc → SPIR-V | zeCommandListAppendLaunchKernel |

### Key abstraction: Array with DeviceBuffer

`Array` (in `lang/array.py`) holds a `DeviceBuffer` — the backend-specific storage:
- **CPU**: `NumpyBuffer` wraps a numpy array. CPU backend accesses via ctypes pointers.
- **Metal**: `MetalBuffer` — Metal shared buffer (zero-copy unified memory on Apple Silicon). The numpy array view points directly into Metal buffer memory.
- **CUDA**: `CUDABuffer` holds a stream-ordered device pointer (`cuMemAllocAsync`/`cuMemFreeAsync` on the context's one stream, so a temporary dropped while a queued launch uses it outlives the launch). A device without memory pools (a GPU in TCC mode on Windows: `_has_memory_pools`) allocates with `cuMemAlloc`; since `cuMemFree` waits for the whole device, a dropped buffer is queued on its context (`_ContextToken.dropped`) and freed once the stream is idle, at the next allocation or sync (`_free_dropped`). Such a device has no virtual memory management either, so it cannot export (`_has_virtual_memory`). Launches, fills and copies are queued on that stream (`_ContextToken.queue()`) and return; reads (`to_numpy`, `read_range`, reductions) first wait for the stream (`_drain`, `cuStreamSynchronize`, which holds nothing a launch needs), never inside a pageable copy, which holds the context. A launch's error surfaces at the next wait. Memory handed outside Tack (`wrap_ptr`, DLPack export, `share()`) and texture launches complete before returning. See `test_cuda_async.py`.
- **HIP**: as CUDA, with one blocking stream per device (`_Queue`, `_queue()`, ordered with the null stream) and memory from `hipMallocAsync`/`hipFreeAsync` (`hipMalloc`/`hipFree` on a device without pools). Every wait is on an event recorded after the queued work (`_StreamMarker`, `_wait_stream`), through ctypes on the runtime library (`_runtime`), never hip-python's `hipStreamSynchronize` (see the vendor workarounds), and a read waits before its copy. A small host write from pageable memory may be staged and queued; a large one waits. `wrap_ptr`, DLPack export (`share()`) and texture launches complete before returning. See `test_hip_async.py`.
- **Level Zero**: `L0Buffer` holds a `zeMemAllocDevice` pointer. Launches, fills and copies are appended to the backend's one asynchronous immediate command list (`_queue`), each followed by a barrier (`_ordered`) to keep program order. Reads queue a copy and a marker under `_launch_lock` and wait on the marker's event outside it (`_ListMarker`); `tack.sync()` is `zeCommandListHostSynchronize`. Copies to the device up to 16 MiB and from it up to 64 KiB go through pooled USM host memory (`_HostStaging`), and host memory a queued copy or fill reads is held (`_hold`) until the next wait. Arrays are freed with `zeMemFreeExt(DEFER_FREE)` (`_free`). `share()`, `wrap_ptr`, DLPack export and textures complete before returning; a driver without `zeCommandListHostSynchronize` gets a synchronous list. See `test_level_zero_async.py`.
- **`asyncio`** (`runtime/aio.py`, optional): `await tack.sync_async()` and `await array.numpy_async()` wait in one of Tack's worker threads (`tack-wait`, 4) so an event loop keeps running; kernel calls stay plain calls. A backend opts in with `Backend.async_marker()`, an object whose blocking `wait()` covers every launch queued so far and raises a launch's error (CUDA and HIP `_StreamMarker`, HIP's through ctypes; Metal `_CommandBufferMarker`; Level Zero `_ListMarker`, `_take_event`), and buffers with `DeviceBuffer.waits_off_thread` and `thread_binding()` (`_BoundContext`, `_BoundDevice`). Without a marker (the CPU) the wait is inline. The mark is taken at the call. See `test_asyncio.py`.

The measurements behind these choices (wait times, launch counts, thread schedules) are in `docs/design/backend-implementations.md` ("How each backend queues and waits") and `docs/design/memory-and-aliasing.md` ("New arrays, fills and one-element reads").

`tack.empty()` (under every array constructor) calls `backend.allocate_array()`. A new array's contents are unspecified on every backend; `tack.zeros` fills it, and `Array.fill` first offers the fill to `Backend.fill_array`, which the CPU backend answers for 256 KiB or more by running `tack.algorithms.copy._fill_kernel` through the normal dispatch, so the workers fill (and first-touch) the pages. On Linux `TACK_CPU_MALLOC=1` tells glibc to keep freed pages (`_tune_malloc`; off by default, since it is process-wide). `Array.__getitem__` reads one element through `DeviceBuffer.read_range(start, count)`, which copies just that range after the queued launches (each GPU backend checks the range against the allocation); `__setitem__` and `__iter__` raise.

### Backend contract

All five backends subclass `Backend` (`runtime/backend.py`), which declares the required methods (`allocate_array`, `wrap_ptr`, `execute`), the hooks with defaults (`fill_array`, `synchronize` for backends that queue launches, `memory_space`, `reduce_array`, the texture hooks, and the disk cache's `cache_target`/`variant_artifact`/`load_variant`), and the capability attributes callers read instead of probing with `hasattr`: `name`, `ordinal`, `display_name`/`label`, `supported_dtypes`, `supports_f64`, `supports_device_reductions`, `supports_workgroups`, `device_options`, `device_memory_spaces`, `memory_domain`.

The four GPU backends derive from `GPUBackend` (`runtime/gpu_backend.py`), which owns `execute` (resolve the variant, evaluate the launch range, bind textures, replace scalars with packed buffers, queue, complete if the launch binds shared memory or a texture image), `_build_variant` (pack, verify, annotate, compile) and `_store_texture_shapes` (software sampling where `texture_in_hardware` says no). A backend supplies `_compile_kernel` and `_launch`, sets `_max_launch` and `_texture_image_type`, and overrides the hooks where it differs: `_device_scope` (entered around the whole of `execute`: CUDA makes its context current, HIP its device; nothing on Metal and Level Zero), `_launch_guard` (the variant's `dispatch_lock` by default, Level Zero's `_launch_lock`, nothing on Metal), `_variant_scalar_buffers`/`_scalar_buffers` (one pack per variant rewritten per launch by default; Metal makes `_Uniforms` per launch) and `_must_complete` (false on Metal, whose launches complete shared memory themselves). A fix to dispatch belongs there, once. Their code generators share `CFamilyCodeGen` (`codegen/c_family.py`): `CUDACodeGen` (with `HIPCodeGen` and `OpenCLCodeGen` under it) and `MSLCodeGen` keep only what their dialect spells differently, and the dialect's name, its 64-bit index type (`_INDEX_TYPE`) and whether it has f64 (`_HAS_F64`) are class attributes. A method that names a module constant is not dialect-neutral even when its text matches: `_INT` was `long long` in CUDA and `long` in MSL. Compare generated sources before and after a codegen refactor.

### Devices

A device is a backend wrapped as a value (`runtime/dispatch.py`). `tack.device(spec, **options)` takes `"cpu"`, `"metal"`, `"cuda"`/`"cuda:0"`, `"hip"`, `"level_zero"` (numbered types take `:N`; a bare numbered spec resolves per call through `Backend.current_ordinal()`: the device of a CUDA context Tack did not make, `hipGetDevice()`, else 0). Without options it comes from the registry `_devices`, one backend per `(type, index)`, made on first use and the same afterwards, so arrays and compiled kernels survive a change of default; options (`Backend.device_options`: CPU `num_threads`, Level Zero `external_context`) construct a new backend each call. CUDA, HIP and Level Zero are `multi_device` (constructor `ordinal=`, `device_count()`, public `tack.device_count(type)`); `ordinal` is the runtime's device number, so `Device.index` and DLPack's `device_id` agree with it. CUDA keeps one context per device (`_ContextToken.device`; `_context_for` adopts a current context on the device, shares a token Tack holds for it, or retains the device's primary context -- the one PyTorch, CuPy and Warp use, so their making it current never strands Tack -- making it current only if the thread has none, and releasing it (`cuDevicePrimaryCtxRelease`), not destroying it, when its last backend goes) and adds the context to `memory_domain`; Level Zero gets each device its own context. The CUDA driver works in the thread's current context and HIP on its current device, so once a context is not current where its device was made, or there are two (`_switching`, set for good), every CUDA operation runs in `_in_context(token)` and every HIP one in `_on_device(device)` (also once a HIP device other than 0 is used, since threads start on 0), which make it current and restore the thread's; otherwise nothing switches. Buffers and texture images take their backend's token or device. `test_multi_device_switching.py` checks this against fake runtimes that refuse calls in the wrong context; `test_multi_device.py` needs two real devices of a type. `tack.Device` (`.type`, `.index`, `.synchronize()`, `str` → `"cuda:0"`, equality by backend identity) is one per backend, `dispatch.as_device(backend)`, cached on the backend. `with device:` pushes onto a per-thread stack (`_scope`, checked only while `_open_scopes` is nonzero); `set_default_device` sets the process default (`_current_backend`, the CPU if unset; `TACK_NO_REINIT` keeps an existing one); `get_backend()` reads stack then default. `set_debug` replaces `init(debug=)`; `TACK_DEBUG` is read at import. There is no `tack.init` and no `tack.cpu`/`tack.cuda` constants.

A space is also a `for_each` domain: `DataSet.node_views(space)` gives entity views (a `_Points` kind for points, edges, quad faces, cells) with a node-dof mixin (`_PointDof`, `_EdgeDof`, `_FaceDof`, `_InteriorDof`, `_CornerDof`, `_OffsetDof`, `_CellDof`: `node_count(i)`, `node_dof(i, j)`) and a node-position mixin (`_MeanPosition` on a straight geometry, `_GeometryAtDof` on an order-2 one, `_CornerPosition`, `_CenterPosition`), added through `domain_view(..., extra, **attributes)`; `interpolate` is one kernel over them (`_at_space_nodes`), its first-writer path kept only for an `L2` geometry's shared nodes.

`tack.dataset` follows the same model: a topology, field, set or dataset has `_device` (what `same_device`/`on_inputs_device` read; `same_device` skips classes, a classmethod's first argument) and `.device`. Topologies find it from their stored arrays (`_STORED`, cached once found); `StructuredTopology` records the device it was made for (`device=`); derived `Faces`/`Edges`/`PolygonFaces`/`SizeBuckets` take theirs from what they hold (`_OnItsArrays`); spaces from their topology. Every public algorithm, topology derivation (`groups`, `faces`, `edges`, `point_links`, `cell_points`), space layout, `Set` builder, `CellLocator`, bucket helper and `for_each`/`launch_groups`/`domain_view` is wrapped in `on_inputs_device`, so it runs and allocates on its dataset's device; a new entry point that allocates needs the decorator too. `Field` and `DataSet` refuse parts on two devices; `DataSet.to_device` rebuilds the topology (`_moved`), spaces (`Space._rebuilt_on`), fields and sets there. Implicit arrays live where the arrays they hold are: `ConstantArray`, `UniformCoordinates`, `CartesianProduct` take `device=`, `_Implicit.to_device` moves one, `_operands` stores host operands beside device ones, and `_metadata` refuses two devices. See `packages/tack-vis/tests/test_dataset_devices.py` (every algorithm on a simulated second device).

A `StructuredTopology` (2D or 3D) derives its faces and edges by arithmetic (`Faces._derive_structured`, `Edges._derive_structured`: one pass over the cells, each local face's or edge's id from (i, j, k) and per-shape tables, `_structured_face_tables` reading side 1's orientations once from the sorted derivation of a tiny grid), numbered exactly as the sort numbers them (`_before`); any other topology emits, sorts and matches (`_derive_sorted`). `test_structured_faces_and_edges_are_the_general_derivations` holds the two to identical arrays.

An `UnstructuredTopology`'s `types`, `offsets` and `connectivity` may be implicit arrays (`ConstantArray`, `CountingArray`), kept as given (`_ids_of`, `_as_field`); `uniform_cells` makes the types and offsets of cells all of one shape, and a topology whose types are a `ConstantArray` groups without the type histogram (`_gathered`). On the CPU a stored types array is counted by chunks into private tables and summed (`_type_counts`: `_count_types`, `_sum_type_counts`), with no atomics: one atomic counter per type put every worker on one counter when the types are equal (34 ms for 1M cells against 2.7 on one thread); GPUs keep the atomic histogram. Filters with one output shape emit them; code that reads a topology's arrays must accept an implicit array (index it in a kernel; `numpy()`, `np.asarray`, `materialize` on the host; `size_of`, never `.shape`). The VTK interop maps them to `vtkConstantArray`/`vtkAffineArray` (`_to_vtk_types`, `_to_vtk_ids`, `_from_vtk`). See `test_implicit_topology.py`.

Polyhedra have no reference element, so values inside them are linear on *sub-tetrahedra*: each face fanned from its points' mean, the fans from the cell's points' mean (both in value too, so linear fields are exact; `docs/design/polyhedra.md` section 13). Cell gradients are Green-Gauss over the fanned faces (`_polyhedral_gradients`, one walk per face); `cell_centers`/`values_at_centers` give the points' mean. The cell view locates and the field view evaluates, since a field view has no positions and a device function cannot call another view's methods: `views._PolyhedralLocation` (`locate`/`locate_from` give `(t, pc, inside)`, a sub-tetrahedron, its coordinates and whether it holds the point, which callers rank first; `cell_point`, `face_point`, `tet_jacobian`) on polyhedral domain views, and `_PolyhedralPointField`/`_PolyhedralCellField` (`tet_value`, `tet_parametric_gradient`), which `Space.mixin_for(key, shape)` picks by shape (`_H1Field.value` names shape functions a `Polyhedron` lacks, and every method of a view is resolved). A `CellLocator` records its topology and geometry and the geometry's `_revision` (`serves(data)`); `probe` and advection take one for any dataset of that topology and geometry and refuse a stale one, and without one use the locator the topology keeps (`cached_locator`, `_Topology._locator`, one at a time), so repeated probes build the bins once. `CellLocator.find` returns `Locations(cells, pcs, tets)` on every dataset (`tets` None except on polyhedra), searching from the points where `tack.dataset.by_id` can read cells by id (`_MeshSearch.find` over `cell_bins()`, ranked as the cell-side search ranks; `_PointValues`/`_CellValues.value(mesh, e, kind, pc, tet)` for probe and advection, which share it) and from the cells elsewhere, or for a fresh locator's first query of fewer points than a quarter of its cells and refuses cells not star-shaped from their cell point (`_not_star_shaped`); the advection kernel's `_PolyhedralMesh` is built from the same mixins. Polygons locate in the triangles fanning them from their cell point (`views._PolygonLocation`, `t = 2k`, `pc = (0, w_a, w_b)`, so `_PolyhedralPointField.tet_value` reads them unchanged) and polylines on their segments (`_PolylineLocation`, `_PolylinePointField`); `by_id` has `_PolygonMesh` and `_PolylineMesh`, and every mesh says `NO_REFERENCE`, which the search and evaluation branch on. VTK 9.7.1's `vtkPolyhedron` is no reference for these (wrong `Derivatives` and `EvaluatePosition`; see section 13). See `test_polyhedra.py`, `test_probe.py`, `test_flow.py`, and `benchmarks/polyhedra_compare.py` for the cost against hexahedra.

A `PartitionedDataSet` (`dataset/partitioned.py`) is an ordered list of `DataSet`s, each with its own device and no `_device` of its own. Per-piece work takes a piece and the caller iterates; only algorithms whose answer needs more than one piece take the container (`statistics`, `entropy`, by field name, each piece reduced on its device, the parts gathered over `pds.world` and combined in global piece order, `statistics._over_pieces`; `fields_of` checks the name). `split` cuts a structured grid through `carry`. Ghost marks are `DataSet.ghosts` (`dataset/ghosts.py`, `Ghosts`: u8 per cell and per point in VTK's `vtkGhostType` bits, moved by `to_device`, mapped to and from `vtkGhostType` by the VTK interop); every shared cell and point has one owner, the other copies are ghosts, and reductions count only owned values (`owned_values`). Filters carry the marks through `carry` (`ghosts.carry_marks`): an output cell takes its source cell's (`Take`/`Pieces`), a kept point its own, an `Interpolate` point its edge's smallest point's by position (`_before`, lexicographic) and an average point its cell's (`Interpolate.average_cells`, clip's centroids), a cell that is an input face its one cell's or, with two sides, its smallest point's (`_faces_marks`), an edge or a cell with no source cell its smallest point's (`_owned_by_points`). With at least one ghost layer the piece owning a point holds every cell around it, which is why this gives the whole's output (`test_filters_give_each_piece_its_share_of_the_wholes_output`). Pieces with duplicate points and no duplicate cells (no ghost layers) cannot place what is made on edges and faces or kept for a kept cell (`_unplaceable`), so those outputs get `_ghosts_lost`, a reason string `owned_values` names; it is kept by `to_device` and cleared by setting `ghosts`. A filter that builds a `DataSet` outside `carry` must call `keep_marks` or `drop_marks`. Hidden cells are never processed: `_keep_cells` clears them (`without_hidden`), the other new-dataset filters start from `visible(data, *field_arguments)`, and chosen face or edge ids and reused locators are refused (`refuse_hidden`). `split(..., ghost_layers=n)` adds layers (`_block_ghosts`); `ghost_cells(pds)` (`algorithms/ghost_cells.py`) adds one to unstructured pieces through `tack.communication`, on the host apart from the exchanges: boundary points (`faces().boundary()`) keyed by a hash of their exact positions (`_position_keys`), `sort_by_key` over the blocks, matched there (`_pairs`), each point's identity `(owner << 40) | index` with the lowest piece as owner (`_owners`; VTK's rule too, `test_ghost_cells_are_vtks`), cells around shared points sent to each neighbour (`_ghost_parts`) and appended, points deduplicated by identity (`_appended`); it keeps two `Plan`s (`_ghost_plans`, points from their owners, cells from their senders) that `update_ghosts(pds, *fields)` runs. `data.global_ids` (`dataset/global_ids.py`, `GlobalIds`: i64 per point and per cell) is carried like the marks (`carry_ids` in `carry` for `Same`/`Take` maps, not `Pieces` or faces/edges as cells; `keep_marks` keeps them) and mapped to VTK's `GlobalIds` attributes; `generate_global_ids(pds)` numbers owned entities by `exclusive_scan` and brings copies' ids from owners by a `Plan` (ghost_cells' kept plans, or matching: points by position, all points when any piece has duplicates, boundary points otherwise; cells by their points' ids), equal to `vtkGenerateGlobalIds` on pieces without ghost cells, keeping the marks' owners on ghosted ones. The matching (`dataset/_matching.py`: `match`, `owners`, `plan`, `from_owners`, identities `(owner << 40) | index`) is shared with `ghost_cells`, which carries global ids and matches by them when every piece has them. `StructuredTopology(..., offset=)` is the grid index of its first point in a larger grid (VTK's extent's start; `split` and `vtk_to_dataset` set it), and `tetrahedralize` alternates by the whole grid's parity with it and ranks other cells' corners by global id when there are global ids, so pieces split shared faces alike (`test_conforming_pieces.py`). `PartitionedDataSet(partitions, world=None)` holds this rank's pieces and its `World`; `remove_ghosts` drops duplicate and hidden cells but keeps owned points (`keep_points=` of `_keep_cells`), structured blocks staying structured. No mapping helpers or parallel field types: Tack is a functionality layer, not a convenience one. See `test_partitioned.py` and the vault's partitioned-data proposal for the later phases (global ids, ghost generation, MPI).

`tack.communication` (tack-core, `communication/`) is the layer algorithms over pieces are written on, a subset of DIY with Tack arrays as messages (the vault's communication-layer proposal). `Blocks(devices, world)` holds this rank's blocks, global ids contiguous by rank (`_starts`, from the gathered block counts; `rank_of`, `device`). `exchange` takes `{source: {destination: {name: Array}}}` and returns `{destination: {source: message}}` for every local block, arrays on the destination's device (the sent arrays where its device addresses their memory). `all_gather`, `all_reduce` (in global-id order, so float sums have the same bits however blocks are assigned) and `exclusive_scan` take small host values; `link_by_bounds` finds neighbours by boxes and a margin. A `Plan` keeps sends and receives (index arrays per pair, checked across ranks when made) and `run` gathers, exchanges and scatters (`_kernels.py`: `gather`, `scatter`, `take`, `search_right`, kernels over scalar views, whole elements). `sort_by_key(blocks, keys, values)` is a sample sort (local `argsort`, samples weighted by count, `count - 1` splitters every block finds alike, ranges by `search_right`, exchange, sort again): every copy of a key lands on one block, stable by block then index. The transport is the `World`: `_allgather(value)` and `_exchange(messages, rank_of, device_of)`. `World()` is one process; `World(comm)` takes a duck-typed mpi4py communicator (Tack imports no MPI module) and communicates on `comm.Dup()`: headers (name, element type, shape, NumPy shape per array) go by `alltoall`, payloads by `Isend`/`Irecv` under one tag (`_TAG`), relying on MPI's order between two ranks, and every request is waited for before `_exchange` returns. Payloads are one host buffer per rank pair (`_staged`), or with `gpu_aware=True` each array itself through DLPack, received into `tack.empty` on the destination's device. The cases in `tests/_communication_cases.py` (`(ranks, body, check)`) run on threads sharing a hub in `test_communication.py` (`_ThreadWorld`) and under `mpiexec -n 3` through `_communication_mpi.py`, staged and GPU-aware; a new transport must pass them. `test_ghost_update.py` refreshes ghost points over `split`'s pieces with a plan built from gathered positions, in one process and on 2 and 3 MPI ranks (`_ghost_update_mpi.py`). MPI tests take the root conftest's `mpiexec` fixture, which skips without mpi4py (`--extra mpi`, built against the system MPI) or `mpiexec`, unless `TACK_REQUIRE_MPI` is set, as CI's `mpi` job (Open MPI from apt) sets it.

An `Array` records its backend (`_device`; `a.device`, `a.to_device()`, `device=` on every constructor, `*_like` default to the prototype's); copies, astype, reshape, reductions, fill, concat and textures use it. `Kernel.__call__` asks `dispatch.launch_backend(name, args)`: the default backend if it can address every array (equal `Backend.memory_domain`, `(name, ordinal)`, CUDA and Level Zero adding their context; two CPU backends with different thread counts share one) or there are none, else the arrays' own; two domains raise `ValueError`. A device class's arrays choose only when no plain array does; `check_devices` verifies expanded template arrays on the slow path (`_resolve_variant`) and the fast path (`_fast_resolve`, when the plan expands templates). `tack.inspect` uses `launch_backend` too. Library functions that allocate (`tack.algorithms` scans, sorts, stats) are wrapped in `on_inputs_device`, which makes the first array argument's device the default for the call. `tack.sync()` and `sync_async` wait for every live backend (`_live`, `live_backends()`); `from_dlpack` without `device=` takes the default device if it can wrap the tensor, else the tensor's own (`_tensor_device`). An array holds its backend, so a CUDA context outlives every array in it. See `test_devices.py`, which simulates a second device with a CPU backend of another memory domain.

`supports_workgroups` is false on CPU and true on GPU backends. CPU rejects
shared allocations, barriers, thread IDs and block reductions before variant
construction; public inspection and direct LLVM generation also reject them.
`lang/workgroup_support.py` discovers nested/inlined requirements, memoized
only on immutable frontend templates. Direct LLVM generation scans mutable
IR afresh. GPU targets bypass this rejection check. Local arrays, ordinary
atomics and host array reductions remain supported on CPU. This flag does
not itself prove participation or atomic type/scope support. GPU collectives
also require full 256-lane groups and conservative uniform participation
(`workgroup_participation.py`). Check variants before optimization/packing;
cached launches recheck counts without structural analysis. Inspection and
direct GPU generators validate participation too. Scalar-pack params carry
`_is_scalar_pack` to preserve uniformity. Metal pipeline and Level Zero
X/total device limits cannot silently shrink collective groups. MSL uses
structural feature discovery, including thread IDs in conditions. Explicit
user barriers fence shared and global array memory within the workgroup.
See `test_workgroup_contract.py`, `test_workgroup_participation.py` and the
language contract.

`Backend.supported_atomic_dtypes` declares add/min/max targets: CPU all
shipped scalar widths; CUDA/HIP i32/u32/i64/u64/f32/f64; Metal/Level Zero
only i32/u32/f32. `lang/atomic_support.py` checks global array targets,
including uniquely traced inlined pointer copies, before optimization and
in all inspection modes. Direct generators scan mutable IR afresh. Reject
private/shared arrays, textures, unsupported widths and ambiguous targets.
Atomic arrays require natural alignment; `DeviceBuffer.address` exposes
current storage for the cold/cached checks. Variants retain target parameter
indices/alignments, not addresses; warm dispatch does not scan IR. CPU
floating extrema use real CAS loops; unsigned extrema use unsigned RMW.
CUDA/HIP wide operations use 64-bit CAS. OpenCL user atomics and array
reductions use explicit relaxed device scope, rather than legacy atomics'
workgroup-only guarantee. Values convert once to target precision and are
captured outside CAS retries. Atomics are relaxed, spanning CPU workers or
GPU workgroups on one device; they are not publication fences or host/device
system-scope operations. Floating extrema support finite values and numeric
zeros, with unspecified result sign on a zero tie; NaNs/infinities remain
outside their domain.
See `test_atomic_contract.py` and the language contract.

Anything derivable is derived — `supports_f64` comes from `supported_dtypes`, so the two cannot disagree. Level Zero sets `supported_dtypes` in `__init__` because f64 depends on the device.

### Kernel execution flow

All five backends share one entry point: `resolve_variant()` in `runtime/kernel_utils.py`. It turns a call into a compiled variant, running the IR pass pipeline **only when that variant is new**.

Every dispatch:
1. `Kernel.__call__` → `dispatch.launch_backend(name, args)` picks the backend (see Devices) → `backend.execute(kernel, args)` → `resolve_variant(...)`, which first tries the dispatch fast path (below; skipped with keyword arguments or in debug mode); a miss runs `_resolve_variant` inside a `template_scope`:
2. Detect template arguments (`@tack.device_class` classes) and expand them
3. Detect vector arrays and set up scalarization metadata
4. `kernel.get_ir(..., debug=...)` returns the **pristine IR template** for this specialization (lowered, or read from the disk cache); in debug mode append the two hidden arguments (`debug_bounds.hidden_args`); check required workgroup support against the backend
5. Type inference (`infer_param_types`) — annotates params from actual args, sets `_is_array`
6. Build the variant key and look it up (see below): in the backend's cache, then on disk (`_stored_variant`, see Disk cache)
7. Check the call's storage against what the variant recorded: atomic target alignment, and no read-only array (`array_from_ptr`'s default, a read-only DLPack import) bound where the kernel may store (`written_arrays`, from `written_array_params`), counting the store on every other (`DeviceBuffer.version`, also counted by host writes in `Array._check_writable`; `tack.arrays._revision` turns it into a token, which a `CellLocator` records); record the call's signature for the fast path
8. Resolve the loop range from the variant's IR; dispatch (CPU decides serial vs threads, GPU queues a grid launch and returns)

Only on a cache miss (memory and disk):
1. Deep-copy the template with `clone_ir()` — the passes below mutate IR in place and must not touch the template
2. Dimension size resolution (`ir_resolve.py`)
3. Dispatch-time type checking (`check_dispatch_types`) — validates array dtypes against the backend
4. IR optimization: conservative copy propagation (`ir_optimize.py`)
5. Backend `build` callback: scalar packing (GPU), type annotation (`ir_type_annotate.py`, which ends with debug mode's `ir_bounds.guard_accesses`), codegen, compile
6. `_store_variant` writes the variant (post-pass IR and the backend's `variant_artifact`) to disk, if the backend has a `cache_target()`

### The dispatch fast path

`resolve_variant` first tries a per-backend, per-kernel map from the call's *signature* (`_call_signature` in `runtime/kernel_utils.py`) to the variants it resolved before (`_fast_resolve`, outside the template scope); a miss runs `_resolve_variant`, which records the entry. The signature is read straight off the arguments: an array's dtype, shape and element layout; a scalar's kind and integer width; a template object's class (by token), its class constants' current values, and per attribute what `_template_key` and the expansion read, nested templates by path. Equal signatures imply an equal variant key, so a hit expands the templates through a precomputed plan (`_expansion_plan`), decides disjointness, runs the per-call checks (`_check_call`: workgroup launch, atomic alignment, writable arrays) and returns. An entry is `(expansion plan, {disjoint: variant}, array slots)`: it holds the variants it resolved, and `drop_variants` (which a collected template class's token triggers) drops the kernel's entries with them, so a variant it removed is resolved afresh; the map is capped at `_FAST_CAP` signatures. The launch range is evaluated through a plan found once per IR (`_launch_plan`). Debug mode and keyword arguments never use or record entries: debug variants take hidden arguments the fast path does not append (`test_switching_modes_reuses_no_variant_of_the_other`). See `test_dispatch_fast_path.py`, and `test_dispatch_signature_property.py`, which checks the invariant itself: calls drawn from small pools of arguments and device-class attributes of every kind, each answered by the fast path, must get the variant and arguments full resolution gives. A new kind of argument or attribute belongs in its pools.

What a warm call pays (`_fast_resolve`): the call's signature (`_call_signature`, cached per `Array` as `_sig`; per template class the attribute layout `_template_layouts`, relearned when an instance's attribute names or types differ), one weak lookup for the kernel's entry, the disjointness check over the entry's precomputed array slots (`_disjoint_by_slots`, one sorted walk), and the per-call checks. It runs outside `template_scope`, which the slow path enters. `NumpyBuffer.address` and `span` are computed once per array. Keep anything new on this path at that cost; the measurements are in `docs/design/specialization-and-caching.md` ("What a warm call costs").

### Variant cache key

Keyed per `Kernel` (weakly, so compiled code is released with the kernel), then by:
argument type signature + array/scalar/texture categories + vector widths + texture extents + template structure/constants + **`shape_signature`** + (CPU only) whether the call's arrays are disjoint. Template keys preserve actual class identity through a `_ClassToken` rather than the class object, so a `@tack.device_class` class that is garbage-collected takes its IR and compiled variants with it on every backend (a class defined per call still compiles per call, but no longer accumulates). Template keys also preserve typed constants, array metadata, and runtime scalar attribute names; runtime scalar values do not specialize.

That last one matters for correctness, not speed. `ir_resolve` substitutes dimension sizes as literals — `a[i, j]` linearizes to `i * dim1 + j` with `dim1` baked in — so the row stride is part of the compiled code's identity. `shape_signature()` reports exactly the dimensions a kernel bakes in (memoized per IR; empty for 1-D kernels, so varying a flat length does **not** re-specialize).

Because the passes mutate IR in place, the template from `get_ir()` must be treated as immutable — a pass that consumed its `IRDimSize` nodes would leave nothing for a later shape to resolve.

Arrays may share storage, including distinct views and imported pointers. Preserve program order within each race-free iteration; array parameters must not carry unconditional `noalias`/`restrict` promises.

The CPU backend makes that promise *conditionally*. On every dispatch `arrays_disjoint()` (`runtime/kernel_utils.py`) compares the byte ranges of the array arguments: the call is disjoint when no array the kernel stores to (`written_array_params`, memoized per IR) overlaps any other array. Read-only arrays may overlap each other. The answer is the last element of the variant key, so a kernel holds at most two CPU variants per specialization: one compiled with `noalias` on its array pointers, used only for calls that passed the check, and one without. The check is based on storage ranges, never object identity, and costs about 1 µs per dispatch. `cpu._SPECIALIZE_DISJOINT = False` turns it off. GPU backends do not specialize: removing `__restrict__` cost nothing measurable on CUDA.

Metal array pointers are members of one argument buffer, rather than separate
device-buffer kernel arguments (which implicitly promise disjoint storage in
MSL). Members use the packed parameter positions as `[[id(N)]]` indices;
textures retain their separate binding namespace. Each cached dispatch refreshes
the buffer references and declares indirect-resource residency with
`useResource`. Launches are queued, not waited for (`MetalBackend`
`_open_batch`/`_launched`/`synchronize`): each encodes a compute pass into
the open command buffer, committed every 64 launches with at most 4 in
flight, and owns what it reads -- a fresh argument buffer, its packed
scalars as `_Uniforms`, references to its arrays until it completes. The
host waits only when it touches array memory (every `MetalBuffer` method,
reductions, texture uploads) or calls `tack.sync()`; `tack.zeros` takes a
new buffer as Metal cleared it (`starts_zeroed`), without waiting. Bindings
are resolved before an encoder opens, since committing an open encoder aborts. A
buffer's host access waits for every live Metal backend (`_backends`,
`_synchronize_all`), not the one it was made on, in case more than one
exists (a backend constructed directly, as tests do). Memory
handed outside Tack (DLPack, `export_memory`, `wrap_ptr`) is `_shared`: a
launch using it completes before returning. Waiting after every launch cost
about 0.3 ms each. `test_metal.py` covers order across batches, per-launch
scalars and host writes. This fixes the four overlap cases confirmed at `922b642` and
`e265e7f`, without alias-based specialization or disabling vendor optimization.

Metal objects from `new...` methods are owned by the caller, but pyobjc-framework-Metal's metadata says otherwise, so `metal.py` registers the ownership of every `new...` selector it calls (`_OWNED_RESULTS`) before any is called. Add any new one there, or it leaks (`test_arrays_own_their_buffers_alone`). See `docs/design/vendor-workarounds.md`.

A Metal kernel with a sequential loop that stores to an array (or runs an atomic) has its body in `__tack_body__`, a `noinline` function the entry calls (`_stores_inside_sequential_loop` in `msl_gen.py`); workgroup arrays are declared in the entry and passed as pointers. It works around a miscompilation in Apple's GPU back end. Do not merge the two functions or drop `noinline`; loop-free kernels keep one function. See `test_array_updates_in_loops.py`, `docs/design/vendor-workarounds.md` and `docs/design/memory-and-aliasing.md`.

Dispatching one variant from several Python threads: CPU binds arguments
per call and shares no launch state. CUDA and HIP variants share their
scalar pack buffers, so they hold `KernelVariant.dispatch_lock` from the
pack update through the launch (on CUDA and HIP both are queued on the
stream in that order, so a later call's pack update cannot reach an earlier
queued launch). Metal launches own their
argument buffer and scalars, and encode under the backend's batch lock.
Level Zero holds one backend `_launch_lock` (reentrant) over
launches, reductions, copies and `tack.sync()` (a read appends its copy
and a marker under it and waits outside it), because its queue and its kernels'
argument state are backend-wide; a kernel's arguments are captured when
its launch is appended, so the next call may set them at once. Unlocked, half of the CUDA dispatches from four threads
computed with another thread's scalars (`test_concurrent_dispatch.py`).
CUDA's context is the device's primary context (or an embedding application's) and, while it is Tack's only one, is
current only on the thread that made the device (its first `tack.device("cuda")`);
another thread must make it current first, or its dispatch fails with
`CUDA_ERROR_INVALID_CONTEXT`. Once Tack switches contexts (see Devices), every
operation makes its own current.

### Disk cache

`runtime/disk_cache.py` (public `tack.cache`, `python -m tack.cache`)
stores lowered IR (`.ir`, in `Kernel._lower`) and compiled variants
(`.variant`, in `_resolve_variant` on an in-memory miss) under
`TACK_CACHE_DIR/v1-<tack fingerprint>/`. Keys are by content, never
identity: kernel source, closure fingerprints, specialization, template
content (MRO names, `@tack.func` sources, constants), Tack's fingerprint
(installed `RECORD`s or the source tree). `CallBindings.resolve` reports
each resolution to the thread's `_Recorder`; an `.ir` entry is used only if
every recorded name re-resolves to the same fingerprint. Anything lowering
reads from Python must go through `CallBindings` or into the key, or a
stale entry runs. Backends opt into `.variant` with `cache_target()`,
`variant_artifact()` and `load_variant(ir_func, artifact, effective_args)`.
The target must name everything the artifact depends on besides the IR
and argument types (device, compiler and driver versions, compile options,
kept in module constants such as `_NVRTC_OPTIONS` that the compiler and
the target both read), or a stale binary loads. Artifacts (CPU: MCJIT object code; Metal: MSL; CUDA: PTX, loaded with
`cuModuleLoadData`, its target naming the device and compute capability,
the driver that compiles the PTX, NVRTC and its options; HIP: the code
object, loaded with `hipModuleLoadData`, its target naming the device and
`gcnArchName` with its feature flags, the ROCm runtime and driver, hipRTC,
hip-python and the hipRTC options; Level Zero: the
native binary (`zeModuleGetNativeBinary`) and the SPIR-V, loaded with
`zeModuleCreate` as `ZE_MODULE_FORMAT_NATIVE`, from the SPIR-V if the
driver refuses the binary (`native_refused` in the stats), its target
naming the device and its IP version, the driver, ocloc's version, the
ocloc and module options, and f64 support. From SPIR-V, IGC's
finalization is about 50 ms a kernel unless the driver's own
`neo_compiler_cache` has it). A GPU payload's
scalar pack buffers are device memory, so `load_variant` recreates them
from the call's `effective_args`; CPU and Metal ignore that argument. Every
backend now stores variants. Atomic rename writes, checksummed entries, owner-only directories;
under MPI the default mode reads one bundle and writes nothing (a Slurm
batch script is not MPI: Slurm's size is `SLURM_STEP_NUM_TASKS`). Entries
of every version share `TACK_CACHE_SIZE` (1 GiB): a writer prunes by mtime
at most hourly (`.pruned` mark), and write-mode hits refresh mtime. The test
suite runs with `TACK_CACHE=off` (`conftest.py`); `test_disk_cache.py` uses
subprocesses. `TACK_CACHE=write pytest`, run twice, is the end-to-end check.

### Array dimensions

`array.shape[k]` and `len(array)` both lower to `IRDimSize` in lowering (`lang/lowering/arrays.py`), which `ir_resolve.py` folds to a literal wherever it appears — loop bounds, conditions, arithmetic, indices. The dimension index must be a literal (`x.shape[d]` with a runtime `d` raises).

**One exception, and it matters:** the outermost parallel loop's bound is left unresolved. It never reaches generated code — codegen reads the `__loop_end__` parameter — so `_resolve_range_expr` evaluates it per dispatch instead, and `ir_shape_deps` excludes it from the variant key. Without that, `for i in range(x.shape[0])` — the most common line in any kernel — would compile a new variant for every array length. Lowering writes the bound in terms of the arguments (`_launch_bounds_from_arguments` in `lang/lowering/loops.py`): a local bound once before the loop (`n = x.shape[0]; for i in range(n)`, or an inlined `__len__` result) is replaced by its value, an alias of an array parameter by the parameter, and assignments nothing else reads are dropped, since a dimension read outside the bound would specialize. See `test_outside_parallel_loop.py`.

Everywhere else the dimension *is* compiled in, so it specializes:

```python
for i in range(x.shape[0]):        # one variant for all lengths
    out[i] = x[x.shape[0] - 1 - i] # ...but this bakes the length in → one variant per length
```

To avoid that, pass the length as a scalar argument (`def reverse(x, out, n)`) — scalars are runtime parameters and don't specialize.

### Multi-dimensional parallel loops

A top-level `tack.ndindex` of two or three dimensions lowers to one
`IRParallelFor` with `dims` (one index variable per dimension, slowest
first) and `extents` (host-evaluated sizes, like `end`, their product;
not a traversed child, so no pass folds them). Nothing divides: CPU
passes the extents as i64 parameters before `loop_start`/`loop_end`, and
`_emit_parallel_dims` decomposes a chunk's first index once and walks
rows with a carry; Metal dispatches a grid of that shape with a `uint3`
position; CUDA/HIP/OpenCL take `__flat__` and `__ext_k__` after `__n__`
and launch the grid `launch_geometry` picks (block up to 256 along x,
then y, then z), or a flat one with `__flat__` set when the device's
grid limits are exceeded. `ast_transform` flattens workgroup kernels
(`parallel_dims.flatten`); 4+ dimensions and nested `ndindex` keep the
flat form. `_get_launch` returns `(items, extents)` and treats any
non-positive extent as empty. Every pass that knows `var` knows `dims`.
See `test_parallel_dims.py`.

### Textures

`tack.texture3d(array, shape)` copies the array into storage the `Texture3D`
owns, on every backend, and `tex.update()` copies it again; writes to the
array in between do not reach the texture. `Backend.texture_in_hardware()`
picks the storage: a hardware image from `create_texture_image()` (CUDA,
Metal, HIP with image support, Level Zero with samplers, within the device's
3D limit) or a private f32 array sampled in software (CPU and the fallbacks).
Dispatch binds that storage through `bind_textures()`; the grid bound still
reads the source array. No compiled kernel caches texture objects, and none
is keyed by device address: images free their device objects when the
texture is collected. Only f32 arrays of exactly W*H*D elements and
`interp='linear'` are accepted. See `test_texture_snapshot.py`.

### Kernel code inspection

Generated GPU variables and all backend kernel entry names use the shared,
injective encoding in `codegen/identifiers.py`. The GPU generator renames
only a structural IR copy; canonical IR names and dispatch metadata remain
original. Runtime lookup must call `kernel_entry_name` on the original
function name exactly once. Lowering/templates/vector components/scalar
localization/packing allocate generated bindings through `fresh_name`
(`lang/ir_names.py`) so they cannot merge with Python source names.

`tack.inspect(kernel, *args, mode=...)` runs the compilation pipeline and returns the generated code as a string without executing. Modes: `"ir"` (Tack IR), `"source"` (backend code: LLVM IR / MSL / CUDA C / HIP C / OpenCL C), `"optimized"` (post-LLVM-O3 IR; CPU only, other backends raise `ValueError` after the kernel's own checks). It makes dispatch's decisions too: `check_dispatch_types` and the backend's `_store_texture_shapes` (software sampling on HIP/Level Zero devices without texture hardware). Implementation in `lang/inspect_kernel.py`.

### Debug mode (bounds checks)

`tack.set_debug(True)` or `TACK_DEBUG` (read when `tack` is imported) sets
`dispatch._debug`. `Kernel.get_ir(debug=True)` lowers separately (a
`("debug",)` key part): `_visit_subscript_index`/`_linearize_index` and flat
atomic indices go through `_checked`, which wraps the flat index in
`IRBoundsCheck` (per-dimension indices and extents, a site, the errors
array) and records an `ir_bounds.BoundsSite` in `function.bounds_sites`.
Dimension 0 and flat extents are `IRArrayExtent`, which `ir_resolve` turns
into loads from the hidden i64 `__tack_extents__` array (three per
parameter), so nothing specializes on length; other dimensions are
`IRDimSize`, already baked into strides. Arrays use their allocation size.
Two hidden array parameters follow the kernel's (`ir_bounds.HIDDEN_PARAMS`);
`_resolve_variant` appends their arguments (`debug_bounds.hidden_args`,
per-thread arrays) and notes the launch; `Kernel.__call__` (and the result
epilogue) calls `debug_bounds.check()`, which raises `IndexError`. A check
is -1 when out of bounds, after `__tack_in_bounds__` records the first
violation (atomic add on the count). `annotate_types` ends with
`ir_bounds.guard_accesses`: loads become `index >= 0 ? load : 0` (typed
zero, hence after typing), stores and atomics run under `if`. The index is
evaluated twice, which is fine because indices are pure. Every generator emits
`IRBoundsCheck` (`codegen/bounds.py`, `llvm_gen._bounds_helper`); MSL counts
it as a store for `_stores_inside_sequential_loop`. Debug off generates
unchanged code. The disk cache's lowered-IR key includes debug mode
(`disk_cache.lowered_key`), and the variant key is built from that entry,
so debug and plain kernels never share one; a stored debug kernel keeps
its `bounds_sites`. Debug calls bypass the dispatch fast path. See
Debug mode also checks every `tack.dataset` topology's ids when it is made (`topology._checks_ids`, `_csr_problems`: offsets monotone from 0 to their members' length, ids within range), before polygonal and polyline topologies derive facets and ends from them; `topology.validate()` runs the same checks on demand, plus an unstructured topology's types and point counts (`groups()`). See `test_debug_bounds.py`. `TACK_DEBUG=1 pytest` runs every
kernel of the suite with checks, and passes. A test of what plain mode
generates or does (one that reads generated source or IR text, calls a
compiled CPU kernel with its own argument list, compiles generated code with
a reduced toolchain such as OpenCL 1.2 or the host C++ shim without
`atomicAdd`, counts fills or times overlapping launches) takes the
`no_debug` fixture (`conftest.py`), which turns debug mode off for it.

### Lowering (ast_transform.py and lang/lowering/)

`KernelTransformer` (`lang/ast_transform.py`) keeps the `visit_*` walk for statements and expressions and the expression machinery (`visit`, `_visit_expression`, `_capture_value`, `_visit_ordered`, `_source_error`, `_fresh_name`). Everything else is mixins, one module each in `lang/lowering/`, each owning the state it sets up in its `_init_...` method: `inlining` (`Inlining`: device functions, `_NameRenamer`, `_structure_returns`), `vectors` (`VectorValues`: vector and matrix values, components, construction, methods, `@`), `arrays` (`ArrayAccess`: `_visit_subscript_index`, `_linearize_index`, `_element_locations`, vector/matrix/SOA array elements, and the `_vector_arrays`/`_matrix_arrays`/`_soa_arrays` metadata), `loops` (`Loops`: the parallel loop, `range`/`ndindex`, sequential loops, `_launch_bounds_from_arguments`), `checks` (`LoweringChecks`: effects outside the loop, single values, unbound names, loop variables), `local_arrays` (`LocalArrays`: `tack.shared`/`local_array` and their `_like` forms), `bounds` (`BoundsChecks`: `_checked`), and `values` (`VectorValue`, `TupleValue`, shared helpers and name tables). A mixin reaches another's state and methods through `self`; its docstring says what it owns and reads. The split changed no lowered IR, lowering error or generated source over the whole suite: when refactoring lowering, hash every `transform_kernel` result (IR dump or error) and every generated source over the suite before and after, and compare.

### IR structure (lang/ir.py)

The IR is a simple tree of nodes:

**Loops**: `IRParallelFor` (outermost, maps to thread parallelism), `IRSequentialFor` (inner loops, supports `step`), `IRWhile`, `IRBreak`, `IRContinue`

**Control flow**: `IRIf`, `IRIfExp` (ternary)

**Array access**: `IRArrayLoad`, `IRArrayStore`, `IRAtomicOp` (atomic_add/min/max)

**GPU primitives**: `IRSharedAlloc` (threadgroup memory), `IRBarrier` (sync), `IRThreadId` (local thread index)

**Expressions**: `IRBinOp`, `IRUnaryOp`, `IRCompare`, `IRBoolOp`, `IRCall` (math builtins), `IRCast`, `IRConstant`, `IRName`, `IRAttribute`, `IRDimSize`

**Debug mode only**: `IRBoundsCheck` (an access's flat index, or -1 out of bounds), `IRArrayExtent` (a parameter's extent, resolved to a load from the hidden extents array)

**Other**: `IRAssign`, `IRReturn`, `IRPrint` (kernel debugging)

Metadata that passes set on nodes (`IRParam._is_array`, `_is_texture`, `_texture_shape`, `_is_scalar_pack`; `IRFunction.disjoint_arrays`, `bounds_sites`, `held_arrays`; `IRAssign._resolved_type`; `IRCompare._operand_type`; `IRDimSize.index_count`) is declared on its class in `ir.py`, with the value it has until set, and read directly, never through `getattr` with a default of the reader's own (`_is_array` had three). `_is_array` is None until type inference; code generators given hand-built IR treat None as an array.

### IR passes

- **ir_resolve.py**: Replaces `IRDimSize` nodes with concrete constants from array shapes, resolves `IRAtomicOp` sub-expressions, resolves `shared_like` dtypes from arrays, and (debug mode) turns `IRArrayExtent` into loads from `__tack_extents__`
- **ir_bounds.py** (debug mode): `guard_accesses`, run at the end of `annotate_types`, skips each access whose checked index is negative (a load gives a typed zero)
- **ir_optimize.py**: Conservative copy propagation for inlined arguments. Assignment counts are computed once per kernel and reused in nested blocks: the copy target must have one binding and its source must have no assignments, loop bindings, or allocations anywhere in the kernel. Each statement is tested after earlier copies are applied, so a chain resolves to its root in one walk; nested device functions pass an array down as such a chain, and an unresolved link is a local holding an array. This leaves some block-local copies to LLVM/vendor optimization and avoids repeated subtree counting during cold compilation. Custom LICM and CSE remain disabled because they lack memory/control-flow safety analysis.
- **type_inference.py**: Annotates IR params with types from actual arguments. Arrays get `_is_array=True`, scalars get `_is_array=False`. Float scalars auto-promote to `f64` when any array arg uses `f64`; otherwise default to `f32`. Int scalars exceeding i32 range auto-promote to `i64`. `check_dispatch_types()` validates array dtypes against backend capabilities.
- **ir_type_annotate.py**: Sets `dtype` (a `ScalarType`) on every expression IR node. Codegens read `node.dtype` directly instead of reimplementing type inference heuristics.
- **ir_traversal.py**: Explicit structural child schema, preorder `walk_ir`, and postorder `transform_ir`. Resolution, scalar packing, copy substitution, and shape-dependency queries share it. Metadata is not traversed; unregistered node kinds fail loudly. `clone_ir` specializes deep copying for registered plain IR nodes and list/dict containers, copying every attribute (including metadata) with one identity memo. It preserves shared references, cycles and ScalarType identity; other metadata retains Python's deepcopy protocol. Variant preparation, scalar localization, GPU packing and inspection use it; cache hits do not clone IR. Keep all verifier boundaries. Compare copying costs with `uv run --no-sync python benchmarks/ir_clone.py --output /tmp/ir-clone.json`; this is a CPU component benchmark, not an end-to-end latency measurement.
- **ir_verify.py**: Checks lowered templates before caching and variant IR after resolve, infer, scalar localization, optimize, GPU packing, and annotation. Errors report kernel, stage, node kind, and tree path. Checks cover structure, bindings, loop targets, unresolved dimensions/allocation types/texture extents, parameter categories, and scalar annotations. Host-evaluated grid ends retain dimension queries; array pointers need no scalar dtype. Cache hits do not run the verifier. This is not definite-assignment, bounds, race, or barrier-uniformity analysis. Direct pass/codegen calls on IR fragments must request verification themselves.

### Scalar kernel arguments

Kernels accept both arrays and Python scalars (int, float) directly. The `_is_array` attribute on IR params controls codegen: arrays become pointers, scalars become values. All four backends handle this distinction in their codegen and dispatch paths.

### Vector scalarization

`tack.empty(shape, tack.vec(n, dtype))` creates an array of `shape` vector elements over a flat scalar buffer of `prod(shape) * n` components (`_vector_n`, `_logical_shape`); its `shape` counts elements and `numpy()` returns `(*shape, n)`. In kernels, `array[i]` expands to n component loads/stores. Vector operations (add, dot, cross, normalize) are scalarized at the IR level.

Structure of arrays: `tack.soa(x, y, z)` (or `layout="soa"` on a constructor) is a `SoaArray`, an `Array` with no buffer of its own whose `components` are scalar arrays of the logical shape; it sets `_vector_n`/`_logical_shape`/`_matrix_shape` as an interleaved one does, and `layout` (`"aos"` on `Array`) is in the call signature and the template keys. Backends never see one: `_flatten_soa` (in `_expand_template_args`, after template expansion, and in the fast path's plan) replaces it in place by its components, and lowering, told by `vector_arrays[name] == ("soa", width)`, replaces the parameter in place by one per component (`soa_param_names`: `__soa{c}__{name}`). `_element_locations` gives each component's `(array, index)`, `element * n + c` of the array or `element` of the component parameter, for loads, stores, component stores, augmented stores and vector atomics; `_shape_name` sends shape queries, linearization strides and debug extents to the first component. The SOA name itself is never an IR value (`visit_Name` refuses it); an inlined device function given one takes the caller's component parameters with no binding. A scalar atomic at a flat component index is refused. Host methods (`numpy`, `_upload`, `fill`, reductions, `__getitem__`, `copy`, `astype`, `reshape`) work per component; `_scalar_view`, DLPack and `export_memory` refuse, and `copyto` between layouts runs `_copy_kernel`. `_source_name` reads component parameters back as `component c of ...`. See `test_soa.py`.

Comparisons with a vector operand lower per component to a mask
(`_vector_compare`; chains evaluate every link), `and`/`or` over masks
per component without short-circuiting, `any`/`all` to an `IRBoolOp` over
the components, and `tack.select` to one `IRIfExp` per component.
`_scalar_condition` rejects a mask as an `if`/`while` condition with the
remedy. See `test_vector_comparisons.py`.
A vector value is `tack.vec2/vec3/vec4(...)` (`_visit_vector_construct`):
N scalars, one scalar for every component, or one N-vector; on the host the
same names are element dtypes. A list of scalars in kernel source is also a
vector, of any width (`visit_List`); the validator rejects an empty or
nested list, a list target and a loop over a list.
In lowering a vector value is a `VectorValue` (`lang/lowering/values.py`), a list of
component expressions with a `shape`; a source tuple (indices, multiple
results) is a `TupleValue`. Neither enters the IR: `_check_single_values`
scans each lowered statement once and raises `UnsupportedSyntaxError` at
the statement's position for one left in a scalar position, instead of
the verifier's "expected expr node". Arithmetic, math builtins, casts
and conditional expressions map over components (`_componentwise`
repeats scalars and rejects mixed widths). An assignment lowers to one
assignment per component, so `_settle_components` first evaluates any
component that could observe an earlier one's write: a read of an earlier
target component for a variable, any array load for an array element
(arrays may alias). Components are `v[c]`/`v.x`, on variables and on array
elements (`vf[i][c] = x` stores one component); a runtime index selects
through a chain and stores through per-component guards. A named typed or
integer `ArrayConstant` read at a runtime index is instead an `IRTableLoad`
(`constant_table` tags the `VectorValue`; `codegen/tables.py`, an LLVM
internal global): one constant array, clamped to the last entry as the
chain is. Untyped float constants keep the chain (weak literals). Chains
in a storing loop crashed Metal's compiler service; see
`test_constant_tables.py`. A vector index
supplies one dimension per component, in subscripts and in atomics, whose
index may be a tuple with one entry per dimension; a vector value given
to an atomic on a vector array lowers to one atomic per component, while
a scalar value keeps the index a flat component index. Tuple assignment
accepts subscript and component targets through temporaries, stored at
their own position among the targets (left to right, after the whole
right side). `_linearize_index` tags each `IRDimSize` with the number of
indices given, and `ir_resolve` raises `TypeError` when that differs from
the array's dimension count; a single index is a flat index and is not
checked. A device
function returning several values yields a `TupleValue` whose vector
elements are `VectorValue`s; only tuple unpacking consumes that. `v.min()`/`v.max()`/
`v.sum()` are intercepted before the same-named builtins, which need
arguments. See the vector tests in `test_new_features.py`.

A matrix (`tack.mat2/mat3/mat4(...)` from rows, lists of scalars or
vectors, or N*N row-major scalars; `tack.mat(row, ...)` for any shape;
`tack.identity(n)`, a literal `n`: `_visit_matrix_construct`,
`_visit_matrix_identity`; an element of a `tack.mat(n, m)` array) is a `VectorValue`
with a two-dimensional shape and row-major components, so everything
componentwise applies unchanged. A matrix array is a vector array of
`n * m` components with `_matrix_shape`; the runtime reports it to
lowering as the tuple `(n, m)` in the `vector_arrays` dict, which also
keeps a 2x2 matrix array and a 4-vector array in different IR cache
entries. `_matrix_vars`/`_matrix_arrays` hold shapes beside the component
counts; `_bind_vector` and `_variable_value` keep them in step. `@` is
`_matmul` (operands captured first, since every entry is used several
times); `transpose`/`trace`/`determinant`/`inverse` are in
`_emit_matrix_method`; `m[i, j]` goes through `_matrix_selector`. Limit
4x4 (`MAX_MATRIX_EXTENT`). See `test_matrix.py`.

### @tack.func inlining

Functions decorated with `@tack.func` are inlined at the AST level into kernels. Supports return values, multi-return (tuple), nested inlining, and vector propagation. A function may `return f(x)` where `f` returns a tuple holding vectors, and a name may hold such a tuple (`_tuple_vars`, `_bind_tuple`: one temporary per (name, item), so every branch writes the same ones) until it is unpacked or returned; a tuple of scalars alone still binds as a vector. Variables are renamed with unique suffixes to avoid collisions.

`tack.constant(value, dtype=None)` (`lang/constant.py`) returns an `int` or
`float` subclass. `visit_Name`/`visit_Attribute` resolve a name the
function does not bind through the same `CallBindings`; one bound to a
constant lowers to an `IRConstant`, under an `IRCast` when typed, so it is
baked into the cached IR like a literal and needs no variant-key entry.
Nothing else is captured: other names still raise `NameError`.
`math.pi`/`e`/`tau` lower the same way. A tuple (or tuple of tuples)
makes an `ArrayConstant`, lowered by `_named_constant` to a `VectorValue`
of per-component constants; `visit_Subscript` selects its components.
`tack.math` (`src/tack/math.py`) is plain `@tack.func` definitions with
GLSL semantics. See `test_constant.py`, `test_math_helpers.py`.

`tack.random` (`src/tack/random.py`) is PCG RXS-M-XS over an LCG in u32
with explicit state; draws return `(value, next_state)` and have `np_`
mirrors. Keep `uniform` as the top 24 bits times 2^-24 so it stays
bit-exact everywhere.

Device calls resolve by object identity from the defining callable's globals
and closure bindings (`call_bindings.py`), including aliases and module-qualified
calls. Arguments use caller bindings; nested bodies use callee bindings.
Validation and lowering share the resolver. Runtime callable parameters,
ordinary Python functions, object-property lookup, and recursion are rejected.
Numeric captures remain unsupported. Template rewrite returns a local method
map with marked synthetic calls and retains each method's original Python
callable; no global function registry or register/cleanup sequence remains.
Keep the IR cache construction lock. Cached IR retains its inlined bodies;
rebinding a callable requires recreating the kernel. AST-only transforms may
pass an explicit `bindings` dictionary. See `test_func_bindings.py`.

Original kernels and device functions are validated before lowering; unsupported syntax raises `UnsupportedSyntaxError` with the function and captured-source line/column. Device source is checked before return restructuring, including unreachable statements. `and`, `or`, and conditional expressions short-circuit, including inlined effects. Scalar operands/arguments evaluate left to right; augmented stores evaluate their index once, and sequential `range` bounds/steps are captured at entry. Comparisons and logical expressions yield i32 `0`/`1`. Outer parallel `break`, keyword arguments, and assertions are rejected; a kernel `return` is allowed only as the last statement, with a return annotation: `Kernel._split_result` strips it and builds a one-thread epilogue `Kernel` (statements outside the loop plus a store of the expression into a hidden one-element array, parameter `__tack_result__`) that `__call__` runs after the launch and reads with `result[0]`; barriers are statement-only, while an atomic may be an expression whose value is the element's old value (lowered to a temporary among the pre-statements, so it runs in source order; every generator's atomic yields the old value, the CAS loops through helpers). Reading a name the kernel never binds (typically a module-level Python value) raises `NameError` naming the kernel or inlined device function and the source position; binding a second name to a local or shared array (`view = tmp`) raises `UnsupportedSyntaxError`. `Kernel.__call__` lets `UnsupportedSyntaxError` through unwrapped. See `docs/reference/language-contract.md` and `test_source_validation.py` / `test_differential.py`.

The parallel loop must be a `for` directly in the kernel body, exactly once. Statements outside it run any number of times per launch (per GPU thread; per CPU chunk, probe and timing sample), so they may only bind locals, load arrays and declare arrays. `ast_transform` rejects stores, atomics, barriers, block reductions and `print` there by scanning each statement's lowered IR, which catches effects from inlined functions and template methods at their own source line; `ir_verify` enforces the same invariant on later IR. `_localize_outer_scalars` gives each iteration its own copy of any outer local (or scalar parameter) the loop reassigns. See `test_outside_parallel_loop.py`.

### @tack.device_class templates

Classes decorated with `@tack.device_class` can be passed as template arguments. A kernel defined in the class body binds the instance it is called on (`Kernel.__get__` returns a bound method), so `model.step(dt)` passes `model` as the first argument; template detection is by argument type, so nothing else changes. `template_func_methods` unwraps `@staticmethod` over `@tack.func`. Methods and class constants are collected along the MRO when a kernel is lowered (`template_func_methods`, `classify_template_attrs`), so subclasses inherit and override them. An instance attribute holding a `@tack.func` (`template_func_attrs`) is called as that plain function, with no synthetic parameters, and its identity is part of the template cache key. Class-level scalar attributes become compile-time constants (part of cache key; an instance attribute of the same name holding a number overrides one and stays compiled in, one holding anything else drops it, as a subclass attribute does), instance scalar attributes become runtime kernel parameters (no recompilation on change), array attributes become kernel buffer parameters, and `@tack.func` methods are inlined with `self` resolved. Methods can call sibling methods on `self`. A template (or one it holds) passed to a `@tack.func` or to a template's method -- `helper(obj, i)`, `b.combine(a, i)`, `helper(self)`, `helper(self.base)` -- gets the callee copied with that parameter bound to the template's node, as `self` is (`template_rewrite._Specializer`, used by the kernel rewriter, method resolution and the copies themselves through `_BoundResolver`): the copy is a synthetic call in `resolved_funcs`, keeps the callee's Python function for its bindings, and takes the tree's synthetic parameters (`_Node.extra`); copies are shared by (callee, bound nodes). `rewrite_templates` takes the kernel's `python_func` to resolve free callees. `for c in obj:` over a template is rewritten in `_KernelTemplateRewriter.visit_For` to `range` (one name in the class's `__tack_iterate__`) or to an `ndindex` over its names reversed, binding `c` to the index vector fastest first (`iteration_space`, `test_template_iteration.py`).

Indexable objects (`template_rewrite._Indexing`, run before the kernel and method resolvers): where `obj` is a device-class argument or a template it holds (`root.base`), `obj[i]` becomes `obj.__getitem__(i)`, `obj[i] = v` `obj.__setitem__(i, v)`, `obj[i] += v` a get and a set (index in a fresh local unless a name or constant), `len(obj)` `obj.__len__()`; the calls then resolve as methods. Several indices pass one vector (`ast.List`). A missing method or an atomic on an indexable raises `TypeError` at lowering. An instance tuple of numbers is one runtime scalar per entry (`shape__0`, `_TUPLE_ENTRY`; `scalar_path` maps it back to `("shape", 0)` for the expansion plan, and the fast path's layout has `_LAYOUT_TUPLE`); `obj.shape[k]` needs a literal `k` and `len(obj.shape)` is a constant; a class tuple is compile-time constants the same way (`_class_constants`), an instance tuple replacing it stays compiled in. `len(obj)` as the parallel loop's bound resolves on the host like any launch range. See `test_indexable.py`. `tack.arrays` (moved from `tack.dataset.arrays`) builds on it: `_Implicit` defines `__getitem__` (its `get`) and `__len__` (an instance `length`, set by `_metadata` beside the underscored size, width and dtype a kernel does not receive), so plain kernels index every implicit array; `tack.dataset.views.storage` picks a field's view mixin. Writable implicit arrays (`View`, `Strided`, `Permutation`, `Concatenate`, `ExtractComponent`, `Components`, `_Explicit`) define `__setitem__` in terms of their operands' stores, so writability composes; computed arrays define none. Inside a method a missing protocol method is not an error until used (every method of an object is resolved): `_Indexing(lazy=True)` emits `__tack_refuse__("message")`, which `ast_transform.visit_Call` raises as `TypeError` only when that method is inlined (the source validator lets its string through). `check_writable_arrays` names an array a device class holds by its source path, which the rewrite records on the funcdef and lowering keeps on the IR (`IRFunction.held_arrays`), so a kernel lowered from the disk cache names it too. See `test_arrays_protocol.py`.

A template may hold templates as attributes, to any depth (`self.base.get(k)`, `obj.base.step` in a kernel). `template_tree` lists the tree (cycles refused); `classify_template_attrs` flattens it, nested names joined by `__` (`base__step`), so expansion and vector detection are unchanged; `rewrite_templates` resolves chains through `_Node`s, every method of the tree taking the tree's synthetic parameters; a chain stops at the first non-template attribute (`self.data.shape[0]`). Nested paths and classes are in the cache key (`template_structure`) and their class tokens retire variants. `resolve_variant` runs in a `template_scope` that memoizes each object's tree, classification and key part for the call, rather than walking a tree per use (20 µs a level per dispatch before, about 8 after). Composed code inlines to what one would write by hand (identical optimized LLVM IR). See `test_nested_templates.py`.

### Integer floor division and remainder

Integer `//` rounds down and `%` is zero or follows the divisor's sign,
matching Python within the defined fixed-width domain. Both operands are
converted to the annotated promoted type before the operation; unsigned
types use unsigned division. LLVM applies an integer correction to signed
truncating division/remainder. The CUDA/HIP, Metal, and OpenCL generators
share typed helpers in `codegen/integer_division.py`, evaluating operands
once. Do not replace these with floating-point division or rely on C's
implicit signed/unsigned promotions. Evaluated divisors must be nonzero;
signed minimum with divisor -1 is excluded for both operators. See
`test_integer_division.py` and the language contract.

Integer `+`, `-`, `*`, unary negation, bitwise operations, and left shifts
wrap at their annotated width; signed right shifts are arithmetic and
unsigned right shifts logical. Counts must lie in `[0, width)`. Integer
casts/stores wrap modulo the destination width, with source signedness
controlling extension. Promotion preserves both complete operand ranges;
any signed/u64 pair requires an explicit cast. Comparisons and integer
`abs`/`min`/`max` retain exact integer semantics. `codegen/integer_ops.py`
shares unsigned-carrier helpers across GPU generators; Metal uses `as_type`
for signed bits. Three vendor workarounds live here (Metal's noinline
i64/u64-add helpers, OpenCL's noinline narrow `neg`/`abs`, CUDA's inline-PTX
narrow negation, `_ptx_negation`); see `docs/design/vendor-workarounds.md`
before touching them.
LLVM tracks signedness on every
annotated integer expression, including loads of locals and scalar arguments.
Literals/scalars choose i32/i64/u64 by magnitude and reject unrepresentable
integers. Float-to-int inputs must be finite with a representable truncated
value. See `test_integer_semantics.py` and the language contract.

Integer `/` converts operands independently to f32 and returns f32, rather
than truncating; explicit floating casts request f64 on capable backends.
The output array does not widen the intermediate precision. The evaluated
integer divisor must be nonzero. With a floating operand, division uses
the promoted floating precision. Integer `**` and two-argument `pow`
preserve the base type, with an independently typed nonnegative exponent,
and compute exact power modulo the base width. Signed/u64 pairs are valid
for `/` and power. Negative literal integer exponents are rejected; dynamic
negative ones are outside the caller domain. `0 ** 0` is 1. Cast the base
to float for negative powers. GPU generators share exponentiation-by-squaring
helpers in `integer_ops.py`; LLVM emits an internal helper with wrapping
products and a logical count shift. With a floating operand, both power
arguments use the promoted floating precision. See `test_division_and_power.py`.

Floating `//` and `%` convert to f64 if either operand is f64, otherwise
f32, and return that floating type. Shared GPU helpers in
`codegen/float_division.py` and typed LLVM helpers use a truncating
remainder with divisor-sign correction and reconstruct/snap the quotient
instead of directly flooring rounded division. Zero remainder follows the
divisor's sign; zero quotient follows true division's sign. NaNs and
infinite dividends return NaNs; finite/infinite pairs follow Python-style
sign correction. Evaluated divisors must be nonzero, and denormal support
remains outside this portable increment. CUDA omits `--use_fast_math`
and explicitly selects non-flushing, precise division/sqrt and permitted
multiply/add contraction. Metal disables `fastMathEnabled` for every
kernel, including runtime reduction sources on both backends.
CPU floating negation uses LLVM `fneg`; floating `!=` and truth tests use
unordered inequality so NaN guards and signed-zero expressions agree.
See `test_float_division.py` and the language contract.

The floating-point baseline requires annotated precision, exceptional
classes/signs and expression grouping. Adjacent multiply/add contraction
is permitted, inside one expression only: a named value is rounded once
and every read agrees (`x - x` is 0). General reassociation is not
permitted. CPU emits no fast-math flags; HIP/OpenCL retain default settings
without unsafe options, except that HIP passes `-ffp-contract=on`
(`_HIPRTC_OPTIONS`; a vendor workaround, see `docs/design/vendor-workarounds.md`),
since a named value must be rounded once for every read. NVRTC has no such
option and fuses a product into a later statement's sum, so `CUDACodeGen`
emits a floating product that is not an operand of `+`/`-` as
`__fmul_rn`/`__dmul_rn` (`_rounded_products`, off for HIP and OpenCL;
another vendor workaround); host shims compiling generated CUDA define the
two (`test_float_semantics.py`'s later-sum test). Floating math
calls convert arguments to the result precision first; integer arguments
default to f32, irrespective of the destination array. CPU libm calls use
the matching f32/f64 symbols and return that precision before nested
arithmetic. Scalar floating min/max prefer a number to NaN and permit
either zero sign on ties. Denormals, NaN payloads/signaling, exception
flags/traps, cross-backend bitwise agreement and global math error bounds
are outside the portable baseline. CPU callers must retain the default
round-to-nearest, untrapped environment. `test_float_semantics.py` checks
classes/signs, grouping, permitted contraction and bounded-domain math.
Reduction order/accuracy guarantees remain the next stage-five task.

Float literals are weak, as Python scalars are under NumPy's NEP 50.
`ir_type_annotate.py` marks literal expressions (literals under unary
+/-, arithmetic, math builtins and conditional arms); one containing a
float literal is weak and f32 until it meets a non-weak floating operand,
an explicit floating cast, or a floating store/atomic/local target. It
then takes that type: each literal converts once from its Python value
and its operations run at that precision, so `x_f64 * 0.1` and
`tack.f64(0.1)` are exact. Literal-only subtrees are not folded. Alone,
or beside integers only, literals stay f32; `i * 0.1` is a non-weak f32.
A local assigned exactly once, to a weak literal expression, is replaced
by that expression at each read (`_inline_weak_literal_locals` in
`ir_optimize.py`, before copy propagation), so it stays weak; a device
function's literal results reach the caller that way. A conditional counts
only with a constant condition. A local assigned more than once is the
join of its assignments; if they are all weak literal expressions
(`_literal_only_locals`, any condition), it takes the kernel's float
context instead, f64 when an array parameter is f64, as scalar arguments
do (`annotate_types`).
LLVM emits a `double` constant and CUDA/HIP/OpenCL an unsuffixed `repr`
when the constant's dtype is f64; f32 code is unchanged. See
`test_float_literals.py` and the language contract.

### 64-bit loop indices on GPU

GPU backends use 64-bit integers for loop variables and index arithmetic (`long` on Metal, `long long` on CUDA/HIP) to support grids with more than 2^31 elements. CUDA/HIP widen `blockIdx.x` before multiplying by `blockDim.x`: the built-ins are 32-bit unsigned, and the product wrapped silently at 2^32 threads. OpenCL builds the index from `get_group_id(0)` the same way, since Intel's `get_global_id(0)` wraps at 2^32 despite its `size_t` type; native reductions use the same index. The CPU backend already used i64 via LLVM. Metal's `thread_position_in_grid` attribute is limited to `uint`, so max single dispatch is 2^32 threads. The `int()` cast in kernel code remains 32-bit (user semantics).

Each GPU backend's `execute` and native `reduce_array` call `check_launch_size` (`runtime/kernel_utils.py`), which raises `ValueError` for more iterations than one grid can index, instead of a driver error or a wrap. `_max_launch` is queried at init: CUDA max `gridDim.x` × 256; HIP the same, capped at 2^32 − 256 (AMD's dispatch packet counts work-items in 32 bits); Level Zero `maxGroupCountX` × workgroup size (the group count is a ctypes uint32 that would truncate). Metal uses `_MAX_LAUNCH = 2**32`, and reduces arrays of 2^32+ elements through NumPy because its reduction kernel's count and thread position are 32-bit. CUDA/HIP reduction kernels take the count as a `long long` argument. See `test_launch_limits.py`.

### Algorithms (tack.algorithms)

`exclusive_scan` and `inclusive_scan` (`algorithms/scan.py`) reduce, then scan, by chunks of `_CHUNK` (256), the chunk totals scanned recursively; a level of at most `_SERIAL_MAX` (4096) elements on the CPU, or one tile on GPU backends, is one launch one thread scans in order (`_scan_serial`). On backends with workgroups a chunk is a 2048-element tile scanned by a workgroup (`_tile_sums`/`_tile_apply`). Sums form in the output's dtype; in place (`input is output`) works; the total is read as one element.

`argsort`/`sort_by_key` (`algorithms/sort.py`) are a stable LSD radix sort over 8-bit digits for i32/u32/i64/u64 keys (mapped to u64, sign bit flipped). On the CPU: a private histogram per 256-element chunk, the exclusive scan assigns slots, and a second kernel scatters each chunk in order. With workgroups: 4096-key tiles sorted in shared memory (`_sort_tiles`, eight stable one-bit splits, no shared atomics, u8 digits and u16 positions to fit Metal's 32 KB), then runs scattered (`_scatter_tiles`). The pass count follows the key spread (`_key_range`). No atomics anywhere. `unique`/`reduce_by_key` flag run starts, scan them, and reduce one thread per run serially, so results are reproducible. Empty arrays are valid outputs. The measurements behind these shapes are in `docs/design/parallel-execution.md` ("Library algorithms").

### CPU threading decision

The CPU backend fans a loop range out to its thread pool only when the serial run would cost meaningfully more than the fan-out, and measures both sides rather than assuming them: no fixed element count works, since the crossover moves ~1000× with arithmetic intensity. The policy, its measurements and its history are in `docs/design/cpu-threading.md` (its "Implementation notes" keep this section's former detail). What a change to it has to respect:

- The fan-out is one ctypes call into a pool `runtime/cpu_pool.py` JIT-compiles from `llvmlite.ir` (`CPool`, one per process, sized by the CPUs the process may run on, on Apple silicon the performance cores). Each worker waits on its own mailbox (futex on Linux, `os_sync_wait_on_address` on macOS 14.4+, polling elsewhere); a launch of `k` threads wakes workers `0 .. k - 2` and runs as the last; the workers and the caller claim chunks from a counter (`GRAIN`, `MIN_CHUNK`) and run them through a per-kernel wrapper (`chunk_wrapper`, `pack_args`). Workers are not pinned, and are made under the interleave NUMA policy. The pool's launch state is one set of JIT'd globals, so a run is a transaction under `CPool._txn` (TA-45), and a start that fails part way joins the workers it made before returning (TA-46). Its inline asm needs the native asm parser initialized (`_target_machine`), or LLVM aborts the process on Apple silicon. `TACK_CPU_POOL=python` selects the `ThreadPoolExecutor` path, which is also the fallback on Windows; tests that simulate workers through `call_range` or `_get_pool` must set it, as `test_cpu_threading.py` does for its whole module.
- Each `CompiledKernel` keeps `ns_per_elem`, a smoothed serial cost, and from it `parallel_min_elems`, so the hot path is one integer compare.
- The fan-out cost is measured by dispatching empty ranges through the real path (no side effects), as a curve over idle gaps whose knots start as pessimistic priors (`_FAN_OUT_IDLE_PRIOR`) and are replaced by fan-outs at pauses the program actually took. Nothing sleeps.
- Serial samples come from inside the range by a golden-ratio walk, on first sight and on rechecks (dispatches 1, 2, 4, … 1024, then every 1024). Worker spans and rates are summarized by `_worker_median`, the lower middle value. A sample is clean only after a serial run laid the range out (`CompiledKernel.scattered` false): a scattered one may lower the estimate but not raise it within `_CONFIRM_BAND`, and until one clean sample a fan-out inside that budget runs serially, at most twice per kernel (P7). A recheck reading 8× low needs a `_second_opinion`; `_parallel_recheck_due` fans out on a back-off when only `r_p` keeps a kernel serial (P9). `scattered` is cleared by `_run_whole_serial` around `_run_serial`, not inside it, because tests replace `_run_serial` with a four-argument lambda.
- In a kernel with atomics (`CompiledKernel.atomics`, from the IR in `_load_kernel`), a fan-out whose `r_p` comes out above `r_s` (on spans trusted as a bound) does not set the floor: the next dispatch runs the whole range serially (`_arbitrate`); without atomics the floor applies at once, as before, since a whole-range run of an image kernel with a cheap background is the stall the rechecks avoid (`test_timing_a_cheap_prefix_does_not_stall_the_rechecks`, which failed on CI's 4 threads when every kernel arbitrated). Below `r_p`, the kernel is `contended`: that sample is `r_s`, no floor, threshold `_NEVER` (`_threshold`); a P9 recheck that fans out faster per element than `r_s` clears it. Otherwise the floor is `r_p` as before. Contended atomics (one counter for every worker) made the floor lift `r_s` to the contended rate, and the kernel fanned out on every call, 8-9x slower on 28 cores.

`TACK_CPU_THREADS` overrides the thread count, as does `tack.device("cpu", num_threads=N)` (a new device, sharing CPU memory with the others); `1` keeps everything on the calling thread. The threading policy is not selectable: policy v1 was removed on 2026-10-15, and `TACK_CPU_POLICY` accepts only `v2` (or unset or empty), raising for anything else, `v1` included. Boolean variables such as `TACK_NO_REINIT` go through `dispatch.env_flag`: `0`, `false`, `no`, `off` and empty mean off.

## Kernel language features

- **Loops**: `for i in range(n)`, `for i in range(start, end)`, `for i in range(start, end, step)`, `for i, j in tack.ndindex(w, h)` (each argument a size or a `(start, end)` pair, whose extent is clamped at zero; the host evaluates `max`/`min` in the launch size), `while`, `break`, `continue`
- **Vectors and matrices**: `tack.vec3(x, y, z)`, `tack.vec3(s)`, `tack.vec3(v)`, `[a, b, c]`; `tack.mat2([a, b], [c, d])`, `tack.mat2(a, b, c, d)`, `tack.mat(row, ...)` (up to 4x4), `tack.identity(n)`; `.dot`, `.cross`, `.norm`, `.normalized`, `.transpose()`, `.trace()`, `.determinant()`, `.inverse()`, `@`
- **Masks**: comparisons of vectors give a vector of `0`/`1` per component; `any(mask)`, `all(mask)`, `tack.select(mask, a, b)`
- **Math**: `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `exp2`, `log`, `log2`, `log10`, `floor`, `ceil`, `abs`, `min`, `max` (two or more values), `pow`; all apply to each component of a vector
- **Types**: `int()`, `float()` casts, plus explicit `tack.i8()`, `tack.u8()`, `tack.i16()`, `tack.u16()`, `tack.i32()`, `tack.u32()`, `tack.i64()`, `tack.u64()`, `tack.f32()`, `tack.f64()`
- **Atomics**: `tack.atomic_add(array, idx, val)`, `tack.atomic_min(...)`, `tack.atomic_max(...)`
- **GPU primitives**: `tack.shared(dtype, size)`, `tack.shared_like(array, size)`, `tack.barrier()`, `tack.thread_id()`
- **Debug**: `print("label:", value)` — emits printf on CPU/CUDA/HIP, no-op on Metal
- **Arrays**: `array[i]`, `array[i, j]`, `array[()]` (zero-dimensional; `array[None]` is refused), `array.shape[k]`, `len(array)` — usable anywhere in a kernel (loop bounds, conditions, arithmetic, indices), not just as the outer loop bound. The dimension index must be a literal. See "Array dimensions" below for what specializes.
- **Reductions**: `array.sum()`, `array.min()`, `array.max()`, `array.mean()` return Python floats. Eligible f32 arrays reduce on GPU; CPU and other dtypes use shared NumPy semantics. f32/f64 sums retain their precision; signed/unsigned integer sums use wrapping i64/u64 accumulators before float conversion. Floating extrema propagate NaNs and use negative-zero min / positive-zero max ties. Empty sum is +0, empty mean NaN, empty extrema raise. Floating addition order may vary; see the absolute error budget in `docs/reference/language-contract.md`. Runtime kernels and GPU block extrema share `codegen/reductions.py`; block arguments/results are f32, requiring explicit casts for other inputs. CPU rejects cooperative kernels; GPU participation and atomic target/order/scope domains are defined in the language contract. See `test_reduction_semantics.py`.

## Platform-specific dependencies

- **macOS (Metal)**: `pyobjc-framework-Metal`
- **Linux/Windows (CUDA)**: `cuda-python>=13.2`, NVIDIA driver + CUDA toolkit
- **Linux (HIP/ROCm)**: `hip-python`, ROCm toolkit
- **Linux (Level Zero/Intel)**: `libze_loader.so`, `libocloc.so` (Intel compute runtime)
- **CPU-only**: `llvmlite`, `numpy`

## HIP backend notes

The HIP codegen (`hip_gen.py`) extends `CUDACodeGen` — HIP device code uses the same syntax as CUDA (`blockIdx`, `threadIdx`, `__global__`, `__shared__`, `__syncthreads`). The differences are `#include <hip/hip_runtime.h>` and the texture handle type, which HIP spells `hipTextureObject_t` (`_TEXTURE_OBJECT_TYPE`, overridden from CUDA's). The runtime (`hip_backend.py`) uses `hip-python` bindings for hipRTC compilation and dispatch. hipRTC compiles with `_HIPRTC_OPTIONS` (`-ffp-contract=on`, see the floating-point baseline), which `cache_target()` names, so a stored code object compiled with other options is not reused.

**Textures need a device that has them.** CDNA parts — gfx940/941/942, i.e. MI300 — have no texture/image hardware, and hipRTC refuses `tex3D` outright ("The image/texture API not supported on the device"). The backend asks `hipDeviceAttributeImageSupport` at init and falls back to software trilinear sampling where the answer is no, the same way the Level Zero backend handles Xe-HPC. gfx90a (MI200) says yes but refuses a linearly filtered 3D texture (`hipErrorNotSupported` from `hipCreateTextureObject`; point sampling works), so init also makes one 2×2×2 texture and falls back when that is refused. The decision is `texture_in_hardware()`, which `_store_texture_shapes` consults before the variant key is built, because it changes the generated code, and which `Texture3D` consults to choose its storage; `tack.inspect` calls the same hook.

`hip-python` is on PyPI, so the `[hip]` extra declares it (manylinux x86_64 only, behind a platform marker), with a lower bound rather than a pin: its version tracks the ROCm it binds to, and the newest wheel has worked a ROCm or two behind (hip-python 7.2.2 on ROCm 7.0.2). Pin (`hip-python~=7.x.0`) only to fix an actual failure. `uv sync --extra hip`, or `pip install -e 'packages/tack-core[hip]'` in a clone. Versions and history: `docs/reference/backends.md`.

Then: `tack.set_default_device("hip")`, or `device="hip"` per array.

`hiprtcDestroyProgram` is never called: it segfaults in hip-python (7.1 through 7.2.2 at least), and the leak is bounded by kernel caching. See `docs/design/vendor-workarounds.md`.

hip-python 10 is a compatibility package over `rocm.bindings` that moved four things (`LONG_VERSION`, `hip.chip`, output structs returned rather than filled, `str` character arrays); `hip_backend.py` supports it and 7.x side by side (`_HIP_PYTHON_VERSION`, `_runtime_library`, `_call_with_output`, `_text`). Details: `docs/design/backend-implementations.md`.

hip-python 7.x's struct wrappers do not expose array handles or fixed-size arrays, so `HIPTextureImage` writes those through ctypes; MI210 runs that path with point filtering only. See `docs/design/vendor-workarounds.md`.

## Level Zero backend notes

The Level Zero codegen (`opencl_gen.py`) extends `CUDACodeGen` — OpenCL C kernel syntax mirrors CUDA with different qualifiers (`__kernel`/`__global`, `get_group_id(0)*get_local_size(0)+get_local_id(0)`/`blockIdx*blockDim+threadIdx`, `__local`/`__shared__`, `barrier()`/`__syncthreads()`). Math functions are overloaded (no `f` suffix). The runtime (`level_zero_backend.py`) uses ctypes bindings to `libze_loader.so` and `libocloc.so`.

Compilation pipeline: OpenCL C source → `libocloc.so` (in-process, via `oclocInvoke`, `_OCLOC_OPTIONS`) → SPIR-V → `zeModuleCreate` (`_MODULE_BUILD_FLAGS`; IGC finalizes here) → `zeKernelCreate`. The disk cache stores the module's native binary beside the SPIR-V (see Disk cache). The ocloc library is part of the Intel compute runtime (`intel-opencl-icd` package).

Requires: `libze_loader.so` (Level Zero runtime), `libocloc.so` (Intel offline compiler). No Python packages needed.

Then: `tack.set_default_device("level_zero")`, or `device="level_zero"` per array.

**Sharing memory needs a shared context.** A Level Zero USM pointer is only
meaningful in the context that allocated it, and DLPack carries none.
`tack.device("level_zero", external_context={driver, device, context})`
makes a device that adopts another library's handles instead of creating a
context (all three are required; `Backend.device_options` declares accepted
keywords and `tack.device` rejects others). Its `memory_domain` includes the
context, so its arrays and the plain Level Zero device's do not mix in a
launch. `tack.interop.vtk.init_level_zero()` takes the handles from VTK's
Viskores/Kokkos SYCL queue and returns the device it made the default, and
the VTK interop refuses Level Zero arrays from any other context. Intel's driver resolves pointers from any
context, so this check is the only one that catches the mistake. The
backend reports `device_memory_spaces = {"level_zero"}` via
`zeMemGetAllocProperties`. See `test_level_zero_context.py`.

## VTK

Tack requires VTK 9.7.1 or later (the `interop` extra) and uses VTK's pythonic API in all VTK code -- the interop, tests, examples, tutorials, benchmarks and tools: keyword constructors (`vtkImageData(dimensions=..., spacing=...)`), properties (`grid.number_of_cells`, `cells.offsets_array`), `grid.points = vtkPoints(data=...)`, `grid.cells = (type, vtkCellArray(offsets=..., connectivity=...))`, dict-like `point_data`/`cell_data`, numpy-compatible arrays (`np.asarray(arr)`, `vtkFloatArray(x)`) in place of `numpy_support`, implicit arrays (`vtkConstantArray[dtype](n, v)`, `vtkAffineArray[dtype](n, slope, intercept)`), pipelines (`(a >> b)()`, a filter called on its input), and `isinstance` in place of `IsA`. Test a doubtful form against the wheel before relying on it; import `vtkmodules.vtkCommonDataModel` first, which registers the pythonic constructors. Where 9.7.1 has none, the classic call stays. Known gaps and traps in 9.7.1:

- `vtkPoints(data=numpy)` stores doubles whatever the array's type: pass `vtkFloatArray(x)` to keep f32.
- A missing array reads as a `NoneArray`, not `None`: test `name in ds.point_data.keys()`.
- `point_data.scalars`/`.vectors` are not writable, and naming an array does not make it active: `SetActiveScalars`/`SetActiveVectors` stay.
- A `vtkCharArray` has no numpy-compatible wrapper; NumPy reads it as bytes (view it as `int8`), and `point_data[...] = ` refuses one.
- A filter with two outputs called as a function returns a tuple of both.
- `vtkPNGWriter`/`vtkJPEGWriter` called on their input write raw pixels: connect and call `Write()`. An XML writer called on its input works, but returns `None` even when it fails.
- An int cell type (`ug.cells = (type, ...)`) makes the types a `vtkConstantArray`, and `InsertNextCell` then records every new cell as that type.
- Not indexable: `vtkMatrix3x3`/`4x4` (`GetElement`), `vtkPartitionedDataSet` (iterates, `GetPartition(i)`), and a partitioned collection iterates over its leaves.
- `vtkPartitionedDataSet(number_of_partitions=n)` and the collection's `number_of_partitioned_data_sets=` keyword are silently ignored (0 partitions): assign the property after constructing. A `vtkStructuredGrid`'s `dimensions` is not readable (`GetDimensions`).

## Do not mention Claude in git commits
