# Keep compiled kernels between runs

A kernel compiles the first time it is called with new argument types and
shapes. Tack keeps what it compiles on disk, so the next process that calls
the same kernel loads it instead of compiling it again. The cache is on by
default and needs no code. You need the rest of this page only to find where
the cache lives, move it, limit it or turn it off, or use it in an MPI job on
a shared file system.

This example runs one small program twice, each time in a new process, with
the cache in a directory of its own. The first run compiles and stores the
kernel. The second finds it on disk:

<!-- tack-check: cache-first -->
```python
import subprocess
import sys
import tempfile
from pathlib import Path

import tack

tack.set_default_device("cpu")

program = '''
import sys
import tack
import tack.cache

tack.set_default_device("cpu")
tack.cache.configure(directory=sys.argv[1])


@tack.kernel
def scale(x, out, factor):
    for i in range(x.shape[0]):
        out[i] = x[i] * factor


x = tack.ones(8)
scale(x, tack.empty_like(x), 2.0)
print(tack.cache.stats().get("variant_hit", 0))
'''

with tempfile.TemporaryDirectory() as folder:
    script = Path(folder) / "scale.py"
    script.write_text(program)
    cache = str(Path(folder) / "cache")
    runs = [subprocess.run([sys.executable, str(script), cache], capture_output=True,
                           text=True, check=True).stdout.split()[-1] for _ in range(2)]
assert runs == ["0", "1"]          # compiled the first time, loaded the second
```

## What the cache saves

A first call has two parts: Tack's own work on the kernel's Python source, and
the device's compiler. The cache keeps the results of both, so a warm start
skips both. These are first calls of `contour` on a small grid, a
visualization filter that compiles many kernels, in a new process:

| machine and device | cache off | cache warm |
|---|---|---|
| Apple M1 Max, CPU | 445 ms | 31 ms |
| Apple M1 Max, Metal | 1.1 s | 31 ms |
| NVIDIA RTX 6000 Ada, CUDA | 1.15 s | 72 ms |
| Intel Max 1100, Level Zero | 5.1 s | 145 to 190 ms |
| AMD MI210, HIP | 6.9 s | 37 ms |

On CUDA most of the cold time is the vendor's compiler, and on Level Zero and
HIP nearly all of it is. That is why Tack keeps each device's compiled code,
and not only its own intermediate form.

## Change where and how it caches

Three environment variables control the cache. They are read when the first
kernel compiles:

| variable | meaning |
|---|---|
| `TACK_CACHE_DIR` | Where entries are kept. The default is `~/Library/Caches/tack` on macOS, `$XDG_CACHE_HOME/tack` or `~/.cache/tack` on Linux, and `%LOCALAPPDATA%\tack\Cache` on Windows. |
| `TACK_CACHE` | `write` looks entries up and stores new ones, the default. `read` looks up but never stores. `bundle` reads only the directory's bundle file, the default under MPI. `off` turns the cache off. |
| `TACK_CACHE_SIZE` | The most the entries may take, such as `500M` or `2G` in binary units, or `unlimited`. The default is `1G`. |

`tack.cache.configure` sets the same from Python, over the environment. It
takes effect at the next kernel compiled:

<!-- tack-check: cache-configure -->
```python
import tempfile

import tack
import tack.cache

tack.set_default_device("cpu")
folder = tempfile.mkdtemp()                   # a directory only you can write

tack.cache.configure(directory=folder, mode="write", max_size="4G")
directory, mode, rank = tack.cache.settings()
assert str(directory) == folder and mode == "write" and rank is None
assert tack.cache.size_limit() == 4 * 2**30

tack.cache.configure(mode="off")              # no cache for this process
assert tack.cache.settings()[1] == "off"
```

The directory must belong to you and be writable by nobody else, since its
entries hold code that Tack will run. Tack refuses a directory that fails
this, with a warning, and compiles without the cache.

## Know when an entry is used

Two things are stored for each kernel. One is the kernel after Tack has
translated its Python source, before it is specialized for argument types.
The other is each compiled variant: one compiled form of the kernel, for one
set of argument types and shapes.

An entry is used only when everything it was made from is unchanged. That
means the kernel's source, every device function and constant it uses, the
classes it is given, the argument types and shapes, the Tack version, and the
device with its compiler and driver. So editing a kernel, a device function
it calls, a `tack.constant` it reads or a device class it is given makes the
next process compile again. So do upgrading Tack and moving to another
machine. A damaged or partly written entry is ignored, and the kernel
compiles.

Debug mode is part of the key. A kernel compiled with bounds checks is a
different kernel, so switching modes never mixes the two.

## Keep it to a size

The entries of every Tack version share one limit, `TACK_CACHE_SIZE`. Each
Tack version, and each edit of a Tack checkout, keeps entries of its own, and
those of a version you no longer run would otherwise stay for good. Past the
limit, a process that stores entries deletes the least recently used ones
until 90% of the limit is left. Loading an entry marks it as used, so the
kernels you run stay.

The directory is checked at most once an hour, by all processes together,
never once per kernel. The cache can therefore briefly exceed its limit.

The command line shows and manages the cache:

```bash
python -m tack.cache info                  # settings, entries and size
python -m tack.cache prune                 # keep to the limit now
python -m tack.cache prune --max-size 200M # ...or to another limit
python -m tack.cache clear                 # this Tack version's entries
python -m tack.cache clear --all           # every version's entries
```

`clear --all` deletes only what the cache made, so the directory may hold
other files too. The same operations are `tack.cache.describe`,
`tack.cache.prune` and `tack.cache.clear` in Python.

## Use the cache in an MPI job

On a parallel file system such as Lustre or GPFS, thousands of ranks each
looking up a file per kernel load the file system's metadata servers, as
thousands of ranks importing Python do. A *bundle* avoids it: one file that
holds every entry, which each rank reads once.

1. **Warm up.** Run the job's kernels once, on one rank or in a small job,
   with `TACK_CACHE=write` and `TACK_CACHE_DIR` set to a directory of your
   own on scratch.
2. **Pack** the directory into a bundle:

   ```bash
   TACK_CACHE_DIR=/scratch/me/tack-cache python -m tack.cache pack
   ```

3. **Run the job** with the same `TACK_CACHE_DIR`. Under MPI the default
   mode is `bundle`. Each rank reads the bundle once, and never looks up or
   writes a file per kernel. A kernel the bundle lacks compiles in memory,
   as it would without a cache.

Tack recognizes an MPI launch from the launcher's environment variables:
Open MPI, MPICH, Intel MPI, MVAPICH2, Cray PALS and ALPS, and Slurm's `srun`.
A launch of one task counts as no MPI. So does a Slurm batch script itself,
so a warm-up run on the script's own line, without `srun`, stores entries as
it would on a workstation.

To have only rank 0 read the bundle and broadcast it to the others, call
`tack.cache.load_bundle(comm=MPI.COMM_WORLD)` on every rank before the first
kernel, with mpi4py's communicator. It is collective, and returns the number
of entries read. If rank 0 cannot use the bundle, because there is none, it
is damaged, or it is not your own, rank 0 warns why and every rank compiles
its kernels.

With `TACK_CACHE=write` under MPI, only rank 0 stores entries and the other
ranks read. Concurrent writers are safe, but the bundle is the better path
for a large job. [Run under MPI](../communication/mpi.md) covers the rest of
an MPI job's setup.

## Time a cold compile

With the cache on, the first call in a process may load a kernel rather than
compile it. To time a cold compile, run with `TACK_CACHE=off`. See
[Debug and time a kernel](debug-and-time.md#time-a-kernel).

## Details

- **What each device stores.** The CPU stores object code, and Metal its
  generated source. CUDA stores PTX, which the driver compiles for the GPU
  when it loads. HIP stores the code object, machine code for the GPU's
  architecture. Level Zero stores the GPU's native code, which loads without
  compiling, and the SPIR-V it was made from, used if a driver update or
  another stepping of the GPU refuses the native code. The Warp prototype
  stores only Tack's part, and Warp's own cache keeps its compiled code.
- **Safe writes.** Entries are written to a temporary file and renamed into
  place, so a reader never sees a partial one. Each entry carries a checksum.
- **Bundles and the size limit.** Bundles are neither counted against the
  limit nor deleted by pruning. You make and remove them by hand.
- **The counters.** `tack.cache.stats()` returns this process's counts of
  hits, misses and writes, as the first example reads.

## Where to go next

- [Debug and time a kernel](debug-and-time.md), to measure first calls and
  warm calls.
- [Run under MPI](../communication/mpi.md), for MPI jobs on clusters.
- The [`tack.cache` reference](../contracts/runtime-api.md#tackcache), for
  every function's arguments.
