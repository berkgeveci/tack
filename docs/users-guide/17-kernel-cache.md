# The Kernel Cache

A kernel compiles the first time it is called with a new combination of
argument types and shapes. Most of that time is Tack's own front end, and
the rest is the backend's compiler. Tack keeps what it compiles on disk, so
a later process starts warm. On an M1 Max, the first contour of a 21³ grid
in a new process took 445 ms on the CPU and 1.1 s on Metal; with the cache
warm, 31 ms on both. On an RTX 6000 Ada it took 1.15 s on CUDA, 0.88 s
with only the lowered kernels stored, and 72 ms with their PTX stored too.

The cache is on by default and needs no code. This page covers where it
lives, how to turn it off, and how to use it in MPI jobs on a shared file
system.

## Settings

Three environment variables, read when the first kernel compiles:

| Variable | Meaning |
|---|---|
| `TACK_CACHE_DIR` | Where entries are kept. Default: `~/Library/Caches/tack` on macOS, `$XDG_CACHE_HOME/tack` or `~/.cache/tack` on Linux, `%LOCALAPPDATA%\tack\Cache` on Windows. |
| `TACK_CACHE` | `write`: look up and store (the default). `read`: look up, never store. `bundle`: only the directory's bundle file (the default under MPI, below). `off`: no cache. |
| `TACK_CACHE_SIZE` | The most the entries may take, such as `500M` or `2G` (binary units), or `unlimited`. Default: `1G`. See below. |

The same from Python, before the first kernel compiles:

```python
import tack.cache

tack.cache.configure(directory="/scratch/me/tack-cache", mode="write")
tack.cache.configure(max_size="4G")
tack.cache.configure(mode="off")
```

The directory must belong to you and be writable by nobody else, since
its entries hold code. Tack refuses one that is not, with a warning, and
compiles without it.

## What is kept, and when it is not used

Two things are stored for each kernel:

- **The lowered kernel**, before it is specialized for argument types. Its
  key is the kernel's source and specialization, Tack's own version, and
  whether debug mode is on: a kernel lowered with bounds checks is a
  different kernel, and so are the variants compiled from it. Every name the kernel and its device functions resolve (another
  `@tack.func`, a `tack.constant`, a module) is recorded with what it was
  bound to, and the entry is used only if each still resolves to the same
  thing.
- **Each compiled variant**: object code on the CPU, generated source on
  Metal. Its key adds the argument types, the shapes compiled in, and what
  the code depends on: the LLVM version and host CPU, or the GPU and
  system.

So editing a kernel, a device function it calls, a constant it reads or a
template class it is given makes the next process compile again, as does
upgrading Tack or moving to another machine. A damaged or partly written
entry is ignored and the kernel compiled.

## Size

The entries of every Tack version share one limit, `TACK_CACHE_SIZE` (1 GiB
by default): each Tack version, and each edit of a Tack checkout, keeps its
own entries, and those of a version no longer run would otherwise stay
forever. Past the limit, a process that stores entries deletes the least
recently used, until 90% of the limit is left. Loading an entry marks it
used (to the hour), so the kernels you run stay. The directory is checked
at most once an hour by all processes together, never per kernel, so the
cache can briefly exceed the limit. Bundles are neither counted nor
deleted: they are made, and removed, by hand.

```bash
python -m tack.cache info                  # settings, entries and size
python -m tack.cache prune                 # keep to the limit now
python -m tack.cache prune --max-size 200M # ...or to another one
python -m tack.cache clear                 # this Tack version's entries
python -m tack.cache clear --all           # every version's
```

`clear --all` deletes only what the cache made (version directories,
bundles), so the directory may hold other files.

## MPI jobs on a shared file system

Thousands of ranks looking up a file per kernel on Lustre or GPFS is the
same metadata storm as thousands of ranks importing Python. Tack avoids it
with a bundle: one file that holds every entry, which each rank reads once.

1. **Warm up** with a run that compiles the kernels the job uses: one rank,
   or a small job, with `TACK_CACHE=write` and `TACK_CACHE_DIR` set to a
   directory of your own, on scratch.
2. **Pack** the directory into a bundle:

   ```bash
   TACK_CACHE_DIR=/scratch/me/tack-cache python -m tack.cache pack
   ```

3. **Run the job** with the same `TACK_CACHE_DIR`. Under MPI the default
   mode is `bundle`: each rank reads the bundle once and never looks up or
   writes a per-kernel file. A kernel the bundle lacks compiles in memory,
   as it would without a cache.

Tack recognizes MPI from the launcher's rank variables (Open MPI, MPICH,
Intel MPI, MVAPICH2, Cray PALS and ALPS, Slurm's `srun`). A launch of one
task counts as no MPI, and so does a Slurm batch script itself: a warm-up
run on the script's own line, without `srun`, stores entries as it would
on a workstation.

To have only rank 0 read the bundle, and broadcast it to the others, call
this on every rank before the first kernel (with mpi4py's communicator):

```python
from mpi4py import MPI
import tack.cache

tack.cache.load_bundle(comm=MPI.COMM_WORLD)
```

It returns the number of entries read. If rank 0 cannot use the bundle
(there is none, it is damaged, or it is not your own), rank 0 warns why
and every rank compiles its kernels, as it would without a cache.

With `TACK_CACHE=write` under MPI, only rank 0 stores entries; the other
ranks read. Entries are written to a temporary file and renamed into
place, so a reader never sees a partial one. Concurrent writers are safe,
but the bundle is the better path for a large job.

## Measuring compile time

To time a cold compile, turn the cache off (`TACK_CACHE=off`), or the
first call in a process may be a disk hit rather than a compile. See
[Debugging and Timing](15-debugging-and-timing.md).

## Backends

The CPU, Metal and CUDA backends store both levels (CUDA stores PTX, which
the driver compiles for the device at load). On HIP and Level Zero, lowered
kernels are stored, and the variants are compiled in each process; storing
their compiled code is planned.
