# Install

This page installs Tack and checks that it runs on your devices. It is for
anyone about to use Tack for the first time. After it you have both of Tack's
packages installed, with the extras for your GPU, and a check that each device
compiles and runs a kernel.

Tack is not on PyPI yet, so you install it from a clone of its repository. It
needs Python 3.11 or newer. There is no build step: Tack is pure Python and
compiles kernels when they are first called.

```bash
git clone https://github.com/Kitware/tack.git
cd tack
uv sync --extra cpu
uv run python -c "import tack; print(tack.device('cpu'))"    # prints: cpu
```

These commands use [uv](https://docs.astral.sh/uv/getting-started/installation/).
`uv sync` installs both packages into the clone's own environment:
`tack-core`, which holds the core and communication layers, and `tack-vis`,
which holds the data and visualization layers. Run your scripts with
`uv run python` from the clone's directory.

## Choose the extras for your devices

Each backend is an *extra*, a named set of optional dependencies. Add the
extra for each kind of device you have. Keep the `cpu` extra beside a GPU one,
because the CPU is the default device until you choose another.

```bash
uv sync --extra cpu --extra cuda        # the CPU and an NVIDIA GPU
```

| Device | Extra | What it installs | What the system needs |
|---|---|---|---|
| CPU | `cpu` | `llvmlite` | nothing more |
| Apple GPU | `metal` | `pyobjc-framework-Metal` | macOS on Apple silicon |
| NVIDIA GPU | `cuda` | `cuda-python` 13.2 or newer | the NVIDIA driver, and NVRTC 13 |
| AMD GPU | `hip` | `hip-python` (Linux on x86-64 only) | ROCm |
| Intel GPU | `level_zero` | nothing | the Intel compute runtime: `libze_loader.so` and `libocloc.so` |
| NVIDIA Warp (prototype) | `warp` | `warp-lang` and `warp-simdata` | Warp's own requirements |

The Intel extra installs nothing because Tack reaches Level Zero through system
libraries, not Python packages. It exists so that the same commands work for
every device.

Other extras add what some parts of Tack connect to:

| Extra | What it installs | What it is for |
|---|---|---|
| `interop` | VTK 9.7.1 or newer, and PyMFEM | reading and drawing data with VTK, and importing MFEM solutions |
| `mpi` | `mpi4py` 4.0 or newer | running over MPI ranks; it builds against your system's MPI, so `mpicc` must be on your path |
| `dev` | `pytest` and the test tools | running Tack's tests |

PyMFEM has wheels for macOS on Apple silicon and for Linux on x86-64 only.
Elsewhere the `interop` extra installs VTK alone.

## Install into another environment

To use Tack from an application's own environment, install the packages from
the clone with pip. Install `tack-core` first, with the extras for your
devices, then `tack-vis`, which depends on it.

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\Activate.ps1
python -m pip install -e '/path/to/tack/packages/tack-core[cpu,cuda]'
python -m pip install -e /path/to/tack/packages/tack-vis
python -m pip install 'vtk>=9.7.1'                 # optional: VTK interop
```

In this form `interop` is not an extra of either package, so install VTK and
MFEM (`mfem`) yourself if you need them. The `mpi` and `warp` extras are
extras of `tack-core` and `tack-vis` and work the same way. A plain
`tack-core` without the `cpu` extra cannot run kernels on the CPU.

## Check that each device works

This script runs one kernel on every device Tack finds. A device works when
its line says so.

<!-- tack-check: install-devices -->
```python
import tack

tack.set_default_device("cpu")


@tack.kernel
def double(x, out):
    for i in range(x.shape[0]):
        out[i] = 2.0 * x[i]


working = []
for kind in ["cpu", "metal", "cuda", "hip", "level_zero"]:
    if tack.device_count(kind) == 0:
        print(f"{kind}: not found")
        continue
    with tack.device(kind):                      # the default device inside the block
        x = tack.arange(4, dtype=tack.f32)       # made on this device
        out = tack.empty_like(x)
        double(x, out)                           # runs where x and out are
        assert out.numpy().tolist() == [0.0, 2.0, 4.0, 6.0]
    working.append(kind)
    print(f"{kind}: works")

assert "cpu" in working
```

`tack.device_count` says 0 both when a device's software is missing and when
the machine has no such device. To find out which, ask for the device itself.
It raises an error that names what is missing and how to install it:

```text
>>> tack.device("cuda")
RuntimeError: Cannot start the 'cuda' device: missing dependency.
  No module named 'cuda'
  Requires an NVIDIA GPU and the CUDA toolkit; in a clone of the repository:
    pip install -e 'packages/tack-core[cuda]'   (or: uv sync --extra cuda)
```

## Details

**NVIDIA.** Tack is tested with NVRTC 13, the compiler `cuda-python` uses to
build kernels at run time. With an older NVRTC, Tack warns at the first
compile, because NVRTC 12.0 miscompiled a Tack kernel. To get NVRTC 13, install
the `nvidia-cuda-nvrtc` wheel for a CUDA version no newer than your driver's
(the "CUDA Version" that `nvidia-smi` reports), for example
`pip install 'nvidia-cuda-nvrtc==13.2.*'`. `cuda-python` finds it before a
system toolkit.

**AMD.** The version of `hip-python` follows the ROCm release it binds to:
7.1.x for ROCm 7.1, 7.2.x for ROCm 7.2. The extra sets only a lower bound. If
your ROCm is not the newest, install the matching version, for example
`pip install 'hip-python~=7.2.0'`. ROCm 7.0 and 7.1 can miscompile some integer
code, so use a newer ROCm if you can. [Backends](reference/backends.md#installing)
has the details.

**Intel.** Some Intel GPUs have double precision and some do not. Tack asks
the device when it starts, and refuses `f64` arrays on a device without it.

**Apple.** Apple GPUs have no double precision, so Metal has no `f64`. Every
dataset algorithm still runs there, in single precision.

**The documentation's own examples.** The scripts under `docs/examples/` and
`docs/gallery/` run from the clone with `uv run python`. Some need the
`interop` extra for their figures.

## Where to go next

- [The core's first page](core/index.md) writes and runs a first kernel, and
  explains what happens on the first call.
- [Choose a device](core/choose-a-device.md) explains devices, the default
  device and machines with several GPUs.
- [The visualization layer's first page](visualization/index.md) runs a first
  filter on a dataset.
