"""The disk cache of lowered and compiled kernels (``tack.cache``).

Each case runs kernels in fresh processes against a temporary cache
directory: a hit must give the same results as a compile, and every change
that could alter the generated code must miss. The rules for shared file
systems (MPI ranks read one bundle and write nothing, untrusted directories
are refused, damaged entries are ignored) are checked the same way.
"""

import json
import os
import pickle
import stat
import subprocess
import sys
import textwrap

import pytest

import tack
from tack.runtime import disk_cache

_SCRIPT = """
import json, sys
sys.path.insert(0, {moddir!r})
import tack
from tack import cache
from tack.runtime import disk_cache
tack.init(arch=getattr(tack, {arch!r}))
{setup}
import kernels
result = kernels.run()
print(json.dumps({{"result": result, "stats": cache.stats()}}))
"""

_KERNELS = """
import numpy as np
import tack

SCALE = tack.constant(2.0)


@tack.func
def scaled(x):
    return x * SCALE


@tack.kernel
def apply(x, out):
    for i in range(x.shape[0]):
        out[i] = scaled(x[i]) + 1.0


@tack.device_class
class Shift:
    OFFSET = 0.5

    def __init__(self, data):
        self.data = data

    @tack.func
    def get(self, i):
        return self.data[i] + self.OFFSET


class Shift10(Shift):
    @tack.func
    def get(self, i):
        return self.data[i] + 10.0


@tack.kernel
def shifted(s, out):
    for i in range(out.shape[0]):
        out[i] = s.get(i)


@tack.kernel
def lengths(v, out):
    for i in range(out.shape[0]):
        out[i] = v[i].norm()


@tack.kernel
def rows(a, out):
    for i in range(out.shape[0]):
        out[i] = a[i, 0] + a[i, 1]


def run():
    n = 8
    x = tack.field(tack.f32, (n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    out = tack.field(tack.f32, (n,))
    results = []
    apply(x, out)
    results.append(out.to_numpy().tolist())
    shifted(Shift(x), out)
    results.append(out.to_numpy().tolist())
    shifted(Shift10(x), out)
    results.append(out.to_numpy().tolist())
    v = tack.Vector.field(3, tack.f32, (n,))
    v.from_numpy(np.repeat(np.arange(n, dtype=np.float32)[:, None], 3, axis=1))
    lengths(v, out)
    results.append([round(r, 5) for r in out.to_numpy().tolist()])
    a = tack.field(tack.f32, (n, 2))
    a.from_numpy(np.arange(2 * n, dtype=np.float32).reshape(n, 2))
    rows(a, out)
    results.append(out.to_numpy().tolist())
    return results
"""


def _write(path, text):
    path.write_text(textwrap.dedent(text))


def _run(tmp_path, arch="cpu", env=None, setup="", modules=None):
    """Run ``kernels.run()`` in a fresh process; its results and cache counters."""
    moddir = tmp_path / "src"
    moddir.mkdir(exist_ok=True)
    if not (moddir / "kernels.py").exists():
        _write(moddir / "kernels.py", _KERNELS)
    for name, text in (modules or {}).items():
        _write(moddir / name, text)
    script = _SCRIPT.format(moddir=str(moddir), arch=arch, setup=textwrap.dedent(setup))
    full_env = {**os.environ, "TACK_CACHE_DIR": str(tmp_path / "cache"),
                "TACK_CACHE": "write", "PYTHONDONTWRITEBYTECODE": "1"}
    for name in disk_cache._RANK_VARS + disk_cache._SIZE_VARS:
        full_env.pop(name, None)
    full_env.update(env or {})
    full_env = {k: v for k, v in full_env.items() if v is not None}
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=full_env, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return {**json.loads(proc.stdout.strip().splitlines()[-1]), "stderr": proc.stderr}


def _cache_dir(tmp_path):
    path = tmp_path / "cache"
    path.mkdir(exist_ok=True, mode=0o700)
    return path


def _entries(tmp_path):
    return sorted(p for p in (tmp_path / "cache").rglob("*")
                  if p.suffix in (".ir", ".variant"))


def test_a_second_process_reuses_every_kernel(tmp_path, backend):
    _cache_dir(tmp_path)
    first = _run(tmp_path, backend)
    assert first["stats"].get("ir_write", 0) >= 5
    assert first["stats"].get("variant_write", 0) >= 5
    second = _run(tmp_path, backend)
    assert second["result"] == first["result"]
    assert second["stats"].get("ir_miss", 0) == 0
    assert second["stats"].get("variant_miss", 0) == 0
    assert second["stats"].get("variant_hit", 0) == first["stats"]["variant_write"]


def test_a_changed_device_function_misses(tmp_path):
    _cache_dir(tmp_path)
    first = _run(tmp_path)
    source = (tmp_path / "src" / "kernels.py").read_text()
    changed = source.replace("return x * SCALE", "return x * SCALE * 3.0")
    assert changed != source
    second = _run(tmp_path, modules={"kernels.py": changed})
    assert second["stats"].get("ir_stale", 0) == 1
    assert second["result"][0] == [3 * 2.0 * i + 1.0 for i in range(8)]
    assert second["result"][1:] == first["result"][1:]


def test_a_changed_constant_misses(tmp_path):
    _cache_dir(tmp_path)
    _run(tmp_path)
    source = (tmp_path / "src" / "kernels.py").read_text()
    second = _run(tmp_path, modules={
        "kernels.py": source.replace("tack.constant(2.0)", "tack.constant(5.0)")})
    assert second["stats"].get("ir_stale", 0) == 1
    assert second["result"][0] == [5.0 * i + 1.0 for i in range(8)]


def test_a_changed_template_override_misses(tmp_path):
    _cache_dir(tmp_path)
    first = _run(tmp_path)
    assert first["result"][1] != first["result"][2]   # Shift and Shift10 kept apart
    source = (tmp_path / "src" / "kernels.py").read_text()
    second = _run(tmp_path, modules={
        "kernels.py": source.replace("self.data[i] + 10.0", "self.data[i] + 20.0")})
    assert second["result"][2] == [i + 20.0 for i in range(8)]
    assert second["result"][1] == first["result"][1]
    assert second["stats"].get("ir_miss", 0) == 1    # Shift10 only


def test_a_changed_class_constant_misses(tmp_path):
    _cache_dir(tmp_path)
    _run(tmp_path)
    source = (tmp_path / "src" / "kernels.py").read_text()
    second = _run(tmp_path, modules={
        "kernels.py": source.replace("OFFSET = 0.5", "OFFSET = 1.5")})
    assert second["result"][1] == [i + 1.5 for i in range(8)]


def test_kernels_from_one_factory_keep_their_own_entries(tmp_path):
    _cache_dir(tmp_path)
    factory = """
    import numpy as np
    import tack


    @tack.func
    def double(x):
        return x * 2.0


    @tack.func
    def square(x):
        return x * x


    def make(op):
        @tack.kernel
        def k(x, out):
            for i in range(x.shape[0]):
                out[i] = op(x[i])
        return k


    def run():
        x = tack.field(tack.f32, (4,))
        x.from_numpy(np.arange(4, dtype=np.float32))
        out = tack.field(tack.f32, (4,))
        results = []
        for op in (double, square, double):
            make(op)(x, out)
            results.append(out.to_numpy().tolist())
        return results
    """
    first = _run(tmp_path, modules={"kernels.py": factory})
    assert first["result"] == [[0, 2, 4, 6], [0, 1, 4, 9], [0, 2, 4, 6]]
    second = _run(tmp_path)
    assert second["result"] == first["result"]
    assert second["stats"].get("ir_miss", 0) == 0
    assert second["stats"].get("ir_stale", 0) == 0


def test_another_tack_misses(tmp_path):
    _cache_dir(tmp_path)
    _run(tmp_path)
    second = _run(tmp_path, setup="disk_cache._TACK_FINGERPRINT = '0' * 64")
    assert second["stats"].get("ir_hit", 0) == 0
    assert second["stats"].get("variant_hit", 0) == 0


def test_another_target_recompiles_only_the_variants(tmp_path):
    _cache_dir(tmp_path)
    first = _run(tmp_path)
    second = _run(tmp_path, setup="""
        from tack.runtime.dispatch import get_backend
        backend = get_backend()
        target = backend.cache_target()
        type(backend).cache_target = lambda self: target + "; another CPU"
    """)
    assert second["stats"].get("ir_miss", 0) == 0
    assert second["stats"].get("variant_hit", 0) == 0
    assert second["result"] == first["result"]


def test_damaged_entries_are_ignored(tmp_path):
    _cache_dir(tmp_path)
    first = _run(tmp_path)
    entries = _entries(tmp_path)
    for k, path in enumerate(entries):
        data = path.read_bytes()
        # Truncated, flipped, or replaced by another entry's bytes.
        path.write_bytes([data[: len(data) // 2], data[:-1] + bytes([data[-1] ^ 1]),
                          entries[(k + 1) % len(entries)].read_bytes()][k % 3])
    second = _run(tmp_path)
    assert second["result"] == first["result"]
    assert second["stats"].get("bad", 0) == len(entries)
    third = _run(tmp_path)
    assert third["stats"].get("variant_miss", 0) == 0


def test_read_mode_writes_nothing_and_off_touches_nothing(tmp_path):
    _cache_dir(tmp_path)
    first = _run(tmp_path, env={"TACK_CACHE": "read"})
    assert _entries(tmp_path) == []
    assert first["stats"].get("ir_write", 0) == 0
    _run(tmp_path)
    reading = _run(tmp_path, env={"TACK_CACHE": "read"})
    assert reading["stats"].get("variant_miss", 0) == 0
    off = _run(tmp_path, env={"TACK_CACHE": "off", "TACK_CACHE_DIR": str(tmp_path / "none")})
    assert not (tmp_path / "none").exists()
    assert off["stats"] == {}


_RANK_4_OF_8 = {"OMPI_COMM_WORLD_RANK": "4", "OMPI_COMM_WORLD_SIZE": "8"}


def test_mpi_ranks_read_only_the_bundle(tmp_path):
    _cache_dir(tmp_path)
    # Without a bundle, a rank compiles and neither reads nor writes entries.
    rank = _run(tmp_path, env=_RANK_4_OF_8)
    assert _entries(tmp_path) == []
    assert rank["stats"].get("ir_hit", 0) == 0
    # A warm-up run, packed; then the per-kernel files are gone and a rank
    # finds everything in the bundle.
    first = _run(tmp_path)
    path, count = disk_cache.pack(tmp_path / "cache")
    assert count == len(_entries(tmp_path)) and path.exists()
    for entry in _entries(tmp_path):
        entry.unlink()
    rank = _run(tmp_path, env=_RANK_4_OF_8)
    assert rank["result"] == first["result"]
    assert rank["stats"].get("bundle_entries") == count
    assert rank["stats"].get("variant_miss", 0) == 0
    assert _entries(tmp_path) == []


def test_only_rank_zero_writes(tmp_path):
    _cache_dir(tmp_path)
    _run(tmp_path, env={**_RANK_4_OF_8, "TACK_CACHE": "write"})
    assert _entries(tmp_path) == []
    _run(tmp_path, env={"OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_SIZE": "8",
                        "TACK_CACHE": "write"})
    assert _entries(tmp_path)


def test_a_one_task_launch_is_not_mpi(monkeypatch):
    for name in disk_cache._RANK_VARS + disk_cache._SIZE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "1")
    assert disk_cache.mpi_rank() is None
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "2")
    assert disk_cache.mpi_rank() == 0


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_a_directory_others_can_write_is_refused(tmp_path):
    cache = _cache_dir(tmp_path)
    cache.chmod(0o777)
    first = _run(tmp_path)
    assert "others can write to it" in first["stderr"]
    assert _entries(tmp_path) == []
    assert first["stats"].get("ir_write", 0) == 0
    assert stat.S_IMODE(cache.stat().st_mode) == 0o777


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX permissions")
def test_a_directory_the_cache_makes_is_its_own_under_umask_002(tmp_path):
    """A group-writable umask must not leave the directory the cache creates
    group-writable, which the cache would then refuse."""
    first = _run(tmp_path, setup="import os; os.umask(0o002)")
    assert "others can write to it" not in first["stderr"]
    assert first["stats"].get("ir_write", 0) > 0
    assert stat.S_IMODE((tmp_path / "cache").stat().st_mode) == 0o700


def test_clearing_every_version_leaves_other_files(tmp_path):
    """The directory may be shared: only the cache's own names are removed."""
    cache = _cache_dir(tmp_path)
    _run(tmp_path)
    old = cache / "v1-0123456789abcdef"
    (old / "ab").mkdir(parents=True)
    (old / "ab" / "ab12.ir").write_bytes(b"old")
    (cache / "bundle-v1-0123456789abcdef.tack").write_bytes(b"old")
    neighbours = [cache / "venv" / "bin", cache / "vtk-data"]
    for path in neighbours:
        path.mkdir(parents=True)
    files = [cache / "video.mp4", cache / "bundle-notes.txt", cache / "vtk-data" / "a.vti"]
    for path in files:
        path.write_bytes(b"keep")
    entries = len(_entries(tmp_path))   # this version's and the old .ir
    assert disk_cache.clear(cache, all_versions=True) == entries + 1   # and the old bundle
    assert sorted(p.name for p in cache.iterdir()) == [
        "bundle-notes.txt", "venv", "video.mp4", "vtk-data"]
    assert all(path.exists() for path in neighbours + files)


def test_concurrent_writers_leave_whole_entries(tmp_path):
    _cache_dir(tmp_path)
    moddir = tmp_path / "src"
    moddir.mkdir()
    _write(moddir / "kernels.py", _KERNELS)
    script = _SCRIPT.format(moddir=str(moddir), arch="cpu", setup="")
    env = {**os.environ, "TACK_CACHE_DIR": str(tmp_path / "cache"), "TACK_CACHE": "write",
           "PYTHONDONTWRITEBYTECODE": "1"}
    procs = [subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env) for _ in range(4)]
    outputs = [p.communicate(timeout=300) for p in procs]
    assert all(p.returncode == 0 for p in procs), [err for _, err in outputs]
    results = {json.dumps(json.loads(out.strip().splitlines()[-1])["result"])
               for out, _ in outputs}
    assert len(results) == 1
    assert not list((tmp_path / "cache").rglob("*.tmp"))
    after = _run(tmp_path)
    assert after["stats"].get("bad", 0) == 0
    assert after["stats"].get("variant_miss", 0) == 0


class _FakeComm:
    """What ``load_bundle`` uses of an mpi4py communicator."""

    def __init__(self, rank, data=None):
        self.rank = rank
        self.data = data

    def Get_rank(self):
        return self.rank

    def bcast(self, data, root=0):
        return data if self.rank == root else self.data


def test_load_bundle_broadcasts_from_rank_zero(tmp_path):
    _cache_dir(tmp_path)
    _run(tmp_path)
    path, count = disk_cache.pack(tmp_path / "cache")
    try:
        disk_cache.configure(directory=tmp_path / "cache", mode="bundle")
        assert disk_cache.load_bundle(comm=_FakeComm(0)) == count
        # Another rank is handed the bytes and never reads the file.
        moved = path.with_suffix(".moved")
        path.rename(moved)
        assert disk_cache.load_bundle(comm=_FakeComm(1, moved.read_bytes())) == count
    finally:
        disk_cache.configure()


def test_the_command_line(tmp_path):
    _cache_dir(tmp_path)
    _run(tmp_path)

    def cli(*args):
        proc = subprocess.run([sys.executable, "-m", "tack.cache", "--dir",
                               str(tmp_path / "cache"), *args],
                              capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    info = json.loads(cli("info"))
    assert info["entries"] == len(_entries(tmp_path)) > 0
    assert f"{info['entries']} entries" in cli("pack")
    assert json.loads(cli("info"))["bundle"]
    assert "removed" in cli("clear")
    assert _entries(tmp_path) == []


def test_scalar_types_keep_their_identity_through_pickle():
    for dtype in (tack.i8, tack.u16, tack.i32, tack.u64, tack.f32, tack.f64):
        assert pickle.loads(pickle.dumps(dtype)) is dtype


def test_settings_from_the_environment(monkeypatch, tmp_path):
    for name in disk_cache._RANK_VARS + disk_cache._SIZE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TACK_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TACK_CACHE", "read")
    assert disk_cache.settings() == (tmp_path, "read", None)
    monkeypatch.setenv("TACK_CACHE", "")
    assert disk_cache.settings()[1] == "write"
    # A Slurm batch script is no parallel launch; its srun steps are.
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    assert disk_cache.settings()[1:] == ("write", None)
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "1")
    assert disk_cache.settings()[1:] == ("write", None)
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "4")
    assert disk_cache.settings()[1:] == ("bundle", 0)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert disk_cache.settings()[1:] == ("bundle", 3)
    monkeypatch.setenv("TACK_CACHE", "sometimes")
    with pytest.raises(ValueError, match="TACK_CACHE must be one of"):
        disk_cache.settings()
    with pytest.raises(ValueError, match="cache mode"):
        disk_cache.configure(mode="sometimes")
