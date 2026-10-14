"""A disk cache of lowered and compiled kernels, safe for MPI jobs.

A kernel's first call in a process spends most of its time in Tack's own
front end -- lowering, the IR passes and code generation -- and the rest in
the backend's compiler. This cache keeps both across processes, in two
levels:

- **Lowered IR** (``.ir`` entries), keyed by content: the kernel's source,
  its specialization (vector and texture fields, template classes, their
  methods and constants), Tack's own sources and the Python version. Every
  name lowering resolved in the kernel's namespace, and in the namespaces of
  the device functions it inlined, is recorded with a fingerprint of what it
  was bound to; an entry is used only if every name still resolves to the
  same thing. A changed device function or constant therefore misses.
- **Compiled variants** (``.variant`` entries): the IR after the passes, and
  the backend's artifact (object code on the CPU, MSL source on Metal),
  keyed by the lowered entry, the argument types and baked-in shapes, and
  what the backend's code depends on (``Backend.cache_target``).

The rules that make it safe on a parallel file system shared by thousands
of ranks:

- Entries are written to a temporary file and renamed into place, which is
  atomic on POSIX file systems including Lustre and GPFS. Writers of one
  entry write the same bytes, so the last rename wins harmlessly. No locks.
- Every entry carries a checksum and its full key. A partial, corrupt or
  foreign entry is ignored and the kernel compiled as if it were absent.
- Under MPI the default mode is ``bundle``: each rank reads one bundle file
  (made by ``python -m tack.cache pack`` after a warm-up run) once, and never
  looks up or writes per-kernel files. In ``write`` mode under MPI only rank 0
  writes.
- The directory must belong to the user and be writable by nobody else:
  entries hold code, so a directory others can write to is refused.

- Entries of every Tack version share a size limit (1 GiB by default). A
  writing process deletes the least recently used past it, scanning the
  directory at most once an hour across processes (the ``.pruned`` mark),
  never per kernel; a hit refreshes its entry's time at most once an hour.

Settings: ``TACK_CACHE_DIR`` (or ``tack.cache.configure(directory=...)``),
``TACK_CACHE`` = ``off`` | ``bundle`` | ``read`` | ``write`` (the default is
``write``, or ``bundle`` under MPI) and ``TACK_CACHE_SIZE`` (``max_size``).
"""

import contextlib
import hashlib
import importlib.metadata
import os
import pickle
import re
import secrets
import struct
import sys
import threading
import time
import warnings
import weakref
from pathlib import Path
from types import ModuleType

FORMAT = 1
MODES = ("off", "bundle", "read", "write")
_MAGIC = b"TACK-CACHE-1\n"
_BUNDLE_MAGIC = b"TACK-BUNDLE-1\n"
# What _Cache names a version's directory and bundle (see _Cache.version).
_VERSION_DIR = re.compile(r"v\d+-[0-9a-f]{16}")
_VERSION_BUNDLE = re.compile(r"bundle-v\d+-[0-9a-f]{16}\.tack")

# The rank variable each MPI launcher sets (Open MPI, MPICH/Intel MPI/PMI,
# PMIx, MVAPICH2, Cray PALS, Slurm, Cray ALPS), with its world size where
# it has one. Slurm's is the job step's: a batch script also has
# SLURM_PROCID=0 and SLURM_NTASKS, but runs once, and only srun sets
# SLURM_STEP_NUM_TASKS.
_LAUNCHERS = (("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"), ("PMI_RANK", "PMI_SIZE"),
              ("PMIX_RANK", None), ("MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE"),
              ("PALS_RANKID", "PALS_NRANKS"), ("SLURM_PROCID", "SLURM_STEP_NUM_TASKS"),
              ("ALPS_APP_PE", None))
_RANK_VARS = tuple(rank for rank, _ in _LAUNCHERS)
_SIZE_VARS = tuple(size for _, size in _LAUNCHERS if size)

# The size limit, and how pruning keeps to it: a writing process scans the
# directory at most once an interval (across processes, by the mark's time)
# and deletes the least recently used entries until 90% of the limit is left.
# A hit refreshes its entry's time at most once an interval.
_DEFAULT_SIZE_LIMIT = 1 << 30
_PRUNE_INTERVAL = 3600.0
_PRUNE_TO = 0.9
_PRUNED_MARK = ".pruned"
_SIZE_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}

_lock = threading.RLock()
_overrides = {}
_state = None          # the resolved _Cache, built on first use
_stats = {}


def _count(name, n=1):
    _stats[name] = _stats.get(name, 0) + n


def stats() -> dict:
    """Counters since the process started (or since ``reset_stats``)."""
    return dict(_stats)


def reset_stats():
    _stats.clear()


# ── Settings ────────────────────────────────────────────────────────────


def mpi_rank():
    """This process's rank in a parallel launch, or None outside one.

    A launcher's rank variable with a world size of one (``srun -n 1``) is
    not a parallel job, and neither is a Slurm batch script outside srun."""
    for rank_var, size_var in _LAUNCHERS:
        rank = os.environ.get(rank_var, "")
        if rank == "":
            continue
        size = os.environ.get(size_var, "") if size_var else ""
        try:
            if size == "" and rank_var == "SLURM_PROCID":
                return None
            if size != "" and int(size) <= 1:
                return None
            return int(rank)
        except ValueError:
            return 0
    return None


def default_directory() -> Path:
    """The per-user cache directory of the platform."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "tack"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "tack" / "Cache"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "tack"


def configure(directory=None, mode=None, max_size=None):
    """Set the cache directory, mode and size limit for this process, over the
    environment.

    Takes effect at the next kernel compiled. ``None`` leaves a setting to
    ``TACK_CACHE_DIR`` / ``TACK_CACHE`` / ``TACK_CACHE_SIZE`` (or their
    defaults). ``max_size`` is bytes, a size such as ``"500M"`` or ``"2G"``,
    or ``"unlimited"``."""
    global _state
    if mode is not None and mode not in MODES:
        raise ValueError(f"cache mode must be one of {', '.join(MODES)}, not {mode!r}")
    if max_size is not None:
        parse_size(max_size)
    with _lock:
        _overrides["directory"] = None if directory is None else Path(directory)
        _overrides["mode"] = mode
        _overrides["max_size"] = max_size
        _state = None


def parse_size(size):
    """Bytes from an int or a size such as ``"500M"``, ``"2GB"`` or ``"1.5G"``
    (binary units); None for ``"unlimited"``."""
    if isinstance(size, int) and not isinstance(size, bool):
        value = size
    else:
        text = str(size).strip().upper().replace(" ", "")
        if text in ("UNLIMITED", "NONE"):
            return None
        match = re.fullmatch(r"(\d+(?:\.\d*)?)([KMGT]?)(?:I?B)?", text)
        if match is None:
            raise ValueError(f"a cache size is bytes, a size such as 500M or 2G, or "
                             f"unlimited, not {size!r}")
        value = int(float(match[1]) * _SIZE_UNITS[match[2]])
    if value <= 0:
        raise ValueError(f"a cache size must be positive, not {size!r}")
    return value


def size_limit():
    """The size limit in bytes as it applies now, or None for no limit."""
    size = _overrides.get("max_size")
    if size is None:
        size = os.environ.get("TACK_CACHE_SIZE", "").strip() or _DEFAULT_SIZE_LIMIT
    return parse_size(size)


def settings() -> tuple:
    """``(directory, mode, rank)`` as they apply now."""
    directory = _overrides.get("directory")
    if directory is None:
        env = os.environ.get("TACK_CACHE_DIR", "")
        directory = Path(env).expanduser() if env else default_directory()
    rank = mpi_rank()
    mode = _overrides.get("mode")
    if mode is None:
        env = os.environ.get("TACK_CACHE", "").strip().lower()
        if env in ("", "default"):
            mode = "bundle" if rank is not None else "write"
        elif env in ("0", "false", "no", "none"):
            mode = "off"
        elif env in MODES:
            mode = env
        else:
            raise ValueError(f"TACK_CACHE must be one of {', '.join(MODES)}, not {env!r}")
    return directory, mode, rank


def _cache():
    """The active cache, or None when caching is off."""
    global _state
    state = _state
    if state is None:
        with _lock:
            if _state is None:
                _state = _Cache(*settings(), max_size=size_limit())
            state = _state
    return state if state.mode != "off" else None


# ── What Tack itself is ─────────────────────────────────────────────────

_TACK_FINGERPRINT = None
_DISTRIBUTIONS = ("tack-core", "tack-vis", "tack-rendering")


def tack_fingerprint() -> str:
    """A hash of the Tack that is running: any change to it misses every entry.

    Installed packages are identified by their ``RECORD`` (which lists every
    file's hash) -- one small read per package. Source trees on the path
    (editable installs, a checkout) are hashed file by file."""
    global _TACK_FINGERPRINT
    if _TACK_FINGERPRINT is None:
        import tack
        h = hashlib.sha256(f"format {FORMAT}; python {sys.version_info[:2]}".encode())
        for name in _DISTRIBUTIONS:
            try:
                dist = importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                continue
            h.update(f"\n{name} {dist.version}\n".encode())
            h.update((dist.read_text("RECORD") or "").encode())
        for path in sorted(set(tack.__path__)):
            if _installed(path):
                continue
            for file in sorted(Path(path).rglob("*.py")):
                h.update(f"\n{file.relative_to(path)}\n".encode())
                h.update(file.read_bytes())
        _TACK_FINGERPRINT = h.hexdigest()
    return _TACK_FINGERPRINT


def _installed(path) -> bool:
    return any(part in ("site-packages", "dist-packages") for part in Path(path).parts)


# ── Fingerprints of what lowering resolved ──────────────────────────────


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(value) -> tuple:
    """What a resolved name is bound to, in a form that means the same in
    another process. ``("opaque", ...)`` for anything that cannot be."""
    from tack.lang.call_bindings import _DYNAMIC, _UNBOUND
    from tack.lang.constant import ArrayConstant, FloatConstant, IntConstant
    from tack.lang.func import Func
    from tack.lang.kernel import Kernel
    from tack.lang.types import ScalarType

    if value is _UNBOUND:
        return ("unbound",)
    if value is _DYNAMIC:
        return ("dynamic",)
    if isinstance(value, Func):
        return func_fingerprint(value)
    if isinstance(value, Kernel):
        return ("kernel", *_function_name(value.func), _digest(value._source))
    if isinstance(value, IntConstant):
        return ("int constant", repr(value.dtype), int(value))
    if isinstance(value, FloatConstant):
        return ("float constant", repr(value.dtype), struct.pack("!d", value).hex())
    if isinstance(value, ArrayConstant):
        return ("array constant", repr(value.dtype), value.shape, repr(value.components))
    if isinstance(value, ModuleType):
        return ("module", value.__name__)
    if isinstance(value, ScalarType):
        return ("dtype", value.name)
    if isinstance(value, float):
        return ("float", struct.pack("!d", value).hex())
    if value is None or isinstance(value, (bool, int, str, complex)):
        return ("value", type(value).__name__, repr(value))
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if (callable(value) and isinstance(module, str) and isinstance(qualname, str)
            and "<locals>" not in qualname):
        return ("object", module, qualname)
    return ("opaque", type(value).__qualname__)


def _function_name(func) -> tuple:
    return (getattr(func, "__module__", "?"), getattr(func, "__qualname__", "?"))


def func_fingerprint(func) -> tuple:
    """A device function: its name and its source."""
    found = getattr(func, "_tack_fingerprint", None)
    if found is None:
        found = ("func", *_function_name(func.func), _digest(func._source))
        func._tack_fingerprint = found
    return found


_CLASS_FINGERPRINTS = weakref.WeakKeyDictionary()


def _class_fingerprint(cls) -> tuple:
    """A template class: its method resolution order by name, its device-function
    methods by source, and what a loop over it runs through."""
    from tack.lang.template_rewrite import template_func_methods

    found = _CLASS_FINGERPRINTS.get(cls)
    if found is None:
        found = _CLASS_FINGERPRINTS[cls] = (
            tuple((k.__module__, k.__qualname__) for k in cls.__mro__),
            tuple((name, func_fingerprint(f))
                  for name, f in sorted(template_func_methods(cls).items())),
            repr(getattr(cls, "__tack_iterate__", None)),
        )
    return found


def _template_content(obj) -> tuple:
    """What of a template object lowering depends on, as ``_template_key`` has
    it, with classes and device functions by content instead of identity."""
    from tack.lang.template_rewrite import (
        classify_template_attrs,
        template_func_attrs,
        template_structure,
    )
    scalars, fields, runtime = classify_template_attrs(obj)
    return (
        _class_fingerprint(type(obj)),
        tuple((k, type(v).__name__,
               struct.pack("!d", v).hex() if isinstance(v, float) else v)
              for k, v in sorted(scalars.items())),
        tuple((k, f.dtype.name, len(getattr(f, "_logical_shape", None) or f.shape),
               getattr(f, "_vector_n", None), getattr(f, "_matrix_shape", None))
              for k, f in sorted(fields.items())),
        tuple(sorted(runtime)),
        tuple((k, func_fingerprint(f)) for k, f in sorted(template_func_attrs(obj).items())),
        tuple((path, _class_fingerprint(cls)) for path, cls in template_structure(obj)),
    )


def _closure_content(func) -> tuple:
    """A kernel's closure cells, so kernels made by one factory over different
    device functions keep separate entries rather than replacing each other."""
    if not getattr(func, "__closure__", None):
        return ()
    content = []
    for name, cell in zip(func.__code__.co_freevars, func.__closure__):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        found = fingerprint(value)
        content.append((name, found if found[0] != "opaque" else ("opaque",)))
    return tuple(content)


def lowered_key(kernel, vector_fields, texture_fields, template_args, debug=False) -> str:
    """The key of a kernel specialization's lowered IR. Debug mode lowers
    with bounds checks, so its IR, and every variant built from it, is
    stored apart."""
    result = kernel._result[0].name if kernel._result else None
    content = (
        "ir", FORMAT, tack_fingerprint(),
        *_function_name(kernel.func), _digest(kernel._source), result,
        _closure_content(kernel.func),
        tuple(sorted((vector_fields or {}).items())),
        tuple(sorted((texture_fields or {}).items())),
        tuple((idx, name, _template_content(obj))
              for idx, (name, obj) in sorted((template_args or {}).items())),
        bool(debug),
    )
    return _digest(repr(content))


# ── Recording what lowering resolves ────────────────────────────────────


def _owners(kernel, template_args) -> list:
    """The functions whose namespaces lowering resolves names in, before any
    device function is found: the kernel, and its templates' methods and
    function attributes. Recorded queries refer to them by position in this
    list, extended by each device function a query resolves to."""
    from tack.lang.template_rewrite import (
        template_func_attrs,
        template_func_methods,
        template_tree,
    )
    owners = [kernel.func]
    for _, (_, obj) in sorted((template_args or {}).items()):
        for _, node in template_tree(obj):
            owners.extend(f.func for _, f in sorted(template_func_methods(type(node)).items()))
        owners.extend(f.func for _, f in sorted(template_func_attrs(obj).items()))
    return owners


class _Recorder:
    def __init__(self, owners):
        self.owners = list(owners)
        self.index = {}
        for i, fn in enumerate(self.owners):
            self.index.setdefault(id(fn), i)
        self.queries = []
        self.seen = set()
        self.cacheable = True

    def note(self, owner, node, value):
        """Lowering resolved ``node`` in ``owner``'s namespace to ``value``."""
        import ast

        if getattr(node, "_tack_template_call", False):
            return      # resolved from the template, which the key covers
        names = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return      # not a name: lowering treats it by its syntax alone
        names.append(node.id)
        chain = ".".join(reversed(names))
        found = fingerprint(value)
        if found == ("dynamic",):
            return      # a local of the function: its source decides
        ref = None if owner is None else self.index.get(id(owner))
        if ref is None or found[0] == "opaque":
            self.cacheable = False
            return
        if (ref, chain) in self.seen:
            return
        self.seen.add((ref, chain))
        new = None
        if found[0] == "func" and id(value.func) not in self.index:
            new = len(self.owners)
            self.owners.append(value.func)
            self.index[id(value.func)] = new
        self.queries.append((ref, chain, found, new))


_recording = threading.local()


def recorder():
    """The recorder of the lowering in progress on this thread, if any."""
    return getattr(_recording, "active", None)


@contextlib.contextmanager
def _record(owners):
    previous = recorder()
    rec = _recording.active = _Recorder(owners)
    try:
        yield rec
    finally:
        _recording.active = previous


def _lookup(func, chain):
    """``chain`` resolved in ``func``'s namespace, as ``CallBindings`` does."""
    from tack.lang.call_bindings import _DYNAMIC, _UNBOUND

    names = chain.split(".")
    nonlocals = {}
    if func.__closure__:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            try:
                nonlocals[name] = cell.cell_contents
            except ValueError:
                pass
    value = nonlocals.get(names[0], func.__globals__.get(names[0], _UNBOUND))
    for name in names[1:]:
        value = vars(value).get(name, _UNBOUND) if isinstance(value, ModuleType) else _DYNAMIC
    return value


def _still_valid(queries, owners) -> bool:
    """Whether every recorded name still resolves to what it did."""
    owners = list(owners)
    for ref, chain, found, new in queries:
        if ref >= len(owners):
            return False
        value = _lookup(owners[ref], chain)
        if fingerprint(value) != found:
            return False
        if new is not None:
            if new != len(owners):
                return False
            owners.append(value.func)
    return True


def lowered(kernel, vector_fields, texture_fields, template_args, lower, debug=False):
    """``lower()``, or the lowered IR a previous process stored for the same
    content. ``lower`` is the kernel's own transform; it runs while the names
    it resolves are recorded, so the entry can be checked when it is read."""
    cache = _cache()
    if cache is None or getattr(kernel.func, "__globals__", None) is None:
        return lower()
    try:
        key = lowered_key(kernel, vector_fields, texture_fields, template_args, debug)
        owners = _owners(kernel, template_args)
    except Exception:   # an attribute the key cannot describe: compile as usual
        _count("ir_unkeyed")
        return lower()
    entry = cache.get(key, "ir")
    if entry is not None:
        try:
            if _still_valid(entry["queries"], owners):
                module = pickle.loads(entry["module"])
                _mark(module, key, entry["deps"])
                _count("ir_hit")
                return module
            _count("ir_stale")
        except Exception:
            _count("bad")
    else:
        _count("ir_miss")
    with _record(owners) as rec:
        module = lower()
    if rec.cacheable:
        deps = _digest(repr(rec.queries))
        _mark(module, key, deps)
        if cache.writes:
            try:
                blob = pickle.dumps(module, protocol=5)
            except Exception:
                _count("ir_unpicklable")
            else:
                cache.put(key, "ir", {"key": key, "queries": rec.queries, "deps": deps,
                                      "module": blob})
    else:
        _count("ir_uncacheable")
    return module


def _mark(module, key, deps):
    for function in module.functions:
        function._tack_disk = (key, deps)


# ── Compiled variants ───────────────────────────────────────────────────


def variant_key(backend, template, probe, vector_fields, shape_sig, disjoint):
    """The key of a compiled variant, or None if it cannot be stored."""
    if _cache() is None:
        return None
    lowered_entry = getattr(template, "_tack_disk", None)
    if lowered_entry is None:
        return None
    target = backend.cache_target()
    if target is None:
        return None
    params = tuple(
        (p.name, getattr(p.type_annotation, "name", repr(p.type_annotation)),
         p._is_field, p._is_texture, p._texture_shape)
        for p in probe.params)
    content = ("variant", FORMAT, lowered_entry, params,
               tuple(sorted((vector_fields or {}).items())), shape_sig, disjoint,
               backend.name, target)
    return _digest(repr(content))


def load_variant(key):
    cache = _cache()
    entry = cache.get(key, "variant") if cache is not None and key else None
    _count("variant_hit" if entry is not None else "variant_miss")
    return entry


def store_variant(key, entry):
    cache = _cache()
    if cache is not None and key and cache.writes:
        entry = dict(entry, key=key)
        cache.put(key, "variant", entry)


# ── Storage ─────────────────────────────────────────────────────────────


def _encode(entry) -> bytes:
    payload = pickle.dumps(entry, protocol=5)
    return _MAGIC + hashlib.sha256(payload).digest() + payload


def _decode(data, key):
    """The entry in ``data`` if it is whole and is ``key``'s, else None."""
    if not data.startswith(_MAGIC):
        return None
    start = len(_MAGIC)
    digest, payload = data[start:start + 32], data[start + 32:]
    if hashlib.sha256(payload).digest() != digest:
        return None
    entry = pickle.loads(payload)
    if not isinstance(entry, dict) or entry.get("key") != key:
        return None
    return entry


def _trusted(path) -> bool:
    """Owned by this user and writable by nobody else (POSIX)."""
    if not hasattr(os, "getuid"):
        return True
    st = os.stat(path)
    return st.st_uid == os.getuid() and not st.st_mode & 0o022


def _write_atomically(path, data):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class _Cache:
    """One directory, one mode, one optional bundle, for this process."""

    def __init__(self, directory, mode, rank, max_size=None):
        self.directory = Path(directory)
        self.mode = mode
        self.rank = rank
        self.max_size = max_size
        self.version = _version()
        self.root = self.directory / self.version
        self.bundle_path = self.directory / f"bundle-{self.version}.tack"
        self.bundle = None          # key -> entry bytes, once read
        self.files = mode in ("read", "write")
        self.writes = mode == "write" and rank in (None, 0)
        self._checked = False
        self._warned = False
        self._next_prune = 0.0      # time.monotonic() of the next look at the mark

    def _warn(self, message):
        if not self._warned:
            self._warned = True
            warnings.warn(f"Tack's kernel cache: {message}", RuntimeWarning, stacklevel=4)

    def _usable(self) -> bool:
        """Whether the directory may be used: created if writing, and trusted."""
        if self._checked:
            return self.files
        self._checked = True
        try:
            if self.writes:
                # The directory too, not just the version under it: made by
                # parents=True it would take the umask's mode, which is
                # group-writable under the common 002, and be refused below.
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                self.root.mkdir(exist_ok=True, mode=0o700)
            if not self.directory.exists():
                self.files = False
                return False
            for path in (self.directory, self.root):
                if path.exists() and not _trusted(path):
                    self._warn(f"{path} belongs to another user or others can write to it; "
                               f"not used. Choose a directory of your own with TACK_CACHE_DIR.")
                    self.files = self.writes = False
                    return False
        except OSError as e:
            self._warn(f"cannot use {self.directory} ({e}); compiling without it.")
            self.files = self.writes = False
        return self.files

    def _bundle(self) -> dict:
        if self.bundle is None:
            self.bundle = {}
            try:
                if self.bundle_path.exists():
                    if _trusted(self.bundle_path):
                        self.bundle = read_bundle(self.bundle_path.read_bytes())
                    else:
                        self._warn(f"{self.bundle_path} belongs to another user or others "
                                   f"can write to it; not used.")
            except (OSError, ValueError, pickle.UnpicklingError) as e:
                self._warn(f"ignoring the unreadable bundle {self.bundle_path} ({e}).")
            _count("bundle_entries", len(self.bundle))
        return self.bundle

    def get(self, key, kind):
        data = self._bundle().get(f"{key}.{kind}")
        if data is None and self.files and self._usable():
            path = self.root / key[:2] / f"{key}.{kind}"
            try:
                with open(path, "rb") as f:
                    data = f.read()
                    used = os.fstat(f.fileno()).st_mtime
            except OSError:
                return None
            if self.writes and time.time() - used > _PRUNE_INTERVAL:
                # Pruning goes by modification time: a hit is a use.
                with contextlib.suppress(OSError):
                    os.utime(path)
        if data is None:
            return None
        try:
            entry = _decode(data, key)
        except Exception:
            entry = None
        if entry is None:
            _count("bad")
        return entry

    def put(self, key, kind, entry):
        if not self.writes:
            return
        self._usable()
        if not self.writes:
            return
        folder = self.root / key[:2]
        data = _encode(entry)
        try:
            try:
                folder.mkdir(exist_ok=True, mode=0o700)
                _write_atomically(folder / f"{key}.{kind}", data)
            except FileNotFoundError:
                # Another process's prune removed the emptied folder.
                self.root.mkdir(exist_ok=True, mode=0o700)
                folder.mkdir(exist_ok=True, mode=0o700)
                _write_atomically(folder / f"{key}.{kind}", data)
            _count(f"{kind}_write")
        except OSError as e:
            self.writes = False
            self._warn(f"cannot write to {self.root} ({e}); no more entries are stored "
                       f"by this process.")
            return
        self._keep_to_size()

    def _keep_to_size(self):
        """Prune to the size limit if no process has in the last interval."""
        now = time.monotonic()
        if self.max_size is None or now < self._next_prune:
            return
        self._next_prune = now + _PRUNE_INTERVAL
        mark = self.directory / _PRUNED_MARK
        try:
            if time.time() - mark.stat().st_mtime < _PRUNE_INTERVAL:
                return
        except FileNotFoundError:
            pass
        except OSError:
            return
        try:
            # Marked first, so processes starting meanwhile do not scan too.
            mark.touch(mode=0o600)
            _count("prunes")
            _count("pruned", prune(self.directory, self.max_size)[0])
        except OSError as e:
            self._warn(f"cannot prune {self.directory} ({e}).")


# ── Bundles ─────────────────────────────────────────────────────────────


def read_bundle(data: bytes) -> dict:
    if not data.startswith(_BUNDLE_MAGIC):
        raise ValueError("not a Tack cache bundle")
    start = len(_BUNDLE_MAGIC)
    digest, payload = data[start:start + 32], data[start + 32:]
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("the bundle is damaged (checksum mismatch)")
    entries = pickle.loads(payload)
    if not isinstance(entries, dict):
        raise ValueError("not a Tack cache bundle")
    return entries


def pack(directory=None) -> tuple:
    """Gather this Tack version's entries in ``directory`` into one bundle file.

    Returns ``(path, number of entries)``."""
    directory = Path(directory) if directory is not None else settings()[0]
    cache = _Cache(directory, "read", None)
    entries = {}
    if cache.root.exists():
        if not _trusted(cache.root):
            raise PermissionError(f"{cache.root} belongs to another user or others can "
                                  "write to it")
        for file in sorted(cache.root.glob("*/*")):
            if file.suffix in (".ir", ".variant") and not file.name.startswith("."):
                entries[file.name] = file.read_bytes()
    payload = pickle.dumps(entries, protocol=5)
    _write_atomically(cache.bundle_path,
                      _BUNDLE_MAGIC + hashlib.sha256(payload).digest() + payload)
    return cache.bundle_path, len(entries)


def load_bundle(path=None, comm=None) -> int:
    """Read a bundle into this process's cache, once.

    With an mpi4py communicator, rank 0 of ``comm`` reads the file and
    broadcasts it, so the other ranks never touch the file system. Call it
    on every rank of ``comm`` (it is collective), before the first kernel.
    Returns the number of entries."""
    cache = _cache()
    if cache is None:
        return 0
    path = Path(path) if path is not None else cache.bundle_path
    data = None
    if comm is None or comm.Get_rank() == 0:
        try:
            if _trusted(path):
                data = path.read_bytes()
        except OSError:
            data = None
    if comm is not None:
        data = comm.bcast(data, root=0)
    entries = read_bundle(data) if data is not None else {}
    with _lock:
        cache.bundle = entries
    _count("bundle_entries", len(entries))
    return len(entries)


def _version():
    """The name of this Tack's directory under the cache directory."""
    return f"v{FORMAT}-{tack_fingerprint()[:16]}"


def _version_dirs(directory) -> list:
    """Every Tack version's directory: only names this module makes, since
    the directory may be shared with other files (a scratch directory), and
    a "v..." name may be a venv."""
    return [p for p in directory.iterdir()
            if p.is_dir() and not p.is_symlink() and _VERSION_DIR.fullmatch(p.name)]


def _scan(path) -> list:
    """A directory's entries, or none if another process's prune removed it."""
    try:
        with os.scandir(path) as found:
            return list(found)
    except FileNotFoundError:
        return []


def prune(directory=None, max_size=None) -> tuple:
    """Delete the least recently used entries, of every Tack version, until
    they take at most 90% of ``max_size`` (by default the size limit in
    effect), and temporary files writers left an interval ago or more.
    Bundles are kept. Returns ``(files removed, bytes of entries left)``."""
    directory = Path(directory) if directory is not None else settings()[0]
    limit = parse_size(max_size) if max_size is not None else size_limit()
    if not directory.exists():
        return 0, 0
    if not _trusted(directory):
        raise PermissionError(f"{directory} belongs to another user or others can write to it")
    now = time.time()
    removed = 0
    entries = []
    versions = _version_dirs(directory)
    for version in versions:
        for folder in _scan(version):
            if not folder.is_dir(follow_symlinks=False):
                continue
            for item in _scan(folder.path):
                try:
                    if not item.is_file(follow_symlinks=False):
                        continue
                    st = item.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if item.name.startswith(".") and item.name.endswith(".tmp"):
                    if now - st.st_mtime > _PRUNE_INTERVAL:
                        with contextlib.suppress(FileNotFoundError):
                            os.unlink(item.path)
                            removed += 1
                elif item.name.endswith((".ir", ".variant")):
                    entries.append((st.st_mtime, st.st_size, item.path))
    total = sum(size for _, size, _ in entries)
    if limit is not None and total > limit:
        goal = int(limit * _PRUNE_TO)
        for _, size, path in sorted(entries):
            if total <= goal:
                break
            try:
                os.unlink(path)
                removed += 1
            except FileNotFoundError:
                pass
            except OSError:
                continue
            total -= size
    # Folders pruning emptied, and the versions left with none. Another
    # process's write recreates what it needs; rmdir refuses a folder in use.
    for version in versions:
        for folder in _scan(version):
            with contextlib.suppress(OSError):
                os.rmdir(folder.path)
        if version.name != _version():
            with contextlib.suppress(OSError):
                version.rmdir()
    return removed, total


def clear(directory=None, all_versions=False) -> int:
    """Delete this Tack version's entries and bundle (every version's with
    ``all_versions``). Returns the number of files removed."""
    import shutil

    directory = Path(directory) if directory is not None else settings()[0]
    cache = _Cache(directory, "read", None)
    removed = 0
    if not directory.exists():
        return 0
    if all_versions:
        targets = _version_dirs(directory) + [
            p for p in directory.iterdir() if p.is_file()
            and (_VERSION_BUNDLE.fullmatch(p.name) or p.name == _PRUNED_MARK)]
    else:
        targets = [cache.root, cache.bundle_path]
    for target in targets:
        if target.is_dir():
            removed += sum(1 for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
            removed += 1
    return removed


def describe(directory=None) -> dict:
    """The settings in effect and what the directory holds for this Tack."""
    found_dir, mode, rank = settings()
    directory = Path(directory) if directory is not None else found_dir
    cache = _Cache(directory, mode, rank)
    files = [p for p in cache.root.rglob("*") if p.is_file()] if cache.root.exists() else []
    every = ([p for v in _version_dirs(directory) for p in v.rglob("*") if p.is_file()]
             if directory.exists() else [])
    return {
        "directory": str(directory),
        "mode": mode,
        "mpi rank": rank,
        "tack version key": cache.version,
        "entries": sum(1 for p in files if p.suffix in (".ir", ".variant")),
        "bytes": sum(p.stat().st_size for p in files),
        "bytes, every version": sum(p.stat().st_size for p in every),
        "size limit": size_limit(),
        "bundle": str(cache.bundle_path) if cache.bundle_path.exists() else None,
    }
