"""Host compiler preflight shared by syntax and sanitized execution checks.

Unconfigured developer machines may skip absent or unusable tooling. An
explicit compiler choice, or TACK_REQUIRE_CLANG=1, must fail instead. Never
use a different PATH compiler to rescue an invalid explicit selection.
"""

import os
import re
import shlex
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

_CLANG_NAMES = ("clang", *(f"clang-{version}" for version in range(23, 13, -1)))
_PROBE_ENV = ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SDKROOT",
              "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "DEVELOPER_DIR")
_MODES = ("opencl", "cxx", "ubsan")


class _Unavailable(RuntimeError):
    def __init__(self, reason, *, configured=False):
        super().__init__(reason)
        self.configured = configured


def _explicit_compiler(setting):
    value = os.environ[setting]
    found = shutil.which(value)
    if not found:
        raise _Unavailable(f"{setting}={value!r} is not an executable", configured=True)
    return str(Path(found).absolute())


def _resolve_clang(cxx=False):
    setting = "TACK_CLANGXX" if cxx else "TACK_CLANG"
    if os.environ.get(setting):
        return _explicit_compiler(setting)
    if cxx and os.environ.get("TACK_CLANG"):
        compiler = Path(_explicit_compiler("TACK_CLANG"))
        match = re.fullmatch(r"clang((?:-\d+)?(?:\.exe)?)", compiler.name)
        if not match:
            raise _Unavailable(
                f"cannot derive clang++ from TACK_CLANG={str(compiler)!r}; "
                "set TACK_CLANGXX to the matching C++ driver", configured=True,
            )
        sibling = compiler.with_name("clang++" + match[1])
        found = shutil.which(str(sibling))
        if not found:
            raise _Unavailable(
                f"TACK_CLANG has no executable C++ sibling at {sibling}; "
                "set TACK_CLANGXX explicitly", configured=True,
            )
        return str(Path(found).absolute())
    names = tuple(name.replace("clang", "clang++", 1) for name in _CLANG_NAMES) if cxx else _CLANG_NAMES
    for name in names:
        found = shutil.which(name)
        if found:
            return str(Path(found).absolute())
    raise _Unavailable(f"none of {', '.join(names)} was found on PATH")


def _run_probe(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _Unavailable(f"cannot run {shlex.join(command)}: {exc}") from exc
    if result.returncode:
        raise _Unavailable(
            f"{shlex.join(command)} exited {result.returncode}:\n"
            f"{result.stderr or result.stdout}"
        )
    return result


@lru_cache(maxsize=32)
def _probe_clang(compiler, mode, environment, identity):
    """Cache only tool probes, keyed by driver and its execution environment.

    Generated kernels are always compiled by their tests. PATH, library/SDK
    paths, header/toolchain paths and executable metadata belong in the key.
    Changing them must not reuse a preflight from a different configuration.
    """
    try:
        version = _run_probe([compiler, "--version"])
        if "clang version" not in (version.stdout + version.stderr).lower():
            raise _Unavailable(f"{compiler} --version did not identify a Clang compiler")
        with TemporaryDirectory(prefix="tack-compiler-") as directory:
            root = Path(directory)
            if mode == "opencl":
                source = root / "probe.cl"
                source.write_text(
                    "__kernel void probe(__global float* out) {\n"
                    "    out[get_global_id(0)] = 0.0f;\n}\n"
                )
                command = [compiler, "-target", "x86_64-unknown-linux-gnu", "-x", "cl",
                           "-cl-std=CL2.0", "-fsyntax-only", str(source)]
            else:
                source = root / "probe.cpp"
                source.write_text(
                    "#include <cmath>\n#include <cstdint>\n#include <thread>\n"
                    "int main() {\n"
                    "    volatile int one = 1; int result = 0;\n"
                    "    std::thread worker([&] { result = one + 1; });\n"
                    "    worker.join();\n    return result != 2;\n}\n"
                )
                command = [compiler, "-std=c++17"]
                if mode == "cxx":
                    command += ["-fsyntax-only", str(source)]
                else:
                    binary = root / ("probe.exe" if os.name == "nt" else "probe")
                    command += ["-O2", "-pthread", "-fsanitize=undefined",
                                "-fno-sanitize-recover=all", str(source), "-o", str(binary)]
            _run_probe(command)
            if mode == "ubsan":
                _run_probe([str(binary)])
    except _Unavailable as exc:
        return str(exc)
    return None


def require_clang(mode="opencl"):
    """Return a usable driver for OpenCL syntax, C++ syntax or UBSan execution.

    TACK_CLANG overrides C/OpenCL discovery. TACK_CLANGXX overrides C++
    discovery; otherwise an explicit TACK_CLANG selects its matching sibling
    without resolving symlinks or falling back to another installation.
    TACK_REQUIRE_CLANG=1 requires every used mode, including headers and the
    UBSan link/runtime, rather than merely an executable with the right name.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown compiler mode: {mode}")
    required = os.environ.get("TACK_REQUIRE_CLANG", "") not in ("", "0")
    configured = bool(os.environ.get("TACK_CLANG")) or (
        mode != "opencl" and bool(os.environ.get("TACK_CLANGXX")))
    try:
        compiler = _resolve_clang(cxx=mode != "opencl")
        try:
            info = os.stat(compiler)
        except OSError as exc:
            raise _Unavailable(f"cannot inspect {compiler}: {exc}") from exc
        error = _probe_clang(compiler, mode,
                             tuple(os.environ.get(name) for name in _PROBE_ENV),
                             (info.st_mtime_ns, info.st_size))
        if error:
            raise _Unavailable(f"{mode} preflight failed: {error}")
    except _Unavailable as exc:
        problem, bad_choice = str(exc), exc.configured
    else:
        return compiler
    # Report the tooling diagnostic without chaining the helper's traceback.
    message = (f"Clang is required for host {mode} checks: {problem}. "
               "Configure TACK_CLANG/TACK_CLANGXX; TACK_REQUIRE_CLANG=1 "
               "makes missing or unusable tools a failure.")
    if required or configured or bad_choice:
        pytest.fail(message, pytrace=False)
    pytest.skip(message)
