"""Compiler selection, tool failures and real host preflight coverage."""

import subprocess
from pathlib import Path

import compiler_tools as tools
import pytest


@pytest.fixture(autouse=True)
def clear_probe_cache():
    tools._probe_clang.cache_clear()
    yield
    tools._probe_clang.cache_clear()


@pytest.fixture
def compiler_env(monkeypatch, tmp_path):
    for name in ("TACK_CLANG", "TACK_CLANGXX", "TACK_REQUIRE_CLANG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    return tmp_path


def _driver(root, name="clang"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder for controlled subprocess tests\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def successful_probe(monkeypatch):
    calls = []

    def run(command, **kwargs):
        assert kwargs == {"capture_output": True, "text": True, "timeout": 30}
        calls.append(command)
        return subprocess.CompletedProcess(command, 0,
                                           "clang version 15.0.7\n" if "--version" in command else "", "")

    monkeypatch.setattr(tools.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("mode", ["opencl", "cxx", "ubsan"])
def test_host_compiler_preflight(mode):
    """CI runs this first with TACK_REQUIRE_CLANG=1; developers may skip."""
    print(f"{mode}: {tools.require_clang(mode)}")


@pytest.mark.parametrize("mode", ["cxx", "ubsan"])
def test_absent_cpp_tools_skip_only_when_optional(compiler_env, monkeypatch, mode):
    with pytest.raises(pytest.skip.Exception, match="none of clang"):
        tools.require_clang(mode)
    monkeypatch.setenv("TACK_REQUIRE_CLANG", "1")
    with pytest.raises(pytest.fail.Exception, match="none of clang"):
        tools.require_clang(mode)


def test_outside_path_override_wins_over_path_compiler(
        compiler_env, monkeypatch, successful_probe):
    _driver(compiler_env)
    selected = _driver(compiler_env / "outside PATH")
    monkeypatch.setenv("TACK_CLANG", selected)
    assert tools.require_clang() == selected
    assert {command[0] for command in successful_probe} == {selected}


@pytest.mark.parametrize("name", ["clang", "clang-15", "clang.exe", "clang-15.exe"])
def test_cpp_sibling_tracks_explicit_clang(compiler_env, monkeypatch, successful_probe, name):
    _driver(compiler_env, "clang++")  # A different installation must not win.
    selected = _driver(compiler_env / "outside", name)
    sibling = _driver(compiler_env / "outside", name.replace("clang", "clang++", 1))
    monkeypatch.setenv("TACK_CLANG", selected)
    assert tools.require_clang("cxx") == sibling
    assert {command[0] for command in successful_probe} == {sibling}


def test_cpp_override_wins_even_when_c_override_is_invalid(
        compiler_env, monkeypatch, successful_probe):
    selected = _driver(compiler_env / "outside", "custom-cxx")
    monkeypatch.setenv("TACK_CLANG", str(compiler_env / "missing-clang"))
    monkeypatch.setenv("TACK_CLANGXX", selected)
    assert tools.require_clang("cxx") == selected


@pytest.mark.parametrize("setting,mode", [("TACK_CLANG", "opencl"), ("TACK_CLANGXX", "cxx")])
@pytest.mark.parametrize("problem", ["absent", "nonexecutable", "directory"])
def test_bad_explicit_choice_fails_without_required_flag_or_fallback(
        compiler_env, monkeypatch, setting, mode, problem):
    _driver(compiler_env, "clang")
    _driver(compiler_env, "clang++")
    chosen = compiler_env / "chosen"
    if problem == "nonexecutable":
        chosen.write_text("not executable")
    elif problem == "directory":
        chosen.mkdir()
    monkeypatch.setenv(setting, str(chosen))
    with pytest.raises(pytest.fail.Exception, match=setting):
        tools.require_clang(mode)


@pytest.mark.parametrize("name", ["clang-15", "wrapper"])
def test_missing_cpp_sibling_never_falls_back_to_path(
        compiler_env, monkeypatch, name):
    _driver(compiler_env, "clang++")
    monkeypatch.setenv("TACK_CLANG", _driver(compiler_env / "outside", name))
    with pytest.raises(pytest.fail.Exception, match="TACK_CLANGXX"):
        tools.require_clang("cxx")


def test_symlink_keeps_selected_installation_sibling(compiler_env, monkeypatch):
    actual = _driver(compiler_env / "real", "clang-15")
    selected = compiler_env / "selected" / "clang"
    selected.parent.mkdir()
    selected.symlink_to(actual)
    sibling = _driver(selected.parent, "clang++")
    monkeypatch.setenv("TACK_CLANG", str(selected))
    assert tools._resolve_clang(cxx=True) == sibling


def test_versioned_drivers_are_found_without_unversioned_names(
        compiler_env, successful_probe):
    clang = _driver(compiler_env, "clang-15")
    clangxx = _driver(compiler_env, "clang++-15")
    assert tools.require_clang() == clang
    assert tools.require_clang("cxx") == clangxx


@pytest.mark.parametrize("mode,stage,diagnostic", [
    ("opencl", "version", "libz3.so could not be loaded"),
    ("opencl", "identity", "not a compiler"),
    ("opencl", "compile", "OpenCL builtins are missing"),
    ("cxx", "compile", "cmath header not found"),
    ("ubsan", "compile", "libclang_rt.ubsan_standalone not found"),
    ("ubsan", "execute", "sanitizer runtime could not be loaded"),
])
@pytest.mark.parametrize("policy", ["optional", "required", "explicit"])
def test_unusable_tools_have_preflight_diagnostics(
        compiler_env, monkeypatch, mode, stage, diagnostic, policy):
    driver = _driver(compiler_env, "clang" if mode == "opencl" else "clang++")
    if policy == "required":
        monkeypatch.setenv("TACK_REQUIRE_CLANG", "1")
    elif policy == "explicit":
        monkeypatch.setenv("TACK_CLANG" if mode == "opencl" else "TACK_CLANGXX", driver)

    def run(command, **kwargs):
        current = "version" if "--version" in command else (
            "compile" if command[0] == driver else "execute")
        if stage == "identity" and current == "version":
            return subprocess.CompletedProcess(command, 0, diagnostic, "")
        if current == stage:
            return subprocess.CompletedProcess(command, 1, "", diagnostic)
        return subprocess.CompletedProcess(command, 0,
                                           "clang version 15.0.7" if current == "version" else "", "")

    monkeypatch.setattr(tools.subprocess, "run", run)
    exception = pytest.skip.Exception if policy == "optional" else pytest.fail.Exception
    with pytest.raises(exception) as info:
        tools.require_clang(mode)
    expected = "did not identify" if stage == "identity" else diagnostic
    assert expected in str(info.value)


@pytest.mark.parametrize("problem", ["launch", "timeout"])
def test_preflight_reports_launch_errors(compiler_env, monkeypatch, problem):
    monkeypatch.setenv("TACK_CLANG", _driver(compiler_env))

    def run(command, **kwargs):
        if problem == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        raise OSError("cannot execute this binary format")

    monkeypatch.setattr(tools.subprocess, "run", run)
    with pytest.raises(pytest.fail.Exception, match="cannot run"):
        tools.require_clang()


@pytest.mark.parametrize("change", ["PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SDKROOT",
                                    "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "DEVELOPER_DIR", "driver"])
def test_probe_cache_tracks_execution_environment_and_driver(
        compiler_env, monkeypatch, successful_probe, change):
    driver = _driver(compiler_env)
    monkeypatch.setenv("TACK_CLANG", driver)
    tools.require_clang()
    first = len(successful_probe)
    tools.require_clang()
    assert len(successful_probe) == first
    if change == "driver":
        with Path(driver).open("a") as stream:
            stream.write("changed executable")
    else:
        monkeypatch.setenv(change, "a different tool environment")
    tools.require_clang()
    assert len(successful_probe) == first * 2


def test_unknown_mode_is_a_programming_error():
    with pytest.raises(ValueError, match="unknown compiler mode"):
        tools.require_clang("cuda")
