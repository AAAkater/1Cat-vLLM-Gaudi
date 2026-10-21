# SPDX-License-Identifier: Apache-2.0
"""Pure-CPU contract tests for the unified DeepSeek V4.1 runtime builder."""

from __future__ import annotations

import builtins
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/build_deepseek_v41_runtime.py"
NATIVE_BUILDER = ROOT / "tools/build_deepseek_v41.py"
BRIDGE_BUILDER = ROOT / "tools/communication/build_tp2_fused_ar_norm_bridge.py"
BRIDGE_MODULE = "tp2_fused_ar_norm_bridge"
BRIDGE_BINARY_NAME = f"{BRIDGE_MODULE}{importlib.machinery.EXTENSION_SUFFIXES[0]}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def stale_aggregate(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "deepseek_v41_runtime_build.json"
    path.write_text('{"stale": true}\n')
    return path


def runtime_tool():
    spec = importlib.util.spec_from_file_location("deepseek_v41_runtime_build_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_kind(command: list[str]) -> str:
    if len(command) > 1:
        name = Path(command[1]).name
        if name == NATIVE_BUILDER.name:
            return "native"
        if name == BRIDGE_BUILDER.name:
            return "bridge"
    if command and command[0] == "git":
        return "git"
    return "other"


@pytest.fixture
def inputs(tmp_path):
    bridge_source = tmp_path / "bridge-source"
    bridge_source.mkdir()
    cmake_build = tmp_path / "bridge-cmake"
    cmake_build.mkdir()
    libraries = tmp_path / "libraries"
    libraries.mkdir()
    synapse = libraries / "libSynapse.so"
    synapse.write_bytes(b"fixture synapse")
    hcl = libraries / "libhcl.so"
    hcl.write_bytes(b"fixture hcl")
    backend = libraries / "libhabana_pytorch_backend.so"
    backend.write_bytes(b"fixture backend")
    native_hccl = libraries / "_hccl_eager_C.so"
    native_hccl.write_bytes(b"fixture native hccl")
    build_root = tmp_path / "build-root"
    return SimpleNamespace(
        bridge_source=bridge_source,
        cmake_build=cmake_build,
        synapse=synapse,
        hcl=hcl,
        backend=backend,
        native_hccl=native_hccl,
        build_root=build_root,
        bridge_build=build_root / "bridge",
        output=tmp_path / "output",
    )


def runtime_argv(inputs, jobs: int = 5) -> list[str]:
    return [
        "--bridge-source", str(inputs.bridge_source),
        "--bridge-cmake-build-directory", str(inputs.cmake_build),
        "--synapse-library", str(inputs.synapse),
        "--hcl-library", str(inputs.hcl),
        "--build-root", str(inputs.build_root),
        "--output-dir", str(inputs.output),
        "--jobs", str(jobs),
    ]


def stage_native(output: Path, *, declared_sha256: str | None = None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    binary = output / "libdeepseek_v4_gaudi2_kernels.so"
    binary.write_bytes(b"fixture native runtime")
    digest = declared_sha256 or sha256_file(binary)
    sources = {"csrc/deepseek_v4/pytorch/hpu_dsv4_fixture.cpp": sha256_bytes(b"fixture source")}
    kernel_manifest = {"sources": sources, "binaries": {binary.name: digest}}
    host_manifest = dict(kernel_manifest, host_gather_abi_version=1)
    write_json(output / "deepseek_v4_build.json", kernel_manifest)
    write_json(output / "deepseek_v41_build.json", host_manifest)
    return binary


def dependency_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def stage_bridge(build: Path, inputs, *, declared_sha256: str | None = None) -> tuple[Path, Path]:
    build.mkdir(parents=True, exist_ok=True)
    binary = build / BRIDGE_BINARY_NAME
    binary.write_bytes(b"fixture bridge runtime")
    abi = {
        "schema": 1,
        "binary_sha256": declared_sha256 or sha256_file(binary),
        "adapter_sources": {"tools/communication/tp2_fused_ar_norm_bridge.cpp": sha256_bytes(b"adapter")},
        "bridge_dependencies": {
            "backend": dependency_record(inputs.backend),
            "native_hccl": dependency_record(inputs.native_hccl),
        },
        "native_dependencies": {
            "synapse": dependency_record(inputs.synapse),
            "hcl": dependency_record(inputs.hcl),
        },
        "tp8_prepared_group_version": 1,
        "tp8_native_graph_available": True,
        "device_engram_shared_mapping_version": 1,
        "tp4_index_gather_pair_version": 1,
    }
    abi_path = binary.with_suffix(".abi.json")
    write_json(abi_path, abi)
    return binary, abi_path


def successful_build_run(inputs, calls: list, *, native_sha256=None, bridge_sha256=None):
    def run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        kind = command_kind(command)
        if kind == "native":
            stage_native(inputs.output, declared_sha256=native_sha256)
        elif kind == "bridge":
            stage_bridge(inputs.bridge_build, inputs, declared_sha256=bridge_sha256)
        elif kind == "git":
            return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n", stderr="")
        else:  # pragma: no cover - guards against an accidental real subprocess
            raise AssertionError(f"Unexpected subprocess: {command}")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return run


def test_help_does_not_import_torch(monkeypatch, capsys):
    real_import = builtins.__import__

    def reject_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("--help imported torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    tool = runtime_tool()
    with pytest.raises(SystemExit) as exit_info:
        tool.parse_args(["--help"])

    assert exit_info.value.code == 0
    assert "--bridge-source" in capsys.readouterr().out


def test_jobs_must_be_positive(inputs, capsys):
    tool = runtime_tool()
    assert tool.parse_args([*runtime_argv(inputs), "--jobs", "3"]).jobs == 3
    for invalid in ("0", "-1"):
        with pytest.raises(SystemExit) as exit_info:
            tool.parse_args([*runtime_argv(inputs), "--jobs", invalid])
        assert exit_info.value.code == 2
    assert "--jobs must be positive" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "error"),
    [
        ("--bridge-source", NotADirectoryError),
        ("--bridge-cmake-build-directory", NotADirectoryError),
        ("--synapse-library", FileNotFoundError),
        ("--hcl-library", FileNotFoundError),
    ],
)
def test_invalid_required_path_fails_before_build(inputs, monkeypatch, tmp_path, flag, error):
    argv = runtime_argv(inputs)
    argv[argv.index(flag) + 1] = str(tmp_path / "missing")
    aggregate = stale_aggregate(inputs.output)
    calls = []

    tool = runtime_tool()
    monkeypatch.setattr(tool.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    with pytest.raises(error):
        tool.main(argv)

    assert calls == []
    assert not aggregate.exists()


@pytest.mark.parametrize("explicit_overrides", [False, True])
def test_builders_run_in_order_with_unified_environment(inputs, monkeypatch, explicit_overrides):
    monkeypatch.setenv("GAUDI_PYTORCH_BRIDGE_ROOT", "stale-root")
    monkeypatch.setenv("GAUDI_PYTORCH_BRIDGE_BUILD_ROOT", "stale-build")
    monkeypatch.setenv("MAX_JOBS", "999")
    monkeypatch.setenv("DSV41_TEST_MARKER", "preserved")
    calls = []
    tool = runtime_tool()
    monkeypatch.setattr(tool.subprocess, "run", successful_build_run(inputs, calls))

    argv = runtime_argv(inputs, jobs=5)
    if explicit_overrides:
        argv.extend([
            "--backend-library", str(inputs.backend),
            "--native-hccl-library", str(inputs.native_hccl),
        ])
    tool.main(argv)

    kinds = [command_kind(command) for command, _ in calls]
    assert kinds[:2] == ["native", "bridge"]
    native_command, native_kwargs = calls[0]
    bridge_command, bridge_kwargs = calls[1]
    expected_native = [
        sys.executable,
        str(NATIVE_BUILDER),
        "--build-root", str(inputs.build_root.resolve()),
        "--output-dir", str(inputs.output.resolve()),
        "--jobs", "5",
    ]
    expected_bridge = [
        sys.executable,
        str(BRIDGE_BUILDER),
        "--bridge-source", str(inputs.bridge_source.resolve()),
        "--cmake-build-directory", str(inputs.cmake_build.resolve()),
        "--build-directory", str(inputs.bridge_build.resolve()),
        "--synapse-library", str(inputs.synapse.resolve()),
        "--hcl-library", str(inputs.hcl.resolve()),
    ]
    if explicit_overrides:
        expected_bridge.extend([
            "--backend-library", str(inputs.backend.resolve()),
            "--native-hccl-library", str(inputs.native_hccl.resolve()),
        ])
    assert native_command == expected_native
    assert bridge_command == expected_bridge
    assert "--jobs" not in bridge_command
    if not explicit_overrides:
        assert "--backend-library" not in bridge_command
        assert "--native-hccl-library" not in bridge_command
    assert native_kwargs["check"] is True
    assert bridge_kwargs["check"] is True
    for kwargs in (native_kwargs, bridge_kwargs):
        assert kwargs["env"]["GAUDI_PYTORCH_BRIDGE_ROOT"] == str(inputs.bridge_source.resolve())
        assert kwargs["env"]["GAUDI_PYTORCH_BRIDGE_BUILD_ROOT"] == str(inputs.cmake_build.resolve())
        assert kwargs["env"]["MAX_JOBS"] == "5"
        assert kwargs["env"]["DSV41_TEST_MARKER"] == "preserved"


def test_success_publishes_hashed_bridge_and_relative_aggregate_paths(inputs, monkeypatch):
    calls = []
    tool = runtime_tool()
    monkeypatch.setattr(tool.subprocess, "run", successful_build_run(inputs, calls))
    tool.main(runtime_argv(inputs))

    staged_binary = inputs.bridge_build / BRIDGE_BINARY_NAME
    staged_abi = staged_binary.with_suffix(".abi.json")
    published_binary = inputs.output / staged_binary.name
    published_abi = inputs.output / staged_abi.name
    aggregate_path = inputs.output / "deepseek_v41_runtime_build.json"
    aggregate = json.loads(aggregate_path.read_text())

    assert published_binary.read_bytes() == staged_binary.read_bytes()
    assert json.loads(published_abi.read_text()) == json.loads(staged_abi.read_text())
    assert aggregate["runtime_executed"] is False
    assert aggregate["native"]["manifest"]["path"] == "deepseek_v41_build.json"
    assert aggregate["bridge"]["binary"]["path"] == published_binary.name
    assert aggregate["bridge"]["abi_manifest"]["path"] == published_abi.name
    assert aggregate["bridge"]["binary"]["sha256"] == sha256_file(published_binary)
    assert aggregate["bridge"]["abi_manifest"]["sha256"] == sha256_file(published_abi)
    assert aggregate["build_parameters"]["jobs"] == 5

    recorded = set()
    for item in aggregate["outputs"]:
        relative = Path(item["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        assert sha256_file(inputs.output / relative) == item["sha256"]
        recorded.add(relative)
    assert Path("deepseek_v41_build.json") in recorded
    assert Path("deepseek_v4_build.json") in recorded
    assert Path(published_binary.name) in recorded
    assert Path(published_abi.name) in recorded


@pytest.mark.parametrize(
    ("failing", "returncode"),
    [("native", 19), ("bridge", 23)],
)
def test_builder_failure_removes_stale_aggregate_and_stops(inputs, monkeypatch, failing, returncode):
    aggregate = stale_aggregate(inputs.output)
    calls = []

    def fail_selected(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        kind = command_kind(command)
        if kind == "native":
            if failing == "native":
                raise subprocess.CalledProcessError(returncode, command)
            stage_native(inputs.output)
        elif kind == "bridge":
            if failing == "bridge":
                raise subprocess.CalledProcessError(returncode, command)
            stage_bridge(inputs.bridge_build, inputs)
        elif kind == "git":
            raise AssertionError("build failure continued to source metadata")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    tool = runtime_tool()
    monkeypatch.setattr(tool.subprocess, "run", fail_selected)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        tool.main(runtime_argv(inputs))

    assert failure.value.returncode == returncode
    assert not aggregate.exists()
    assert [command_kind(command) for command, _ in calls] == (
        ["native"] if failing == "native" else ["native", "bridge"]
    )


def test_native_manifest_hash_mismatch_aborts_before_bridge(inputs, monkeypatch):
    aggregate = stale_aggregate(inputs.output)
    calls = []
    tool = runtime_tool()
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        successful_build_run(inputs, calls, native_sha256="0" * 64),
    )

    with pytest.raises(RuntimeError, match="differs"):
        tool.main(runtime_argv(inputs))

    assert [command_kind(command) for command, _ in calls] == ["native"]
    assert not aggregate.exists()


def test_bridge_abi_hash_mismatch_publishes_no_aggregate(inputs, monkeypatch):
    aggregate = stale_aggregate(inputs.output)
    calls = []
    tool = runtime_tool()
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        successful_build_run(inputs, calls, bridge_sha256="0" * 64),
    )

    with pytest.raises(RuntimeError, match="[Bb]ridge"):
        tool.main(runtime_argv(inputs))

    assert [command_kind(command) for command, _ in calls] == ["native", "bridge"]
    assert not aggregate.exists()


def test_cli_preserves_builder_nonzero_exit_code(inputs, monkeypatch, capsys):
    aggregate = stale_aggregate(inputs.output)
    calls = []

    def fail_native(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        raise subprocess.CalledProcessError(37, command)

    monkeypatch.setattr(subprocess, "run", fail_native)
    monkeypatch.setattr(sys, "argv", [str(TOOL), *runtime_argv(inputs)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(TOOL), run_name="__main__")

    assert exit_info.value.code == 37
    assert [command_kind(command) for command, _ in calls] == ["native"]
    assert not aggregate.exists()
    assert "exit code 37" in capsys.readouterr().err
