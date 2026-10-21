# SPDX-License-Identifier: Apache-2.0
"""Build and publish the DeepSeek V4.1 native runtime with one command."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

MANIFEST_NAME = "deepseek_v41_runtime_build.json"
BRIDGE_MODULE = "tp2_fused_ar_norm_bridge"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, output: Path) -> dict:
    path = path.resolve()
    try:
        relative = path.relative_to(output).as_posix()
    except ValueError as error:
        raise ValueError(f"Published artifact is outside --output-dir: {path}") from error
    return {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_copy(source: Path, target: Path) -> Path:
    if source.resolve() == target.resolve():
        return target
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def checked_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def checked_directory(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{label} does not exist: {path}")
    return path


def load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-source", type=Path, required=True, help="Bridge source tree")
    parser.add_argument("--bridge-cmake-build-directory", type=Path, required=True,
                        help="Bridge CMake build directory passed to the bridge builder")
    parser.add_argument("--synapse-library", type=Path, required=True)
    parser.add_argument("--hcl-library", type=Path, required=True)
    parser.add_argument("--backend-library", type=Path,
                        help="Override the bridge builder's default backend discovery")
    parser.add_argument("--native-hccl-library", type=Path,
                        help="Override the bridge builder's default native HCCL discovery")
    parser.add_argument("--build-root", type=Path, default=root / "build/deepseek_v41_runtime")
    parser.add_argument("--output-dir", type=Path, default=root / "vllm_gaudi/lib")
    parser.add_argument("--bridge-build-directory", type=Path,
                        help="Bridge extension staging directory (default: BUILD_ROOT/bridge)")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def validate_native(output: Path) -> tuple[Path, dict]:
    invalid = output / "INVALID_BUILD.json"
    if invalid.exists():
        raise RuntimeError(f"Native build left an invalid-build marker: {invalid}")
    manifest_path = checked_file(output / "deepseek_v41_build.json", "native manifest")
    checked_file(output / "deepseek_v4_build.json", "native kernel manifest")
    manifest = load_json(manifest_path, "native manifest")
    sources, binaries = manifest.get("sources"), manifest.get("binaries")
    if not isinstance(sources, dict) or not isinstance(binaries, dict) or not binaries:
        raise ValueError("Native manifest must contain source and binary fingerprints")
    for name, expected in binaries.items():
        if Path(name).name != name:
            raise ValueError(f"Native manifest contains an invalid binary name: {name}")
        binary = checked_file(output / name, "native binary")
        if sha256_file(binary) != expected:
            raise RuntimeError(f"Native binary differs from manifest: {name}")
    return manifest_path, manifest


def find_bridge_artifacts(build: Path) -> tuple[Path, Path, dict]:
    candidates = set()
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidates.update(build.glob(f"{BRIDGE_MODULE}*{suffix}"))
    pairs = {}
    for binary in sorted(candidates, key=str):
        if not binary.is_file():
            continue
        abi_path = binary.with_suffix(".abi.json")
        if not abi_path.is_file():
            continue
        try:
            abi = load_json(abi_path, "bridge ABI manifest")
        except ValueError:
            continue
        if (abi.get("schema") == 1 and isinstance(abi.get("binary_sha256"), str)
                and isinstance(abi.get("adapter_sources"), dict)
                and isinstance(abi.get("bridge_dependencies"), dict)
                and sha256_file(binary) == abi["binary_sha256"]):
            pairs[(binary.resolve(), abi_path.resolve())] = abi
    if len(pairs) != 1:
        raise RuntimeError(f"Expected one valid bridge binary/ABI pair in {build}, found {len(pairs)}")
    (binary, abi_path), abi = next(iter(pairs.items()))
    return binary, abi_path, abi


def validate_bridge_dependencies(abi: dict, requested: dict) -> None:
    groups = {
        "synapse": ("native_dependencies", "synapse"),
        "hcl": ("native_dependencies", "hcl"),
        "backend": ("bridge_dependencies", "backend"),
        "native_hccl": ("bridge_dependencies", "native_hccl"),
    }
    for name, path in requested.items():
        if path is None:
            continue
        group, key = groups[name]
        record = abi.get(group, {}).get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Bridge ABI manifest does not identify the requested {name} library")
        if Path(record.get("path", "")).resolve() != path or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"Bridge ABI manifest does not match the requested {name} library: {path}")


def output_artifacts(output: Path, native_path: Path, native: dict, bridge: Path, abi_path: Path) -> list:
    paths = {native_path, abi_path, bridge, output / "deepseek_v4_build.json"}
    paths.update(output / name for name in native["binaries"])
    audit = output / "tpc_loop_audit.json"
    if audit.is_file():
        paths.add(audit)
    return [artifact(path, output) for path in sorted(paths, key=lambda item: item.resolve())]


def source_metadata(root: Path) -> dict:
    """Read HEAD without running git; return an unavailable commit if this is not a checkout."""
    git = root / ".git"
    try:
        if git.is_file():
            marker = git.read_text().strip()
            if marker.startswith("gitdir:"):
                git = (root / marker[7:].strip()).resolve()
        value = (git / "HEAD").read_text().strip()
        if value.startswith("ref: "):
            value = (git / value[5:]).read_text().strip()
    except OSError:
        return {"commit": None, "version": None}
    commit = value.lower() if len(value) == 40 and all(c in "0123456789abcdef" for c in value.lower()) else None
    return {"commit": commit, "version": commit[:12] if commit else None}


def main(argv=None) -> None:
    root = Path(__file__).resolve().parents[1]
    args = parse_args(argv)
    output = args.output_dir.resolve()
    build_root = args.build_root.resolve()
    bridge_build = (args.bridge_build_directory or build_root / "bridge").resolve()
    aggregate = output / MANIFEST_NAME
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"--output-dir is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    aggregate.unlink(missing_ok=True)

    bridge_source = checked_directory(args.bridge_source, "bridge source")
    cmake_build = checked_directory(args.bridge_cmake_build_directory, "bridge CMake build directory")
    requested = {
        "synapse": checked_file(args.synapse_library, "Synapse library"),
        "hcl": checked_file(args.hcl_library, "HCL library"),
        "backend": checked_file(args.backend_library, "backend library") if args.backend_library else None,
        "native_hccl": (checked_file(args.native_hccl_library, "native HCCL library")
                        if args.native_hccl_library else None),
    }
    if build_root.exists() and not build_root.is_dir():
        raise NotADirectoryError(f"--build-root is not a directory: {build_root}")
    if bridge_build.exists() and not bridge_build.is_dir():
        raise NotADirectoryError(f"--bridge-build-directory is not a directory: {bridge_build}")
    build_env = dict(
        os.environ,
        GAUDI_PYTORCH_BRIDGE_ROOT=str(bridge_source),
        GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(cmake_build),
        MAX_JOBS=str(args.jobs),
    )

    native_command = [sys.executable, str(root / "tools/build_deepseek_v41.py"),
                      "--build-root", str(build_root), "--output-dir", str(output), "--jobs", str(args.jobs)]
    subprocess.run(native_command, check=True, env=build_env)
    native_path, native = validate_native(output)

    bridge_command = [sys.executable, str(root / "tools/communication/build_tp2_fused_ar_norm_bridge.py"),
                      "--bridge-source", str(bridge_source), "--cmake-build-directory", str(cmake_build),
                      "--build-directory", str(bridge_build), "--synapse-library", str(requested["synapse"]),
                      "--hcl-library", str(requested["hcl"])]
    for key in ("backend", "native_hccl"):
        if requested[key] is not None:
            bridge_command.extend((f"--{key.replace('_', '-')}-library", str(requested[key])))
    subprocess.run(bridge_command, check=True, env=build_env)

    bridge_staged, abi_staged, abi = find_bridge_artifacts(bridge_build)
    validate_bridge_dependencies(abi, requested)
    bridge = atomic_copy(bridge_staged, output / bridge_staged.name)
    abi_path = atomic_copy(abi_staged, output / abi_staged.name)
    if sha256_file(bridge) != abi["binary_sha256"]:
        raise RuntimeError("Published bridge binary differs from its ABI manifest")

    manifest = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source_metadata(root),
        "runtime_executed": False,
        "native": {
            "manifest": artifact(native_path, output),
            "summary": {"sources": len(native["sources"]), "binaries": len(native["binaries"]),
                        "host_gather_abi_version": native.get("host_gather_abi_version")},
        },
        "bridge": {
            "binary": artifact(bridge, output),
            "abi_manifest": artifact(abi_path, output),
            "summary": {key: abi.get(key) for key in (
                "tp8_prepared_group_version", "tp8_native_graph_available",
                "device_engram_shared_mapping_version", "tp4_index_gather_pair_version")},
        },
        "outputs": output_artifacts(output, native_path, native, bridge, abi_path),
        "build_parameters": {
            "build_root": str(build_root),
            "output_dir": str(output),
            "bridge_build_directory": str(bridge_build),
            "jobs": args.jobs,
            "bridge_source": str(bridge_source),
            "bridge_cmake_build_directory": str(cmake_build),
            "synapse_library": str(requested["synapse"]),
            "hcl_library": str(requested["hcl"]),
            "backend_library": str(requested["backend"]) if requested["backend"] else None,
            "native_hccl_library": str(requested["native_hccl"]) if requested["native_hccl"] else None,
        },
        "environment_overrides": {
            "GAUDI_PYTORCH_BRIDGE_ROOT": str(bridge_source),
            "GAUDI_PYTORCH_BRIDGE_BUILD_ROOT": str(cmake_build),
            "MAX_JOBS": str(args.jobs),
        },
        "commands": {"native": native_command, "bridge": bridge_command},
    }
    atomic_json(aggregate, manifest)
    print(aggregate)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"Build command failed with exit code {error.returncode}: {error.cmd}", file=sys.stderr)
        sys.exit(error.returncode)
