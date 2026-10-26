# SPDX-License-Identifier: Apache-2.0
"""Build an isolated native runtime with graph-local scratch reuse.

The build recipe is derived from an existing CMake/Ninja build directory.  It
does not require a previously extracted ``parent/source`` tree or a hand-made
``commands.json``.  The source used for the baseline relink is taken from Git
HEAD by default, which makes the proof independent of uncommitted runtime
diagnostics.

Only ordered segments within one graph share scratch.  Allocations remain
owned by that graph's existing segment resources until final completion.
Neither captured addresses nor graph class layouts are changed after capture.
The native 2 GiB per-graph workspace cap is retained unchanged.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Iterable


WORKSPACE_ALIGNMENT = 8191
MIN_ARENA_BYTES = 8192
WORKSPACE_CAP_BYTES = 2 << 30
SOURCE_SUFFIX = "synapse/src/runtime/scal/common/stream_compute_scal.cpp"
OBJECT_SUFFIX = "src/CMakeFiles/Synapse.dir/runtime/scal/common/stream_compute_scal.cpp.o"
REUSE_ENV = "VLLM_HPU_DSV41_NATIVE_WORKSPACE_REUSE"
WORKSPACE_LOG_ENV = "DSV41_NATIVE_WORKSPACE_LOG"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def next_arena_bytes(workspace_size: int) -> int:
    """Return the geometric arena size used by the C++ patch."""
    if workspace_size <= 0:
        return 0
    allocation = MIN_ARENA_BYTES
    while allocation < workspace_size:
        allocation <<= 1
    return allocation


def reusable_arena(previous_allocation_bytes: Iterable[int], requested: int) -> int | None:
    """Return the first reusable arena capacity, or None when growth is needed."""
    if requested <= 0:
        return None
    for allocation in previous_allocation_bytes:
        if allocation >= WORKSPACE_ALIGNMENT and allocation - WORKSPACE_ALIGNMENT >= requested:
            return allocation
    return None


def _replace_once(value: str, old: str, new: str, *, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise ValueError(f"Unsupported {label}: expected one anchor, found {count}")
    return value.replace(old, new, 1)


def patch_source(original: str) -> str:
    """Patch the native segment allocation policy.

    The anchors intentionally tolerate whitespace and the diagnostic changes
    currently carried by the runtime branch.  The source used for a proof
    should normally be ``git HEAD`` so the baseline relink remains reproducible.
    """
    if REUSE_ENV in original:
        raise ValueError("Source is already patched with native workspace reuse")

    initialize = re.compile(
        r"(?P<indent>[ \t]*)status\s*=\s*segment\.memory->initialize\("
        r"globalSize, arcSize, workspaceSize\);"
    )
    match = initialize.search(original)
    if match is None:
        raise ValueError("Unsupported native segment allocation implementation")
    indent = match.group("indent")
    replacement = f'''{indent}const char* reuseMode = std::getenv("{REUSE_ENV}");
{indent}const bool reuseScratch = reuseMode && reuseMode[0] == '1' && reuseMode[1] == '\\0';
{indent}const char* reuseLogMode = std::getenv("{WORKSPACE_LOG_ENV}");
{indent}const bool logReuse = reuseLogMode && reuseLogMode[0] == '1' && reuseLogMode[1] == '\\0';
{indent}uint64_t scratchAddress = 0;
{indent}uint64_t allocationSize = workspaceSize;
{indent}if (reuseScratch && workspaceSize)
{indent}{{
{indent}    // Recipes execute in capture order on this graph's compute stream.
{indent}    // Only scratch is shared; program memory and launch tensors are not.
{indent}    for (const auto& previous : m_segments)
{indent}    {{
{indent}        const auto bytes = previous.memory->workspaceAllocationBytes();
{indent}        if (bytes >= 8191 && bytes - 8191 >= workspaceSize)
{indent}        {{
{indent}            scratchAddress = previous.memory->workspaceDeviceAddress();
{indent}            break;
{indent}        }}
{indent}    }}
{indent}    if (scratchAddress)
{indent}        allocationSize = 0;
{indent}    else
{indent}    {{
{indent}        // Retain old arenas rather than relocating captured pointers.
{indent}        // Geometric growth bounds retained capacity below twice the max.
{indent}        allocationSize = 8192;
{indent}        while (allocationSize < workspaceSize) allocationSize <<= 1;
{indent}    }}
{indent}}}
{indent}if (allocationSize && allocationSize + 8191 > maximumWorkspaceBytes - m_workspaceAllocationBytes)
{indent}{{
{indent}    status = synOutOfDeviceMemory;
{indent}    m_state = State::INVALID;
{indent}    return nullptr;
{indent}}}
{indent}status = segment.memory->initialize(globalSize, arcSize, allocationSize);'''
    candidate = original[:match.start()] + replacement + original[match.end():]

    address_anchor = re.compile(
        r"(?P<indent>[ \t]*)segment\.workspaceAddress\s*=\s*"
        r"workspaceSize\s*\?\s*segment\.memory->workspaceDeviceAddress\(\)\s*:\s*workspaceAddress;"
    )
    address_match = address_anchor.search(candidate)
    if address_match is None:
        raise ValueError("Unsupported native scratch binding implementation")
    address_indent = address_match.group("indent")
    address_replacement = f'''{address_indent}segment.workspaceAddress = scratchAddress ? scratchAddress :
{address_indent}    (workspaceSize ? segment.memory->workspaceDeviceAddress() : workspaceAddress);
{address_indent}if (reuseScratch && workspaceSize && logReuse)
{address_indent}    std::fprintf(stderr, "NATIVE_WORKSPACE_REUSE requested=%llu allocated=%llu reused=%u\\n",
{address_indent}        static_cast<unsigned long long>(workspaceSize),
{address_indent}        static_cast<unsigned long long>(allocationSize), unsigned(scratchAddress != 0));'''
    candidate = (candidate[:address_match.start()] + address_replacement +
                 candidate[address_match.end():])

    cap_anchor = re.compile(
        r"if\s*\(\s*m_workspaceAllocationBytes\s*>\s*maximumWorkspaceBytes\s*\|\|\s*"
        r"workspaceSize\s*>\s*maximumWorkspaceBytes\s*\|\|\s*"
        r"\(workspaceSize\s*&&\s*workspaceSize\s*\+\s*8191\s*>\s*"
        r"maximumWorkspaceBytes\s*-\s*m_workspaceAllocationBytes\)\s*\)",
        re.DOTALL,
    )
    candidate, count = cap_anchor.subn(
        "if (m_workspaceAllocationBytes > maximumWorkspaceBytes || workspaceSize > maximumWorkspaceBytes)",
        candidate,
        count=1,
    )
    if count != 1:
        raise ValueError("Unsupported native workspace bound")

    if "#include <cstdlib>" not in candidate:
        candidate = _replace_once(
            candidate, "#include <limits>\n", "#include <limits>\n#include <cstdlib>\n",
            label="limits include",
        )
    if "#include <cstdio>" not in candidate:
        candidate = _replace_once(
            candidate, "#include <cstdlib>\n", "#include <cstdlib>\n#include <cstdio>\n",
            label="cstdlib include",
        )
    return candidate


def _run(command: list[str], *, cwd: Path, log) -> None:
    printable = " ".join(shlex.quote(value) for value in command)
    log.write(f"$ (cd {shlex.quote(str(cwd))} && {printable})\n")
    log.flush()
    subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def _run_capture(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)
    return result.stdout


def _replace_option(command: list[str], option: str, value: str) -> list[str]:
    result = list(command)
    try:
        index = result.index(option)
    except ValueError as error:
        raise ValueError(f"Command has no {option} option") from error
    result[index + 1] = value
    return result


def _replace_path(command: list[str], old: str, new: str, *, suffix: str | None = None) -> list[str]:
    result = list(command)
    indices = [index for index, value in enumerate(result) if value == old]
    if not indices and suffix is not None:
        indices = [index for index, value in enumerate(result) if value.endswith(suffix)]
    if len(indices) != 1:
        raise ValueError(f"Expected one command path ending in {suffix or old}, found {len(indices)}")
    result[indices[0]] = new
    return result


def _load_compile_command(base: Path) -> tuple[Path, list[str], Path]:
    commands_path = base / "compile_commands.json"
    if not commands_path.is_file():
        raise ValueError(f"Missing compile_commands.json: {commands_path}")
    entries = json.loads(commands_path.read_text())
    matches = [entry for entry in entries if str(entry.get("file", "")).endswith(SOURCE_SUFFIX)]
    if len(matches) != 1:
        raise ValueError(f"Expected one compile command for {SOURCE_SUFFIX}, found {len(matches)}")
    entry = matches[0]
    if isinstance(entry.get("arguments"), list):
        command = list(entry["arguments"])
    else:
        command = shlex.split(entry["command"])
    directory = Path(entry.get("directory", base)).resolve()
    source = Path(entry["file"]).resolve()
    return directory, command, source


def _load_ninja_link(base: Path, ninja: str) -> list[str]:
    output = _run_capture([ninja, "-C", str(base), "-t", "commands", "lib/libSynapse.so"], cwd=base)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Ninja produced no libSynapse.so commands")
    parts = [part.strip() for part in lines[-1].split(" && ")]
    if parts and parts[0] == ":":
        parts = parts[1:]
    if not parts or not parts[0].startswith(("/usr/bin/", "c++", "g++")):
        raise ValueError("Could not isolate the libSynapse.so link command")
    return shlex.split(parts[0])


def _git_output(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def _materialize_source(build: Path, source: Path, source_ref: str) -> tuple[Path, dict]:
    source_dir = source.parent
    repo = Path(_git_output(source_dir, "rev-parse", "--show-toplevel").decode().strip())
    relative_source = source.relative_to(repo)
    relative_dir = relative_source.parent
    commit = _git_output(repo, "rev-parse", source_ref).decode().strip()
    names = _git_output(repo, "ls-tree", "-r", "--name-only", source_ref, "--", str(relative_dir)).decode().splitlines()
    if str(relative_source) not in names:
        raise ValueError(f"{source_ref} does not contain {relative_source}")
    source_root = build / "source"
    records = {}
    for name in names:
        path = repo / name
        if not path.is_file():
            continue
        content = _git_output(repo, "show", f"{source_ref}:{name}")
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        records[name] = sha256_bytes(content)
    return source_root / str(relative_source), {
        "mode": "git-head",
        "repo": str(repo),
        "ref": source_ref,
        "commit": commit,
        "source_path": str(source),
        "files": records,
    }


def _compile_command(command: list[str], original_source: Path, candidate_source: Path,
                     object_path: Path, include_root: Path) -> list[str]:
    result = list(command)
    result = _replace_path(result, str(original_source), str(candidate_source), suffix=original_source.name)
    result = _replace_option(result, "-o", str(object_path))
    if "-MF" in result:
        result = _replace_option(result, "-MF", str(object_path.with_suffix(".d")))
    if "-MT" in result:
        result = _replace_option(result, "-MT", str(object_path))
    result[1:1] = [f"-I{include_root}"]
    return result


def _link_command(command: list[str], object_path: Path, output: Path) -> list[str]:
    result = _replace_path(result := list(command), OBJECT_SUFFIX, str(object_path), suffix="stream_compute_scal.cpp.o")
    return _replace_option(result, "-o", str(output))


def _postprocess(binary: Path, debug: Path, *, cwd: Path, log) -> None:
    _run(["strip", str(binary), "--only-keep-debug", "-o", str(debug)], cwd=cwd, log=log)
    _run(["strip", str(binary), "--strip-unneeded"], cwd=cwd, log=log)
    _run(["objcopy", f"--add-gnu-debuglink={debug}", str(binary)], cwd=cwd, log=log)


def _profile(base: Path | None, output: Path, candidate: Path, result: Path) -> dict:
    if base is not None:
        profile = json.loads(base.read_text())
        if not isinstance(profile, dict):
            raise ValueError("Runtime profile must be a JSON object")
    else:
        profile = {"schema": 1, "environment": {}, "additional_libraries": [], "configuration_files": []}
    profile.setdefault("environment", {})[REUSE_ENV] = "1"
    libraries = [
        item for item in profile.get("additional_libraries", [])
        if Path(item.get("path", "")).name != candidate.name
    ]
    libraries.append({
        "path": str(candidate),
        "sha256": digest(candidate),
        "compilation_proof": {"path": str(result), "sha256": digest(result)},
    })
    profile["additional_libraries"] = libraries
    profile.setdefault("configuration_files", [])
    output.write_text(json.dumps(profile, indent=2) + "\n")
    return profile


def build(args: argparse.Namespace) -> dict:
    base = args.base.resolve()
    build_dir = args.build.resolve()
    if not base.is_dir():
        raise ValueError(f"Base build directory does not exist: {base}")
    if build_dir.exists() and any(build_dir.iterdir()):
        raise ValueError(f"Output directory must be absent or empty: {build_dir}")
    build_dir.mkdir(parents=True, exist_ok=True)

    base_library = base / "lib/libSynapse.so"
    base_debug = base / "lib/libSynapse.so.debug"
    for path in (base_library, base_debug, base / "compile_commands.json", base / "build.ninja"):
        if not path.is_file():
            raise ValueError(f"Missing base build input: {path}")

    compile_cwd, original_compile, source = _load_compile_command(base)
    link_command = _load_ninja_link(base, args.ninja)
    candidate_source, source_record = _materialize_source(build_dir, source, args.source_ref)
    original = candidate_source.read_text()
    patched = patch_source(original)
    candidate_source.write_text(patched)

    patch_path = build_dir / "source.patch"
    patch_path.write_text("".join(difflib.unified_diff(
        original.splitlines(True), patched.splitlines(True),
        fromfile=f"a/{source.name}", tofile=f"b/{source.name}",
    )))
    include_root = candidate_source.parents[3]
    object_path = build_dir / "stream_compute_scal.cpp.o"
    candidate_unstripped = build_dir / "candidate-unstripped.so"
    candidate_library = build_dir / "libSynapse.so"
    candidate_debug = build_dir / "libSynapse.so.debug"
    baseline_dir = build_dir / "baseline"
    baseline_dir.mkdir()
    baseline_library = baseline_dir / "libSynapse.so"
    baseline_debug = baseline_dir / "libSynapse.so.debug"

    compile_command = _compile_command(original_compile, source, candidate_source, object_path, include_root)
    # Keep the baseline link command byte-for-byte equivalent, including its
    # relative object path.  Only the output location changes.
    baseline_link = _replace_option(link_command, "-o", str(baseline_library))
    candidate_link = _link_command(link_command, object_path, candidate_unstripped)
    commands = {
        "compile": compile_command,
        "baseline_link": baseline_link,
        "candidate_link": candidate_link,
        "postprocess": [
            ["strip", "<binary>", "--only-keep-debug", "-o", "<debug>"],
            ["strip", "<binary>", "--strip-unneeded"],
            ["objcopy", "--add-gnu-debuglink=<debug>", "<binary>"],
        ],
    }
    (build_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")

    with (build_dir / "build.log").open("w") as log:
        _run(compile_command, cwd=compile_cwd, log=log)
        _run(baseline_link, cwd=base, log=log)
        _postprocess(baseline_library, baseline_debug, cwd=build_dir, log=log)
        _run(candidate_link, cwd=base, log=log)
        _postprocess(candidate_unstripped, candidate_debug, cwd=build_dir, log=log)
        candidate_unstripped.rename(candidate_library)

    parent_sha = digest(base_library)
    baseline_sha = digest(baseline_library)
    candidate_sha = digest(candidate_library)
    baseline_match = baseline_sha == parent_sha and digest(baseline_debug) == digest(base_debug)
    if not baseline_match:
        raise RuntimeError(
            "Baseline relink does not reproduce the qualified runtime: "
            f"expected {parent_sha}, got {baseline_sha}"
        )

    result_path = build_dir / "RESULT.json"
    result = {
        "schema": 1,
        "kind": "native_ordered_scratch_v1",
        "status": "built_unqualified",
        "default_enabled": False,
        "base_build_dir": str(base),
        "base_library": str(base_library),
        "base_library_sha256": parent_sha,
        "base_debug_sha256": digest(base_debug),
        "parent_sha256": parent_sha,
        "parent_relinked_sha256": baseline_sha,
        "baseline_match": baseline_match,
        "library": str(candidate_library),
        "sha256": candidate_sha,
        "debug_library": str(candidate_debug),
        "debug_sha256": digest(candidate_debug),
        "compilation_sha256": parent_sha,
        "source_patch": str(patch_path),
        "source_patch_sha256": digest(patch_path),
        "source": source_record,
        "source_prerequisites": source_record["files"],
        "compile_commands": str(base / "compile_commands.json"),
        "build_ninja": str(base / "build.ninja"),
        "replaced_runtime_unit": "stream_compute_scal.cpp.o",
        "class_layout_changed": False,
        "final_completion_retirement_unchanged": True,
        "graph_compiler_objects_unchanged": True,
        "hcl_unchanged": True,
        "logging_only": True,
        "logging_only_note": "Only successful reuse stderr logging is gated; allocation/replay behavior is unchanged.",
        "success_logging": {
            "environment": WORKSPACE_LOG_ENV,
            "default": "0",
            "enabled_value": "1",
            "success_only": True,
        },
        "workspace_cap_bytes": WORKSPACE_CAP_BYTES,
        "workspace_cap_expression": "uint64_t(2) << 30",
        "workspace_reuse": {
            "environment": REUSE_ENV,
            "only_workspace": True,
            "stable_addresses": True,
            "ordered_segments": True,
            "old_arena_retained": True,
            "geometric_growth": True,
        },
        "stream_order_evidence": [
            "NativeComputeProgram::prepare appends segment commands in order",
            "native HCL waits are inserted at consumer segment boundaries",
            "native replay rejects out-of-order segment indices",
        ],
        "candidate_preload": str(candidate_library),
        "recommended_environment": {
            "LD_PRELOAD": str(candidate_library),
            REUSE_ENV: "1",
            WORKSPACE_LOG_ENV: "0",
        },
    }
    profile_path = args.profile_output.resolve() if args.profile_output else build_dir / "runtime-profile.json"
    result["runtime_profile"] = str(profile_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    # The profile points at RESULT.json by hash.  Do not rewrite RESULT after
    # this point or that compilation proof would become stale.
    _profile(args.profile_base.resolve() if args.profile_base else None, profile_path, candidate_library, result_path)
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True,
                        help="Existing CMake/Ninja build directory containing lib/libSynapse.so")
    parser.add_argument("--build", type=Path, required=True, help="New isolated candidate directory")
    parser.add_argument("--source-ref", default="HEAD", help="Git ref used for the frozen baseline source")
    parser.add_argument("--ninja", default=shutil.which("ninja") or "ninja")
    parser.add_argument("--profile-base", type=Path,
                        help="Optional existing runtime profile to extend")
    parser.add_argument("--profile-output", type=Path,
                        help="Optional runtime profile output path")
    args = parser.parse_args(argv)
    try:
        build(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
