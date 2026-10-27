#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a bounded GLM-5.3 KDA smoke test without loading the full model.

CPU mode is safe by default and never invokes hl-smi or creates HPU tensors.
HPU mode first proves that one explicitly selected card is idle, pins that card,
and only then creates small test tensors.  An occupied or unparseable target is
a hard failure; it is never converted into a passing result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_BUSY = 3
SCHEMA = "glm53-hpu-smoke.v1"

GLM_MODEL_GEOMETRY = {
    "global_num_heads": 64,
    "tensor_parallel_size": 8,
    "local_num_heads": 8,
    "head_dim": 128,
}

TOOL_PATH = Path(__file__).resolve()
GAUDI_ROOT = TOOL_PATH.parents[1]
ENGINE_ROOT = GAUDI_ROOT.parent / "vllm-glm53-flash"


class DeviceBusyError(RuntimeError):
    """Raised before HPU allocation when the selected card is not provably idle."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_command(command: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def _first_number(value: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def _parse_compute_processes(raw_output: str) -> list[dict[str, object]]:
    """Parse the sparse hl-smi Compute Processes table.

    The table uses variable-width columns and can be absent when no process is
    active.  An unparsed table is not treated as evidence that a card is free;
    the caller also checks memory usage and requires a target CSV row.
    """
    processes: list[dict[str, object]] = []
    in_table = False
    for line in raw_output.splitlines():
        if "Compute Processes:" in line:
            in_table = True
            continue
        if not in_table or not line.startswith("|"):
            continue
        fields = [field for field in re.split(r"\s{2,}", line.strip().strip("|")) if field]
        if len(fields) < 5 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        processes.append(
            {
                "index": int(fields[0]),
                "pid": int(fields[1]),
                "type": fields[2],
                "process": fields[3],
                "memory": fields[4],
            }
        )
    return processes


def _collect_hl_smi_evidence(target_index: int, idle_memory_limit_mib: float) -> dict[str, object]:
    """Collect a fail-closed occupancy snapshot for one AIP index."""
    query_command = [
        "hl-smi",
        "-Q",
        "index,module_id,memory.used,memory.total,utilization.aip",
        "-f",
        "csv,noheader",
    ]
    process_command = ["hl-smi"]
    evidence: dict[str, object] = {
        "checked_at_utc": _utc_now(),
        "target_index": target_index,
        "idle_memory_limit_mib": idle_memory_limit_mib,
        "commands": [query_command, process_command],
        "errors": [],
        "devices": [],
        "processes": [],
        "target": None,
        "busy": True,
        "reasons": [],
    }

    try:
        query = _run_command(query_command)
    except Exception as exc:  # subprocess failures must fail closed
        evidence["errors"].append(f"hl-smi query failed: {type(exc).__name__}: {exc}")
        query = None
    try:
        process_snapshot = _run_command(process_command)
    except Exception as exc:
        evidence["errors"].append(f"hl-smi process snapshot failed: {type(exc).__name__}: {exc}")
        process_snapshot = None

    devices: list[dict[str, object]] = []
    if query is not None:
        evidence["query_returncode"] = query.returncode
        evidence["query_stdout"] = query.stdout
        evidence["query_stderr"] = query.stderr
        if query.returncode != 0:
            evidence["errors"].append(f"hl-smi query exited with {query.returncode}")
        for line in query.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                continue
            index = _first_number(parts[0])
            module_id = _first_number(parts[1])
            used = _first_number(parts[2])
            total = _first_number(parts[3])
            utilization = _first_number(parts[4])
            if None in (index, module_id, used, total, utilization):
                continue
            devices.append(
                {
                    "index": int(index),
                    "module_id": int(module_id),
                    "memory_used_mib": float(used),
                    "memory_total_mib": float(total),
                    "utilization_percent": float(utilization),
                }
            )

    processes: list[dict[str, object]] = []
    if process_snapshot is not None:
        evidence["process_returncode"] = process_snapshot.returncode
        evidence["process_stdout"] = process_snapshot.stdout
        evidence["process_stderr"] = process_snapshot.stderr
        if process_snapshot.returncode != 0:
            evidence["errors"].append(f"hl-smi process snapshot exited with {process_snapshot.returncode}")
        processes = _parse_compute_processes(process_snapshot.stdout)

    evidence["devices"] = devices
    evidence["processes"] = processes
    target = next((device for device in devices if device["index"] == target_index), None)
    evidence["target"] = target
    reasons: list[str] = list(evidence["errors"])
    if target is None:
        reasons.append(f"target AIP index {target_index} was absent or unparseable in hl-smi output")
    else:
        target_processes = [process for process in processes if process["index"] == target_index]
        target["processes"] = target_processes
        if target_processes:
            pids = sorted({process["pid"] for process in target_processes})
            reasons.append(f"target AIP {target_index} has active compute process PID(s) {pids}")
        if float(target["memory_used_mib"]) > idle_memory_limit_mib:
            reasons.append(
                f"target AIP {target_index} uses {target['memory_used_mib']:.0f} MiB, "
                f"above idle limit {idle_memory_limit_mib:.0f} MiB"
            )
        if float(target["utilization_percent"]) > 0:
            reasons.append(
                f"target AIP {target_index} reports {target['utilization_percent']:.0f}% utilization"
            )
    evidence["reasons"] = reasons
    evidence["busy"] = bool(reasons)
    return evidence


def _require_idle_target(target_index: int, idle_memory_limit_mib: float | None) -> dict[str, object]:
    if idle_memory_limit_mib is None:
        raise ValueError("HPU mode requires an idle-memory limit")
    evidence = _collect_hl_smi_evidence(target_index, idle_memory_limit_mib)
    if evidence["busy"]:
        raise DeviceBusyError("; ".join(str(reason) for reason in evidence["reasons"]))
    return evidence


def _static_provenance(args: argparse.Namespace) -> dict[str, object]:
    return {
        "tool": {
            "path": str(TOOL_PATH),
            "sha256": _file_sha256(TOOL_PATH),
        },
        "repo_roots": {
            "vllm_gaudi": str(GAUDI_ROOT),
            "vllm": str(ENGINE_ROOT),
        },
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
        },
        "git": {
            "vllm_gaudi": _git_metadata(GAUDI_ROOT),
            "vllm": _git_metadata(ENGINE_ROOT),
        },
        "requested_environment": {
            "VLLM_PLUGINS": os.environ.get("VLLM_PLUGINS"),
            "VLLM_TARGET_DEVICE": os.environ.get("VLLM_TARGET_DEVICE"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        "guardrails": {
            "full_model_load": "prohibited and not attempted",
            "model_weights_read": False,
            "shared_environment_files_modified": False,
            "single_hpu_only_in_hpu_mode": True,
        },
        "requested_device": args.device,
        "requested_device_id": args.device_id,
    }


def _git_metadata(root: Path) -> dict[str, object]:
    result: dict[str, object] = {"root": str(root), "head": None, "branch": None, "dirty": None}
    try:
        head = _run_command(["git", "-C", str(root), "rev-parse", "HEAD"])
        branch = _run_command(["git", "-C", str(root), "branch", "--show-current"])
        status = _run_command(["git", "-C", str(root), "status", "--short", "--untracked-files=all"])
        if head.returncode == 0:
            result["head"] = head.stdout.strip()
        if branch.returncode == 0:
            result["branch"] = branch.stdout.strip()
        if status.returncode == 0:
            paths = [line for line in status.stdout.splitlines() if line.strip()]
            result["dirty"] = bool(paths)
            result["dirty_paths"] = paths[:64]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _prepare_environment(device: str) -> None:
    """Set only process-local variables before importing torch or vLLM."""
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_TARGET_DEVICE"] = "hpu" if device == "hpu" else "cpu"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _pin_hpu(module_id: int) -> None:
    os.environ["HABANA_VISIBLE_MODULES"] = str(module_id)
    os.environ["HLS_MODULE_ID"] = str(module_id)
    # Eager execution avoids making this tiny numerical check depend on a recipe cache.
    os.environ["PT_HPU_LAZY_MODE"] = "0"
    os.environ["PT_HPU_ENABLE_EAGER_CACHE"] = "0"


def _isolate_imports() -> None:
    for root in (ENGINE_ROOT, GAUDI_ROOT):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def _is_below(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _prepare_runtime(seed: int) -> dict[str, object]:
    _isolate_imports()
    import torch
    import vllm
    import vllm_gaudi
    from vllm_gaudi.ops import hpu_kda_pytorch as kda

    vllm_file = Path(vllm.__file__).resolve()
    gaudi_file = Path(vllm_gaudi.__file__).resolve()
    if not _is_below(vllm_file, ENGINE_ROOT):
        raise RuntimeError(f"vllm import escaped the isolated worktree: {vllm_file}")
    if not _is_below(gaudi_file, GAUDI_ROOT):
        raise RuntimeError(f"vllm_gaudi import escaped the isolated worktree: {gaudi_file}")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 4))
    torch.manual_seed(seed)
    return {
        "torch": torch,
        "kda": kda,
        "vllm": vllm,
        "vllm_gaudi": vllm_gaudi,
        "details": {
            "torch_version": torch.__version__,
            "torch_git_version": getattr(torch.version, "git_version", None),
            "torch_num_threads": torch.get_num_threads(),
            "vllm_version": getattr(vllm, "__version__", None),
            "vllm_file": str(vllm_file),
            "vllm_gaudi_version": getattr(vllm_gaudi, "__version__", None),
            "vllm_gaudi_file": str(gaudi_file),
            "kda_file": str(Path(kda.__file__).resolve()),
            "sys_path_head": sys.path[:8],
            "selected_environment": {
                name: os.environ.get(name)
                for name in (
                    "VLLM_PLUGINS",
                    "VLLM_TARGET_DEVICE",
                    "HABANA_VISIBLE_MODULES",
                    "HLS_MODULE_ID",
                    "PT_HPU_LAZY_MODE",
                    "PT_HPU_ENABLE_EAGER_CACHE",
                )
            },
        },
    }


def _sync(torch_module: object, device_mode: str) -> None:
    if device_mode == "hpu" and hasattr(torch_module, "hpu") and hasattr(torch_module.hpu, "synchronize"):
        torch_module.hpu.synchronize()


def _move(torch_module: object, context: dict[str, object], tensor: object) -> object:
    if context["device_mode"] == "hpu":
        context["hpu_tensors_created"] = True
        return tensor.to(torch_module.device("hpu"))
    return tensor


def _compare(
    torch_module: object,
    context: dict[str, object],
    name: str,
    actual: object,
    expected: object,
    *,
    atol: float,
    rtol: float,
    expected_dtype: object,
) -> dict[str, object]:
    actual_cpu = actual.detach().to(device="cpu").float()
    expected_cpu = expected.detach().to(device="cpu").float()
    shape_ok = tuple(actual_cpu.shape) == tuple(expected_cpu.shape)
    finite = bool(torch_module.isfinite(actual_cpu).all().item())
    if shape_ok:
        difference = (actual_cpu - expected_cpu).abs()
        denominator = expected_cpu.abs().clamp_min(torch_module.finfo(torch_module.float32).tiny)
        relative = difference / denominator
        close = bool(torch_module.isclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol).all().item())
        max_abs = float(difference.max().item()) if difference.numel() else 0.0
        mean_abs = float(difference.mean().item()) if difference.numel() else 0.0
        max_rel = float(relative.max().item()) if relative.numel() else 0.0
        mismatch_count = int((~torch_module.isclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol)).sum().item())
    else:
        close = False
        max_abs = None
        mean_abs = None
        max_rel = None
        mismatch_count = int(actual_cpu.numel())
    dtype_ok = actual.dtype == expected_dtype
    device_ok = actual.device.type == context["device_mode"]
    return {
        "name": name,
        "ok": bool(shape_ok and finite and close and dtype_ok and device_ok),
        "shape": list(actual_cpu.shape),
        "expected_shape": list(expected_cpu.shape),
        "shape_ok": shape_ok,
        "finite": finite,
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected_dtype),
        "dtype_ok": dtype_ok,
        "actual_device": actual.device.type,
        "device_ok": device_ok,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_rel_error": max_rel,
        "mismatch_count": mismatch_count,
    }


def _boolean_check(name: str, ok: bool, detail: object = None) -> dict[str, object]:
    result: dict[str, object] = {"name": name, "ok": bool(ok)}
    if detail is not None:
        result["detail"] = detail
    return result


def _case(name: str, config: dict[str, object], *checks: dict[str, object]) -> dict[str, object]:
    return {"name": name, "ok": all(check["ok"] for check in checks), "config": config, "checks": list(checks)}


def _inputs(torch_module: object, batch: int, seq_len: int, heads: int, head_dim: int, seed: int, dtype: object):
    generator = torch_module.Generator(device="cpu").manual_seed(seed)
    q = torch_module.randn(batch, seq_len, heads, head_dim, generator=generator).to(dtype)
    k = torch_module.randn(batch, seq_len, heads, head_dim, generator=generator).to(dtype)
    v = torch_module.randn(batch, seq_len, heads, head_dim, generator=generator).to(dtype)
    g = -torch_module.sigmoid(torch_module.randn(batch, seq_len, heads, head_dim, generator=generator)) * 0.25
    beta = torch_module.sigmoid(torch_module.randn(batch, seq_len, heads, generator=generator)).to(dtype)
    return q, k, v, g, beta


def _l2norm(torch_module: object, tensor: object) -> object:
    return tensor / torch_module.sqrt(torch_module.sum(tensor.square(), dim=-1, keepdim=True) + 1e-6)


def _oracle_sequence(
    torch_module: object,
    q: object,
    k: object,
    v: object,
    g: object,
    beta: object,
    initial_state: object | None,
    *,
    scale: float | None,
) -> tuple[object, object]:
    """Independent matrix/vector KDA oracle for one sequence."""
    seq_len, heads, head_dim = q.shape
    qf = _l2norm(torch_module, q.float())
    kf = _l2norm(torch_module, k.float())
    vf = v.float()
    gf = g.float()
    bf = beta.float()
    qf = qf * (head_dim**-0.5 if scale is None else scale)
    if initial_state is None:
        state = torch_module.zeros(heads, head_dim, head_dim, dtype=torch_module.float32)
    else:
        state = initial_state.float().clone()
    output = torch_module.empty(seq_len, heads, head_dim, dtype=torch_module.float32)
    for token in range(seq_len):
        decayed = state * torch_module.exp(gf[token]).unsqueeze(1)
        next_state = torch_module.empty_like(state)
        for head in range(heads):
            prediction = decayed[head].matmul(kf[token, head])
            residual = (vf[token, head] - prediction) * bf[token, head]
            next_state[head] = decayed[head] + torch_module.outer(residual, kf[token, head])
            output[token, head] = next_state[head].matmul(qf[token, head])
        state = next_state
    return output, state


def _oracle_batch(torch_module: object, q, k, v, g, beta, initial_state, *, scale=None):
    outputs = []
    states = []
    for batch_index in range(q.shape[0]):
        state_i = None if initial_state is None else initial_state[batch_index]
        output_i, state_i = _oracle_sequence(
            torch_module,
            q[batch_index],
            k[batch_index],
            v[batch_index],
            g[batch_index],
            beta[batch_index],
            state_i,
            scale=scale,
        )
        outputs.append(output_i)
        states.append(state_i)
    return torch_module.stack(outputs), torch_module.stack(states)


def _tolerances(dtype: object, torch_module: object, *, chunk: bool = False) -> tuple[float, float]:
    if dtype == torch_module.float32:
        value = 5e-5 if chunk else 3e-5
        return value, value
    return 2e-2, 2e-2


def _case_gate(runtime: dict[str, object], context: dict[str, object], dtype: object, label: str):
    torch_module = runtime["torch"]
    kda = runtime["kda"]
    heads, head_dim = 3, 4
    generator = torch_module.Generator(device="cpu").manual_seed(100 if dtype == torch_module.float32 else 101)
    raw_g = torch_module.randn(2, 5, heads, head_dim, generator=generator).to(dtype)
    raw_g[0, 0, 0, 0] = 80.0
    raw_g[0, 1, 1, 1] = -80.0
    a_log = torch_module.log(torch_module.tensor([0.25, 1.0, 4.0], dtype=torch_module.float32))
    dt_bias = torch_module.linspace(-0.5, 0.75, heads * head_dim, dtype=torch_module.float32).reshape(heads, head_dim)
    lower_bound = -3.5
    expected = lower_bound * torch_module.sigmoid(
        a_log.exp().view(1, 1, heads, 1) * (raw_g.float() + dt_bias.float())
    )
    actual = kda.hpu_kda_gate(
        _move(torch_module, context, raw_g),
        _move(torch_module, context, a_log),
        _move(torch_module, context, dt_bias),
        lower_bound,
        heads,
        head_dim,
    )
    atol, rtol = _tolerances(dtype, torch_module)
    bounds_ok = bool((actual >= lower_bound).all().item() and (actual <= 0).all().item())
    return _case(
        label,
        {"shape": [2, 5, heads, head_dim], "input_dtype": str(dtype), "lower_bound": lower_bound},
        _compare(
            torch_module,
            context,
            "gate_vs_scalar_oracle",
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            expected_dtype=torch_module.float32,
        ),
        _boolean_check("bounded_range", bounds_ok, {"minimum": lower_bound, "maximum": 0}),
    )


def _case_recurrent(runtime: dict[str, object], context: dict[str, object], dtype: object, label: str):
    torch_module = runtime["torch"]
    kda = runtime["kda"]
    seed = 200 if dtype == torch_module.float32 else 201
    q, k, v, g, beta = _inputs(torch_module, 2, 7, 2, 4, seed, dtype)
    initial_state = torch_module.randn(2, 2, 4, 4, generator=torch_module.Generator().manual_seed(seed + 1)) * 0.15
    expected_out, expected_state = _oracle_batch(torch_module, q, k, v, g, beta, initial_state, scale=None)
    device_state = _move(torch_module, context, initial_state.clone())
    actual_out, actual_state = kda.hpu_fused_recurrent_kda(
        _move(torch_module, context, q),
        _move(torch_module, context, k),
        _move(torch_module, context, v),
        _move(torch_module, context, g),
        _move(torch_module, context, beta),
        initial_state=device_state,
        inplace_final_state=False,
    )
    state_unchanged = _compare(
        torch_module,
        context,
        "input_state_not_mutated",
        device_state,
        initial_state,
        atol=0.0,
        rtol=0.0,
        expected_dtype=torch_module.float32,
    )
    atol, rtol = _tolerances(dtype, torch_module)
    return _case(
        label,
        {"shape": [2, 7, 2, 4], "input_dtype": str(dtype), "state_dtype": "float32"},
        _compare(torch_module, context, "output_vs_matrix_oracle", actual_out, expected_out, atol=atol, rtol=rtol,
                 expected_dtype=dtype),
        _compare(torch_module, context, "final_state_vs_matrix_oracle", actual_state, expected_state, atol=atol,
                 rtol=rtol, expected_dtype=torch_module.float32),
        state_unchanged,
    )


def _case_chunk(runtime: dict[str, object], context: dict[str, object], dtype: object, label: str):
    torch_module = runtime["torch"]
    kda = runtime["kda"]
    seed = 300 if dtype == torch_module.float32 else 301
    q, k, v, g, beta = _inputs(torch_module, 2, 7, 2, 4, seed, dtype)
    initial_state = torch_module.randn(2, 2, 4, 4, generator=torch_module.Generator().manual_seed(seed + 1)) * 0.15
    expected_out, expected_state = _oracle_batch(torch_module, q, k, v, g, beta, initial_state, scale=None)
    actual_out, actual_state = kda.hpu_chunk_kda(
        _move(torch_module, context, q),
        _move(torch_module, context, k),
        _move(torch_module, context, v),
        _move(torch_module, context, g),
        _move(torch_module, context, beta),
        initial_state=_move(torch_module, context, initial_state.clone()),
        chunk_size=3,
        inplace_final_state=False,
    )
    atol, rtol = _tolerances(dtype, torch_module, chunk=True)
    return _case(
        label,
        {"shape": [2, 7, 2, 4], "input_dtype": str(dtype), "chunk_size": 3, "state_dtype": "float32"},
        _compare(torch_module, context, "output_vs_matrix_oracle", actual_out, expected_out, atol=atol, rtol=rtol,
                 expected_dtype=dtype),
        _compare(torch_module, context, "final_state_vs_matrix_oracle", actual_state, expected_state, atol=atol,
                 rtol=rtol, expected_dtype=torch_module.float32),
    )


def _case_continuity(runtime: dict[str, object], context: dict[str, object], dtype: object, label: str):
    torch_module = runtime["torch"]
    kda = runtime["kda"]
    seed = 400 if dtype == torch_module.float32 else 401
    q, k, v, g, beta = _inputs(torch_module, 1, 9, 2, 3, seed, dtype)
    initial_state = torch_module.randn(1, 2, 3, 3, generator=torch_module.Generator().manual_seed(seed + 1)) * 0.1
    expected_out, expected_state = _oracle_batch(torch_module, q, k, v, g, beta, initial_state, scale=None)
    device_inputs = [_move(torch_module, context, tensor) for tensor in (q, k, v, g, beta)]
    full_out, full_state = kda.hpu_chunk_kda(
        *device_inputs,
        initial_state=_move(torch_module, context, initial_state.clone()),
        chunk_size=3,
        inplace_final_state=False,
    )
    split = 4
    prefix_out, checkpoint = kda.hpu_chunk_kda(
        *(tensor[:, :split] for tensor in device_inputs),
        initial_state=_move(torch_module, context, initial_state.clone()),
        chunk_size=3,
        inplace_final_state=False,
    )
    checkpoint_before = checkpoint.clone()
    suffix_out, suffix_state = kda.hpu_chunk_kda(
        *(tensor[:, split:] for tensor in device_inputs),
        initial_state=checkpoint,
        chunk_size=3,
        inplace_final_state=False,
    )
    joined_out = torch_module.cat((prefix_out, suffix_out), dim=1)
    atol, rtol = _tolerances(dtype, torch_module, chunk=True)
    return _case(
        label,
        {"shape": [1, 9, 2, 3], "input_dtype": str(dtype), "split": split, "chunk_size": 3},
        _compare(torch_module, context, "full_output_vs_matrix_oracle", full_out, expected_out, atol=atol, rtol=rtol,
                 expected_dtype=dtype),
        _compare(torch_module, context, "full_state_vs_matrix_oracle", full_state, expected_state, atol=atol,
                 rtol=rtol, expected_dtype=torch_module.float32),
        _compare(torch_module, context, "split_output_vs_full", joined_out, full_out, atol=atol, rtol=rtol,
                 expected_dtype=dtype),
        _compare(torch_module, context, "split_state_vs_full", suffix_state, full_state, atol=atol, rtol=rtol,
                 expected_dtype=torch_module.float32),
        _compare(torch_module, context, "checkpoint_not_mutated", checkpoint, checkpoint_before, atol=0.0, rtol=0.0,
                 expected_dtype=torch_module.float32),
        _boolean_check("returned_state_is_fresh", suffix_state is not checkpoint),
    )


def _case_model_geometry_recurrent(runtime: dict[str, object], context: dict[str, object], dtype: object,
                                   label: str):
    torch_module = runtime["torch"]
    kda = runtime["kda"]
    seed = 700 if dtype == torch_module.float32 else 701
    seq_len = 9
    q, k, v, g, beta = _inputs(
        torch_module,
        1,
        seq_len,
        GLM_MODEL_GEOMETRY["local_num_heads"],
        GLM_MODEL_GEOMETRY["head_dim"],
        seed,
        dtype,
    )
    initial_state = torch_module.randn(
        1,
        GLM_MODEL_GEOMETRY["local_num_heads"],
        GLM_MODEL_GEOMETRY["head_dim"],
        GLM_MODEL_GEOMETRY["head_dim"],
        generator=torch_module.Generator().manual_seed(seed + 1),
    ) * 0.05
    expected_out, expected_state = _oracle_batch(torch_module, q, k, v, g, beta, initial_state, scale=None)
    device_state = _move(torch_module, context, initial_state.clone())
    actual_out, actual_state = kda.hpu_fused_recurrent_kda(
        _move(torch_module, context, q),
        _move(torch_module, context, k),
        _move(torch_module, context, v),
        _move(torch_module, context, g),
        _move(torch_module, context, beta),
        initial_state=device_state,
        inplace_final_state=False,
    )
    atol, rtol = _tolerances(dtype, torch_module)
    return _case(
        label,
        {
            "shape": [1, seq_len, GLM_MODEL_GEOMETRY["local_num_heads"], GLM_MODEL_GEOMETRY["head_dim"]],
            "input_dtype": str(dtype),
            "state_dtype": "float32",
            **GLM_MODEL_GEOMETRY,
        },
        _compare(torch_module, context, "output_vs_matrix_oracle", actual_out, expected_out, atol=atol, rtol=rtol,
                 expected_dtype=dtype),
        _compare(torch_module, context, "final_state_vs_matrix_oracle", actual_state, expected_state, atol=atol,
                 rtol=rtol, expected_dtype=torch_module.float32),
        _compare(torch_module, context, "input_state_not_mutated", device_state, initial_state, atol=0.0,
                 rtol=0.0, expected_dtype=torch_module.float32),
    )


def _case_model_geometry_chunk(runtime: dict[str, object], context: dict[str, object], dtype: object, label: str):
    torch_module = runtime["torch"]
    kda = runtime["kda"]
    seed = 710 if dtype == torch_module.float32 else 711
    seq_len = 9
    chunk_size = 4
    q, k, v, g, beta = _inputs(
        torch_module,
        1,
        seq_len,
        GLM_MODEL_GEOMETRY["local_num_heads"],
        GLM_MODEL_GEOMETRY["head_dim"],
        seed,
        dtype,
    )
    initial_state = torch_module.randn(
        1,
        GLM_MODEL_GEOMETRY["local_num_heads"],
        GLM_MODEL_GEOMETRY["head_dim"],
        GLM_MODEL_GEOMETRY["head_dim"],
        generator=torch_module.Generator().manual_seed(seed + 1),
    ) * 0.05
    expected_out, expected_state = _oracle_batch(torch_module, q, k, v, g, beta, initial_state, scale=None)
    actual_out, actual_state = kda.hpu_chunk_kda(
        _move(torch_module, context, q),
        _move(torch_module, context, k),
        _move(torch_module, context, v),
        _move(torch_module, context, g),
        _move(torch_module, context, beta),
        initial_state=_move(torch_module, context, initial_state.clone()),
        chunk_size=chunk_size,
        inplace_final_state=False,
    )
    atol, rtol = _tolerances(dtype, torch_module, chunk=True)
    return _case(
        label,
        {
            "shape": [1, seq_len, GLM_MODEL_GEOMETRY["local_num_heads"], GLM_MODEL_GEOMETRY["head_dim"]],
            "input_dtype": str(dtype),
            "state_dtype": "float32",
            "chunk_size": chunk_size,
            "chunk_count": (seq_len + chunk_size - 1) // chunk_size,
            "crosses_chunk_boundaries": True,
            **GLM_MODEL_GEOMETRY,
        },
        _compare(torch_module, context, "output_vs_matrix_oracle", actual_out, expected_out, atol=atol, rtol=rtol,
                 expected_dtype=dtype),
        _compare(torch_module, context, "final_state_vs_matrix_oracle", actual_state, expected_state, atol=atol,
                 rtol=rtol, expected_dtype=torch_module.float32),
    )


def _case_padding_prefill(runtime: dict[str, object], context: dict[str, object], dtype: object, label: str):
    torch_module = runtime["torch"]
    kda = runtime["kda"]
    seed = 500 if dtype == torch_module.float32 else 501
    q, k, v, g, beta = _inputs(torch_module, 1, 12, 2, 3, seed, dtype)
    bucket_len = 4
    valid_lengths = (4, 2, 4)
    for sequence_index, valid_length in enumerate(valid_lengths):
        start = sequence_index * bucket_len + valid_length
        end = (sequence_index + 1) * bucket_len
        q[0, start:end] = 0
        k[0, start:end] = 0
        v[0, start:end] = 0
        g[0, start:end] = 0
        beta[0, start:end] = 0
    initial_state = torch_module.randn(
        3, 2, 3, 3, generator=torch_module.Generator().manual_seed(seed + 1)
    ) * 0.1
    expected_out = torch_module.zeros(1, 12, 2, 3, dtype=torch_module.float32)
    expected_states = []
    for sequence_index, valid_length in enumerate(valid_lengths):
        start = sequence_index * bucket_len
        end = start + valid_length
        output_i, state_i = _oracle_sequence(
            torch_module,
            q[0, start:end],
            k[0, start:end],
            v[0, start:end],
            g[0, start:end],
            beta[0, start:end],
            initial_state[sequence_index],
            scale=None,
        )
        expected_out[0, start:end] = output_i
        expected_states.append(state_i)
    expected_state = torch_module.stack(expected_states)
    actual_out, actual_state = kda.hpu_chunk_kda(
        _move(torch_module, context, q),
        _move(torch_module, context, k),
        _move(torch_module, context, v),
        _move(torch_module, context, g),
        _move(torch_module, context, beta),
        initial_state=_move(torch_module, context, initial_state.clone()),
        prefill_num_seqs=3,
        prefill_seq_len=bucket_len,
        chunk_size=3,
        inplace_final_state=False,
    )
    padding_slices = (slice(6, 8),)
    padding_zero = all(bool(torch_module.count_nonzero(actual_out[:, item]).item() == 0) for item in padding_slices)
    atol, rtol = _tolerances(dtype, torch_module, chunk=True)
    return _case(
        label,
        {"bucket_shape": [1, 12, 2, 3], "prefill_num_seqs": 3, "prefill_seq_len": bucket_len,
         "valid_lengths": list(valid_lengths), "input_dtype": str(dtype)},
        _compare(torch_module, context, "padded_output_vs_sequence_oracle", actual_out, expected_out, atol=atol,
                 rtol=rtol, expected_dtype=dtype),
        _compare(torch_module, context, "padded_states_vs_sequence_oracle", actual_state, expected_state, atol=atol,
                 rtol=rtol, expected_dtype=torch_module.float32),
        _boolean_check("padding_output_is_zero", padding_zero),
    )


def _case_padding_lanes(runtime: dict[str, object], context: dict[str, object], dtype: object, label: str):
    torch_module = runtime["torch"]
    kda = runtime["kda"]
    seed = 600 if dtype == torch_module.float32 else 601
    q, k, v, g, beta = _inputs(torch_module, 1, 4, 2, 3, seed, dtype)
    indices = torch_module.tensor([0, -1, 1, -1], dtype=torch_module.long)
    state = torch_module.randn(3, 2, 3, 3, generator=torch_module.Generator().manual_seed(seed + 1)) * 0.1
    state_before = state.clone()
    expected_outputs = []
    for lane, slot in enumerate(indices.tolist()):
        lane_state = state[slot] if 0 <= slot < state.shape[0] else None
        output_i, _ = _oracle_sequence(
            torch_module,
            q[0, lane : lane + 1],
            k[0, lane : lane + 1],
            v[0, lane : lane + 1],
            g[0, lane : lane + 1],
            beta[0, lane : lane + 1],
            lane_state,
            scale=None,
        )
        expected_outputs.append(output_i)
    expected_out = torch_module.stack(expected_outputs, dim=1)
    actual_out, actual_state = kda.hpu_fused_recurrent_kda(
        _move(torch_module, context, q),
        _move(torch_module, context, k),
        _move(torch_module, context, v),
        _move(torch_module, context, g),
        _move(torch_module, context, beta),
        initial_state=_move(torch_module, context, state.clone()),
        ssm_state_indices=_move(torch_module, context, indices),
    )
    atol, rtol = _tolerances(dtype, torch_module)
    untouched_slot = _compare(
        torch_module,
        context,
        "unmapped_slot_not_mutated",
        actual_state[2],
        state_before[2],
        atol=0.0,
        rtol=0.0,
        expected_dtype=torch_module.float32,
    )
    return _case(
        label,
        {"shape": [1, 4, 2, 3], "state_slots": 3, "indices": indices.tolist(), "input_dtype": str(dtype),
         "padding_rule": "negative or out-of-range lanes use zero state and are never written"},
        _compare(torch_module, context, "padded_lane_output_vs_zero_state_oracle", actual_out, expected_out, atol=atol,
                 rtol=rtol, expected_dtype=dtype),
        untouched_slot,
    )


def _run_scenarios(runtime: dict[str, object], seed: int, device_mode: str) -> list[dict[str, object]]:
    del seed  # Case-local generators make each case independent and reproducible.
    torch_module = runtime["torch"]
    context: dict[str, object] = {"device_mode": device_mode, "hpu_tensors_created": False}
    cases = []
    builders = (
        ("gate", _case_gate),
        ("recurrent", _case_recurrent),
        ("chunk", _case_chunk),
        ("continuity", _case_continuity),
        ("padding_prefill", _case_padding_prefill),
        ("padding_lanes", _case_padding_lanes),
        ("model_geometry_recurrent", _case_model_geometry_recurrent),
        ("model_geometry_chunk", _case_model_geometry_chunk),
    )
    with torch_module.no_grad():
        for dtype_label, dtype in (("fp32", torch_module.float32), ("bf16", torch_module.bfloat16)):
            for case_name, builder in builders:
                label = f"{case_name}_{dtype_label}"
                try:
                    cases.append(builder(runtime, context, dtype, label))
                except Exception as exc:
                    cases.append(
                        {
                            "name": label,
                            "ok": False,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    )
        _sync(torch_module, device_mode)
    runtime["hpu_tensors_created"] = bool(context["hpu_tensors_created"])
    return cases


def _base_report(args: argparse.Namespace) -> dict[str, object]:
    started = _utc_now()
    return {
        "schema": SCHEMA,
        "status": "running",
        "started_at_utc": started,
        "finished_at_utc": None,
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "scope": {
            "operators": ["hpu_kda_gate", "hpu_fused_recurrent_kda", "hpu_chunk_kda"],
            "dtypes": ["float32", "bfloat16"],
            "state_dtype": "float32",
            "model": "not loaded",
            "model_geometry": GLM_MODEL_GEOMETRY,
            "cases": [
                "independent CPU oracle",
                "chunk continuity",
                "prefill padding",
                "decode padding lanes",
                "GLM TP8 local-shard recurrent geometry",
                "GLM TP8 local-shard chunk geometry with cross-chunk carry",
            ],
        },
        "device_mode": args.device,
        "hpu": {
            "requested": args.device == "hpu",
            "launched": False,
            "hpu_tensors_created": False,
            "single_card_only": True,
            "target_index": args.device_id if args.device == "hpu" else None,
            "occupancy_precheck": None,
            "occupancy_preallocation": None,
            "runtime_pin": None,
        },
        "provenance": _static_provenance(args),
        "cases": [],
        "limitations": [
            "This is an operator smoke test, not a full-model or service validation.",
            "No checkpoint, tokenizer, processor, engine, or model weights are loaded.",
            "Chunk execution uses the portable PyTorch implementation; no recipe-cache or throughput claim is made.",
            "Geometry cases exercise one TP8 local shard (8 of 64 heads) without tensor-parallel collectives.",
        ],
    }


def _write_report(report: dict[str, object], output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finish_report(report: dict[str, object], output: Path | None, quiet: bool) -> int:
    report["finished_at_utc"] = _utc_now()
    _write_report(report, output)
    if not quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return int(report.get("exit_code", EXIT_FAIL))


def _run(args: argparse.Namespace) -> dict[str, object]:
    report = _base_report(args)
    if args.device == "hpu":
        try:
            precheck = _require_idle_target(args.device_id, args.idle_memory_limit_mib)
        except DeviceBusyError as exc:
            evidence = _collect_hl_smi_evidence(args.device_id, args.idle_memory_limit_mib)
            report["hpu"]["occupancy_precheck"] = evidence
            report["status"] = "blocked"
            report["exit_code"] = EXIT_BUSY
            report["error"] = f"HPU not launched: {exc}"
            report["limitations"].append("The selected card was occupied or could not be proven idle; no HPU allocation occurred.")
            return report
        report["hpu"]["occupancy_precheck"] = precheck
        _prepare_environment("hpu")
        module_id = int(precheck["target"]["module_id"])
        _pin_hpu(module_id)
        report["hpu"]["runtime_pin"] = {
            "HABANA_VISIBLE_MODULES": str(module_id),
            "HLS_MODULE_ID": str(module_id),
            "PT_HPU_LAZY_MODE": "0",
            "PT_HPU_ENABLE_EAGER_CACHE": "0",
        }
    else:
        _prepare_environment("cpu")

    runtime = _prepare_runtime(args.seed)
    report["provenance"]["runtime"] = runtime["details"]

    if args.device == "hpu":
        try:
            preallocation = _require_idle_target(args.device_id, args.idle_memory_limit_mib)
        except DeviceBusyError as exc:
            evidence = _collect_hl_smi_evidence(args.device_id, args.idle_memory_limit_mib)
            report["hpu"]["occupancy_preallocation"] = evidence
            report["status"] = "blocked"
            report["exit_code"] = EXIT_BUSY
            report["error"] = f"HPU not launched after post-import recheck: {exc}"
            report["limitations"].append("The card became occupied before the first HPU tensor allocation; no tensor was moved to HPU.")
            return report
        report["hpu"]["occupancy_preallocation"] = preallocation

    cases = _run_scenarios(runtime, args.seed, args.device)
    report["cases"] = cases
    report["hpu"]["hpu_tensors_created"] = bool(runtime.get("hpu_tensors_created", False))
    report["hpu"]["launched"] = bool(report["hpu"]["hpu_tensors_created"])
    report["status"] = "passed" if cases and all(case["ok"] for case in cases) else "failed"
    report["exit_code"] = EXIT_PASS if report["status"] == "passed" else EXIT_FAIL
    if args.device == "cpu":
        report["hpu"]["launched"] = False
        report["hpu"]["hpu_tensors_created"] = False

    _write_report(report, args.output)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 pass, 1 numerical/runtime failure, 2 usage error, 3 HPU busy or occupancy unknown.",
    )
    parser.add_argument("--device", choices=("cpu", "hpu"), default="cpu", help="cpu is the safe default; hpu requires an idle card")
    parser.add_argument("--device-id", type=int, default=0, help="AIP index from hl-smi (mapped to module_id before pinning)")
    parser.add_argument("--idle-memory-limit-mib", type=float, default=1024.0,
                        help="Reject the target when hl-smi reports more used memory than this")
    parser.add_argument("--seed", type=int, default=20261024, help="Global seed for runtime setup; cases use fixed seeds")
    parser.add_argument("--output", type=Path, help="Write the provenance and numerical-error JSON report here")
    parser.add_argument("--quiet", action="store_true", help="Do not print the JSON report to stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = _run(args)
    except Exception as exc:
        report = _base_report(args)
        report["status"] = "failed"
        report["exit_code"] = EXIT_FAIL
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    if report["status"] == "blocked":
        print(report.get("error", "HPU occupancy check failed"), file=sys.stderr)
    return _finish_report(report, args.output, args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
