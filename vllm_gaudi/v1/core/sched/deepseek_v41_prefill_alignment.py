# SPDX-License-Identifier: Apache-2.0
"""Pure host helpers for explicitly selected V4.1 prefill alignment."""
import json
import os
from pathlib import Path
import time


def aligned_prefill_count(*, start, count, prompt_tokens, total_tokens, quantum=128):
    """Round only partial prefill work; preserve decode and the final tail.

    A resumed request can replay accepted output history, hence total_tokens-1.
    The worker still computes exactly every token the scheduler allocates.
    Counts, rather than absolute positions, are aligned: an unusual unaligned
    restore may still need a checkpoint split, but cannot create 127 scalar
    tokens on every subsequent scheduling turn merely due to decode budget.
    """
    if quantum not in (0, 128):
        raise ValueError("V4.1 prefill alignment must be 0 or 128")
    if any(type(x) is not int or x < 0 for x in (start, count, prompt_tokens, total_tokens)):
        raise ValueError("Prefill scheduling coordinates must be nonnegative integers")
    end = max(prompt_tokens, total_tokens - 1)
    if not quantum or not count or start >= end or start + count >= end:
        return count
    return count // quantum * quantum


def write_prefill_trace(kind, record):
    """No device sync/events/tensor reads, and no prompt contents or headers."""
    if os.environ.get("DSV41_PREFILL_ALLOCATION_TRACE") != "1":
        return
    directory = Path(os.environ["DSV41_RUN_EVIDENCE"]) / "prefill-allocation-trace"
    directory.mkdir(exist_ok=True)
    row = dict(record)
    row.update(schema=1, kind=kind, pid=os.getpid(), monotonic_ns=time.monotonic_ns())
    # Separate PID files: engine allocation and each worker's actual chunks.
    with (directory / f"pid{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_scheduler_init(record):
    """Immutable process identity; source-only read and a new evidence file."""
    run = Path(os.environ["DSV41_RUN_EVIDENCE"]).resolve()
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    row = dict(schema=1, event="scheduler_init", engine_pid=os.getpid(),
               engine_start_ticks=int(fields[19]), run_evidence=str(run), **record)
    directory = run / "audit"
    directory.mkdir(exist_ok=True)
    with (directory / "scheduler-init.json").open("x") as stream:
        stream.write(json.dumps(row, indent=2) + "\n")
    write_prefill_trace("scheduler_init", row)
