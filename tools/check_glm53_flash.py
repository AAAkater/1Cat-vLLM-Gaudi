#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the CPU-only GLM-5.3-Flash offline preflight and emit a JSON manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local GLM-5.3-Flash model directory")
    parser.add_argument("--tp-size", type=int, default=8, help="Tensor-parallel size for static capacity accounting")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="ModelConfig max length used for config preflight",
    )
    parser.add_argument(
        "--header-sample-limit",
        type=int,
        default=7,
        help="Maximum number of safetensors headers to inspect without reading payloads",
    )
    parser.add_argument("--output", type=Path, help="Write the JSON manifest to this path")
    parser.add_argument("--quiet", action="store_true", help="Do not print the JSON manifest to stdout")
    return parser.parse_args(argv)


def _write_manifest(report: dict, output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # The preflight must never discover the HPU platform or initialize Habana.
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Running a script under ``tools/`` otherwise lets the editable install pick
    # up a different vllm_gaudi checkout with the same module name.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from vllm_gaudi.models.glm5next_preflight import run_preflight

    report = run_preflight(
        args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        header_sample_limit=args.header_sample_limit,
    )
    _write_manifest(report, args.output)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    if report["ok"]:
        return 0
    for issue in report["errors"]:
        print(f"{issue['code']}: {issue['message']}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
