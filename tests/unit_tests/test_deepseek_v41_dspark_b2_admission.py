# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for DSpark concurrent request admission before HPU execution."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.entrypoints.serve.exception_handling.error_response import create_error_response
from vllm.exceptions import VLLMClientError, VLLMValidationError
from vllm.sampling_params import SamplingParams

from vllm_gaudi.platform import HpuPlatform
from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner


@pytest.fixture(autouse=True)
def _isolate_admission_state(monkeypatch):
    monkeypatch.setattr(HpuPlatform, "_v41_dspark_batch_profile", False)
    monkeypatch.setattr(HpuPlatform, "_v41_dspark_b2_profile", False)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREPARED_SHARDS", "1")
    for name in (
            "VLLM_HPU_DSV41_DSPARK",
            "VLLM_HPU_DSV41_DEVICE_VERIFY",
            "VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH",
            "VLLM_HPU_NATIVE_DECODE_GRAPH",
            "QUANT_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)


def _config(*, model_type="deepseek_v41", max_num_seqs=2, tp=8, pp=1, prefix=False, async_scheduling=False):
    from vllm.config import CompilationMode

    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
            is_hybrid=False,
            dtype=torch.bfloat16,
            quantization=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            worker_cls="test.worker",
            distributed_executor_backend=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            async_scheduling=async_scheduling,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=prefix,
            user_specified_block_size=True,
            block_size=128,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            custom_ops=None,
            cudagraph_mode=None,
            cudagraph_capture_sizes=None,
        ),
        speculative_config=None,
        load_config=SimpleNamespace(device=None),
    )


def _record_platform_profile(config):
    with (
            patch.object(HpuPlatform, "set_compile_env_defaults"),
            patch("vllm_gaudi.platform._enable_hpu_v1_dflash2_validation"),
            patch("vllm_gaudi.platform.get_config", return_value=SimpleNamespace(VLLM_CONTIGUOUS_PA=False)),
            patch("vllm_gaudi.ops.deepseek_v41_config.configure"),
    ):
        HpuPlatform.check_and_update_config(config)


def _runner(config):
    runner = object.__new__(V41ModelRunner)
    runner.vllm_config = config
    return runner


@pytest.mark.parametrize("max_num_seqs", (2, 3, 6, 16, 64))
def test_platform_batch_admission_allows_greedy_and_rejects_non_greedy(monkeypatch, max_num_seqs):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_DEVICE_VERIFY", "1")
    _record_platform_profile(_config(max_num_seqs=max_num_seqs))

    HpuPlatform.validate_request(None, SamplingParams(temperature=0, max_tokens=8))
    with pytest.raises(VLLMValidationError, match="greedy-only") as exc_info:
        HpuPlatform.validate_request(None, SamplingParams(max_tokens=8))
    assert isinstance(exc_info.value, VLLMClientError)
    assert create_error_response(exc_info.value).error.code == 400

    with pytest.raises(VLLMValidationError, match="greedy-only"):
        HpuPlatform.validate_request(None, SamplingParams(temperature=1, max_tokens=8))


@pytest.mark.parametrize("max_num_seqs", (2, 3, 6, 8, 16, 64))
def test_worker_batch_admission_matches_platform(monkeypatch, max_num_seqs):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_DEVICE_VERIFY", "1")
    runner = _runner(_config(max_num_seqs=max_num_seqs))

    runner._validate_sampling(SamplingParams(temperature=0, max_tokens=8))
    with pytest.raises(VLLMValidationError, match="greedy-only"):
        runner._validate_sampling(SamplingParams(max_tokens=8))


def test_b1_sampled_admission_keeps_existing_device_verify_behavior(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_DEVICE_VERIFY", "1")
    config = _config(max_num_seqs=1)
    _record_platform_profile(config)

    sampled = SamplingParams(temperature=1, max_tokens=8)
    HpuPlatform.validate_request(None, sampled)
    _runner(config)._validate_sampling(sampled)

    monkeypatch.delenv("VLLM_HPU_DSV41_DEVICE_VERIFY")
    with pytest.raises(VLLMValidationError, match="requires device probability verification"):
        HpuPlatform.validate_request(None, sampled)
    with pytest.raises(VLLMValidationError, match="requires device probability verification"):
        _runner(config)._validate_sampling(sampled)


@pytest.mark.parametrize("max_num_seqs", (2, 3, 6, 16, 64))
def test_profile_reset_from_batch_to_b1_clears_platform_gate(monkeypatch, max_num_seqs):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_DEVICE_VERIFY", "1")
    _record_platform_profile(_config(max_num_seqs=max_num_seqs))
    sampled = SamplingParams(temperature=1, max_tokens=8)
    with pytest.raises(VLLMValidationError, match="greedy-only"):
        HpuPlatform.validate_request(None, sampled)

    _record_platform_profile(_config(max_num_seqs=1))
    HpuPlatform.validate_request(None, sampled)


@pytest.mark.parametrize(("model_type", "dspark"), (("opt", "0"), ("deepseek_v41", "0")))
def test_non_v41_or_dspark_off_is_not_temperature_restricted(monkeypatch, model_type, dspark):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", dspark)
    _record_platform_profile(_config(model_type=model_type, max_num_seqs=1))
    sampled = SamplingParams(temperature=1, max_tokens=8)

    HpuPlatform.validate_request(None, sampled)
    _runner(_config(model_type=model_type, max_num_seqs=1))._validate_sampling(sampled)


@pytest.mark.parametrize("max_num_seqs", (2, 6, 64))
def test_batch_admission_still_rejects_unsupported_penalties(monkeypatch, max_num_seqs):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    _record_platform_profile(_config(max_num_seqs=max_num_seqs))
    penalized = SamplingParams(temperature=0, presence_penalty=1.0, max_tokens=8)

    with pytest.raises(VLLMValidationError, match="penalties"):
        HpuPlatform.validate_request(None, penalized)
    with pytest.raises(VLLMValidationError, match="penalties"):
        _runner(_config(max_num_seqs=max_num_seqs))._validate_sampling(penalized)
