# SPDX-License-Identifier: Apache-2.0
"""Eight-rank queued collectives, sampler bytes and compiled native replay gate."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replays', type=int, default=96)
    p.add_argument('--lifecycle', action='store_true', help='Retire the owner generation and verify fresh native capture')
    args = p.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    modules = os.environ['HABANA_VISIBLE_MODULES'].split(',')
    if len(modules) != 8 or int(os.environ['WORLD_SIZE']) != 8:
        raise ValueError('This gate requires exactly eight HPU ranks')
    os.environ['HLS_MODULE_ID'] = modules[rank]
    os.environ.setdefault('VLLM_HPU_DSV41_GRAPH_REPLAY', '1')
    os.environ.setdefault('VLLM_HPU_DSV41_BATCH_DECODE', '1')
    os.environ.setdefault('PT_HPU_ENABLE_LAZY_COLLECTIVES', '0')
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment()
    import torch
    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                  destroy_model_parallel, destroy_distributed_environment)
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime, _resolve_runtime
    from vllm_gaudi.compilation.deepseek_v41_prepared import PreparedTP4Group
    from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats, shutdown_prepared_group_plans
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    bind_worker_cpu(rank)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'rank': rank, 'world_size': 8, 'status': 'running', 'queued': [], 'replay': []}
    path = args.output/f'rank{rank}.json'
    try:
        init_distributed_environment(world_size=8, rank=rank, distributed_init_method='env://', local_rank=rank,
                                     backend='hccl')
        config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=8, pipeline_parallel_size=1))
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=8, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            bridge, backend, _ = _resolve_runtime()
            assert backend.size() == 8 and bridge.native_tp8_decode_graph_available()
            # Rank-specific exactly representable inputs catch missing or duplicated peers.
            for width in (128, 5120, 32768, 40960):
                cpu = (torch.arange(width).remainder(7).float()/8 + rank).bfloat16().reshape(1, -1)
                value = cpu.to('hpu')
                gather = torch.ops.vllm_gaudi.tp_peer_allgather(value, 8).cpu().reshape(8, width)
                expected = torch.cat([(cpu.float()-rank+r).bfloat16() for r in range(8)])
                torch.testing.assert_close(gather, expected, rtol=0, atol=0)
                reduced = bridge.tp4_allreduce_current_stream(backend, value, torch.empty_like(value)).cpu()
                wanted = expected.float().sum(0, keepdim=True).bfloat16()
                torch.testing.assert_close(reduced, wanted, rtol=0, atol=0)
                report['queued'].append({'width': width, 'gather_exact': True, 'reduce_exact_fixture': True})
            _, native_gather = stage_collectives(rank, True, 8, native_fp32_gather=True, ordered_peer_sum=False)
            bits = torch.tensor([[0x3f800000 + rank, 100000 + rank]], dtype=torch.int32)
            actual = native_gather(bits.view(torch.float32).to('hpu'), 1).cpu().view(torch.int32)
            expected = torch.cat([torch.tensor([[0x3f800000+r, 100000+r]], dtype=torch.int32) for r in range(8)], 1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            report['sampler_fp32_bytes_exact'] = True
            reduce, gather = stage_collectives(rank, False, 8)

            class Chain(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.generation, self.search_length = 1, 32768
                    self.register_buffer('state', torch.zeros(1, 128, dtype=torch.bfloat16, device='hpu'))
                def forward(self, value):
                    reduced = reduce(value * 0.5 + rank / 16)
                    shards = gather(reduced * 0.125 + rank / 8, 1)
                    output = shards.reshape(1, 8, 128).sum(1) * 0.125
                    self.state.copy_(output)
                    return output + 0.0

            import copy
            model = Chain()
            oracle_model = copy.deepcopy(model)
            oracle = torch.compile(oracle_model, backend=make_backend(), fullgraph=True, dynamic=False)
            candidate = PreparedTP4Group(model, model, make_backend(native_group_owner=model))
            # All ordinary-communication oracles finish before native capture.
            values = [(torch.arange(128).remainder(7).float()/32 + (step % 5)/16).bfloat16().reshape(1,128)
                      for step in range(args.replays)]
            expected = [oracle(value.to('hpu')).cpu() for value in values]
            for step, (value, wanted) in enumerate(zip(values, expected, strict=True)):
                actual = candidate(value.to('hpu')).cpu()
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                torch.testing.assert_close(model.state.cpu(), wanted, rtol=0, atol=0)
                report['replay'].append({'step': step, 'exact_same_tp8_oracle': True})
            stats = prepared_group_stats()
            assert stats['native_replays'] >= args.replays - 3, stats
            assert stats['native_collectives'] >= 2, stats
            report.update(status='passed', native=stats,
                          loaded_libraries=sorted({r.split()[-1] for r in Path('/proc/self/maps').read_text().splitlines()
                                                   if '/libhcl.so' in r or '/libSynapse.so' in r}))
            if args.lifecycle:
                retired_state = model.state
                retired_value = retired_state.cpu()
                model.state = torch.zeros_like(retired_state)
                model.generation += 1
                for value, wanted in zip(values[:8], expected[:8], strict=True):
                    actual = candidate(value.to('hpu')).cpu()
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                    torch.testing.assert_close(model.state.cpu(), wanted, rtol=0, atol=0)
                    torch.testing.assert_close(retired_state.cpu(), retired_value, rtol=0, atol=0)
                recreated = prepared_group_stats()
                assert recreated['native_graphs'] == 1, recreated
                assert recreated['native_captures'] == stats['native_captures'] + 1, recreated
                assert recreated['native_replays'] >= 6, recreated
                assert recreated['native_invalidations'].get('variant:tp4_state_generation', 0) >= 1, recreated
                report['lifecycle'] = {'state_rebound': True, 'retired_state_unchanged': True,
                                       'fresh_owner_generation': model.generation, 'native': recreated}
            shutdown_prepared_group_plans()
            assert prepared_group_stats()['native_graphs'] == 0
            report['shutdown_native_graphs'] = 0
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, default=str)+'\n')
        print(json.dumps({'rank': rank, 'status': report['status'], 'report': str(path)}), flush=True)


if __name__ == '__main__':
    main()
