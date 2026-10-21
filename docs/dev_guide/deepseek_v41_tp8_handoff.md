# TP8 DSpark development handoff

Updated: 2026-10-21. This is a source/development handoff, not evidence of a
successful Gaudi2 deployment. TP8 × PP1 DSpark still requires
`max_num_seqs=1`; no batched DSpark serving switch is available.

## Source pins

The plugin branch is `feat/tp8-unified-runtime-build`. Its unified runtime
builder was introduced in `894ba9bf1f1ad61008c9700fe822d747399b9903`.
The subsequent metadata commit is a CPU preparation step, not a runner change.
Use the checked-out branch's full SHA when recording the deployment.

| Repository in the AAAkater forks | Ref | Required source pin |
| --- | --- | --- |
| `1Cat-vLLM-Gaudi` | `feat/tp8-unified-runtime-build` | Record the full HEAD containing this handoff |
| [`vllm`](https://github.com/AAAkater/vllm) | `feat/dsv41-serving-engine` | `64e9c1cb9b045b53b4feb174389c1f288e375e38` (base `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`) |
| `HCL` | `feat/tp8-native-hcl` | `af8e5be897b13fb4dc9a3caac0daf98b81cea376` |
| `Intel_Gaudi3_Software` | `feat/native-runtime-source` | `e64079d1c05be2edb969f9781c9a519b773f47d3` |
| `gaudi-pytorch-bridge` | `import/tp8-patches` | `d92427b592b97fa9322556fca54cfb58fefb5130` |

The Bridge patch branch has been published to the user's fork. Fetch that ref
and verify its SHA; the ordinary `v1.24.2` checkout is **not** the patched
Bridge. Publishing does not rewrite the existing patch authors or commit IDs.

The serving engine is the already-patched
[AAAkater/vllm](https://github.com/AAAkater/vllm) branch
`feat/dsv41-serving-engine`. Its published commit reuses the existing 18-file
patched source tree exactly; it was not regenerated from a patch. It is based on
`e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`, not the development checkout's local
HEAD `fa75d3456d60a5ffe923818c4b495295675b38e1`, and has not been
hardware-validated as a deployment. Use a new checkout:

```bash
NEWDIR=../vllm-dsv41
git clone --branch feat/dsv41-serving-engine \
  https://github.com/AAAkater/vllm.git "$NEWDIR"
test "$(git -C "$NEWDIR" rev-parse HEAD)" = \
  64e9c1cb9b045b53b4feb174389c1f288e375e38
test -z "$(git -C "$NEWDIR" status --porcelain)"
```

Do not run `tools/prepare_deepseek_v41_engine.py` and do not reapply an engine
patch. The old `dsv41-serving-engine.patch` and `.json` files in this repository
remain for historical audit, legacy workflows and existing source-integrity
checks. They are not the source download or patch-application step for this
branch; the optional installer may still read the JSON to verify source hashes.

## What is implemented

- [DSparkBatchMetadata](../../vllm_gaudi/ops/deepseek_v41_batch_speculation.py)
  constructs host-side request/slot ownership, cumulative query and draft
  lengths, flattened proposals, budgets, padding masks and logits indices.
- Raw target row indices (`logits_indices`) are distinct from indices into
  the selected logits (`target_logits_indices`, `bonus_logits_indices`).
- The module validates host inputs without implicit device-to-host copies.
  Its greedy accepted-prefix helpers are CPU reference oracles, not captured
  graph operators or device commit implementations.
- Metadata accepts up to eight requests, but this does **not** enable B2/B8
  execution. No runner imports this new module yet; existing admission remains.
- [CPU tests](../../tests/standalone/deepseek_v41/test_batch_speculation.py)
  cover mixed proposal counts, index spaces, padding, partial acceptance,
  empty drafts and invalid/aliased request owners.

Target verification normally consumes six rows per request: an anchor plus
five proposals. The current draft backbone uses five queries (anchor plus
four noise inputs) to generate five proposals. Preserve that distinction.
Within the Markov head, iterate over the five draft steps while vectorizing
across requests; do not serialize all five steps for each request in turn.

## Validation observed on the development host

Run from the plugin repository root using an existing compatible environment:

```bash
PYTHON="/path/to/existing/compatible-venv/bin/python"
"$PYTHON" -m pytest -q tests/standalone/deepseek_v41/test_batch_speculation.py
"$PYTHON" -m pytest -q tests/standalone/deepseek_v41/test_batch_speculation.py \
  tests/standalone/deepseek_v41/test_batch_metadata.py \
  tests/standalone/deepseek_v41/test_bounded_speculative_sampling.py
```

Observed results: six tests in the new file; thirteen tests in the combined
selection. Ruff, Python compilation and whitespace checks also passed.
These tests used CPU tensors only and did not validate native compilation,
HPU execution, TP8 communication, model quality or serving performance.

A broader run involving
[test_speculative_sampling.py](../../tests/standalone/deepseek_v41/test_speculative_sampling.py)
had four failures importing `torchaudio` because its binary did not match the
installed Torch ABI. Do not install random package versions or bypass checks
to hide that environment problem. Running tests from the parent workspace
rather than this repository can also cause `ModuleNotFoundError`.

## Deployment order and gates

1. Record the actual Gaudi2 module IDs, driver/firmware/SDK, Python and Gaudi
   Torch versions, CPU/RAM/storage budgets, permissions and shared device leases.
   Eight TP ranks require eight reserved modules, not eight requests.
2. Check all source pins above. Build patched HCL, then patched Synapse, then
   patched Bridge against the matching SDK/Torch environment. Retain private
   library outputs; never overwrite the shared system SDK as a shortcut.
3. Clone and verify the applied-engine branch as described above, then install
   the engine and plugin in that same environment without an uncontrolled
   dependency upgrade.
4. Use the existing unified runtime builder described in
   [DeepSeek V4.1 preparation](../features/deepseek_v41.md#unified-v41-runtime-build).
   It produces native kernels/host gather and the communication extension in
   one invocation. No extra independent extension build step is needed.
5. Validate the native manifests, aggregate manifest and Bridge ABI sidecar
   against actual library hashes/loading paths. A successful build is not a
   model execution result. Do not reuse another host's build/recipe cache as
   proof of compatibility.
6. Prepare/transfer the TP8 shards, both FP8 sidecars, checkpoint audit and
   Engram sources separately, preserving their hashes and precision contracts.
7. Establish a single-request greedy TP8 DSpark baseline first. Only after
   that baseline exists should the server proceed to B2 implementation.

### No runtime profile: launcher limitation

This route does not require a runtime profile JSON. However,
[run_deepseek_v41.py](../../tools/run_deepseek_v41.py) currently requires
`--runtime-profile`, so that evidence wrapper is **not** a working no-profile
launcher. Do not fabricate a profile or remove validation to make it start.

The [dedicated entrypoint](../../vllm_gaudi/entrypoints/deepseek_v41.py) accepts
an optional profile, but its default user settings may select one implicitly.
A direct-entry candidate must explicitly select an isolated settings file
without a `runtime_profile` key, preserve verified native-library environment
settings, and independently reserve modules and collect logs/evidence.
It has not been hardware-validated here and is not an equivalent replacement
for the wrapper's locking, NUMA and evidence behavior. Keep V2 and ordinary
`BATCH_DECODE` off, PP1, `max_num_seqs=1`, and prefix caching off.

## Next implementation milestones

- Batched target C6 and draft five-query execution with per-request attention,
  page tables, SWA/compressor/draft-context/Engram ownership.
- Batched verify/commit tickets bound to request slot generation and round
  generation, with independent accepted lengths and stale-result rejection.
- Explicit restoration/isolation of rejected speculative state; target
  verification cannot know the rejected suffix before it computes it.
- B2 fixed-address HPU graphs, padding no-write and bounded workspaces.
- Integration with the existing scheduler's request mapping and rollback.
  Do not apply rejection rollback twice to the scheduler's authoritative counters.
- B4/B8 and dynamic admission only after B2 correctness and hardware gates.
  Prefix reuse, sampled verification, multiple in-flight batches and PP2
  require separate work; none is implied by this metadata commit.

## Transfer reminders

This committed handoff and the metadata/tests travel with the plugin branch.
Full workspace-level progress/deployment documents and the offline diagrams
are separate files outside this Git repository and must be copied separately
if needed. Their previous local-only/Bridge-bundle warnings are snapshots;
recheck current remote refs and the actual plugin HEAD before transfer.

Git bundles do not include uncommitted changes, untracked files, models or
runtime artifacts. Do not transfer `.venv`, `__pycache__` or old compiled
caches as a working environment. Preserve local documentation changes before
switching checkouts, and do not force-push, reset or clean to resolve drift.
