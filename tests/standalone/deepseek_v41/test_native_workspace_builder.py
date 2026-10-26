# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the isolated native workspace-reuse candidate builder."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest

from tools import build_deepseek_v41_native_workspace as builder


BASE = Path(os.environ.get(
    "VLLM_DSV41_NATIVE_BUILD_BASE",
    "/data/project/build/synapse-hccl-wrapper-20261021/synapse_release_build",
))


def synthetic_source(*, diagnostic: bool = False) -> str:
    cap_body = '''    {
        status = synOutOfDeviceMemory;
        m_state = State::INVALID;
        return nullptr;
    }'''
    if diagnostic:
        cap_body = '''    {
        LOG_ERR(SYN_MEM_ALLOC, "Native compute graph workspace cap exceeded");
        status = synOutOfDeviceMemory;
        invalidate(status, "workspace_cap");
        return nullptr;
    }'''
    status_line = "    status         = segment.memory->initialize(globalSize, arcSize, workspaceSize);"
    if not diagnostic:
        status_line = "    status = segment.memory->initialize(globalSize, arcSize, workspaceSize);"
    return f'''#include <limits>

CompiledSegment* NativeComputeGraph::addPendingSegment(
    uint64_t globalSize, uint64_t arcSize, uint64_t workspaceSize)
{{
    constexpr uint64_t maximumWorkspaceBytes = uint64_t(2) << 30;
    if (m_workspaceAllocationBytes > maximumWorkspaceBytes || workspaceSize > maximumWorkspaceBytes ||
        (workspaceSize && workspaceSize + 8191 > maximumWorkspaceBytes - m_workspaceAllocationBytes))
{cap_body}
{status_line}
    if (status != synSuccess)
    {{
        m_state = State::INVALID;
        return nullptr;
    }}
    segment.workspaceAddress = workspaceSize ? segment.memory->workspaceDeviceAddress() : workspaceAddress;
    return &segment;
}}
'''


def test_patch_source_preserves_two_gib_cap_and_adds_ordered_reuse():
    patched = builder.patch_source(synthetic_source())
    assert builder.REUSE_ENV in patched
    assert builder.WORKSPACE_LOG_ENV in patched
    assert "const char* reuseMode = std::getenv" in patched
    assert "const char* reuseLogMode = std::getenv" in patched
    assert "for (const auto& previous : m_segments)" in patched
    assert "allocationSize = 8192;" in patched
    assert "while (allocationSize < workspaceSize) allocationSize <<= 1;" in patched
    assert "allocationSize + 8191 > maximumWorkspaceBytes - m_workspaceAllocationBytes" in patched
    assert "status = segment.memory->initialize(globalSize, arcSize, allocationSize);" in patched
    assert "segment.workspaceAddress = scratchAddress ? scratchAddress" in patched
    assert "if (reuseScratch && workspaceSize && logReuse)" in patched
    assert "maximumWorkspaceBytes = uint64_t(2) << 30" in patched
    assert "workspaceSize + 8191 > maximumWorkspaceBytes - m_workspaceAllocationBytes" not in patched
    assert "#include <cstdlib>" in patched
    assert "#include <cstdio>" in patched


def test_patch_source_accepts_des_cartes_diagnostic_anchor():
    patched = builder.patch_source(synthetic_source(diagnostic=True))
    assert 'LOG_ERR(SYN_MEM_ALLOC, "Native compute graph workspace cap exceeded")' in patched
    assert 'invalidate(status, "workspace_cap")' in patched
    assert "status = segment.memory->initialize(globalSize, arcSize, allocationSize);" in patched
    assert "maximumWorkspaceBytes" in patched


def test_patch_source_rejects_missing_anchor():
    with pytest.raises(ValueError, match="allocation implementation"):
        builder.patch_source("#include <limits>\nint main() {}\n")


def test_reuse_boundary_and_geometric_growth():
    assert builder.next_arena_bytes(0) == 0
    assert builder.next_arena_bytes(1) == 8192
    assert builder.next_arena_bytes(8192) == 8192
    assert builder.next_arena_bytes(8193) == 16384
    assert builder.next_arena_bytes(16384) == 16384

    # NativeRecipeMemory records allocationSize + 8191 bytes.
    assert builder.reusable_arena([8192 + 8191], 8192) == 8192 + 8191
    assert builder.reusable_arena([8192 + 8191], 8193) is None
    assert builder.reusable_arena([8192 + 8191, 16384 + 8191], 16000) == 16384 + 8191
    assert builder.reusable_arena([], 1) is None


def _extract_cpp_method(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"Unclosed method: {signature}")


@pytest.mark.skipif(not (BASE / "compile_commands.json").is_file(), reason="native build base unavailable")
def test_patched_add_pending_segment_executes_with_fake_pool(tmp_path):
    """Compile the actual patched methods against a CPU fake SCAL pool."""
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("No CPU C++ compiler")

    base = BASE.resolve()
    _, _, source_path = builder._load_compile_command(base)
    candidate_source, _ = builder._materialize_source(tmp_path, source_path, "HEAD")
    patched = builder.patch_source(candidate_source.read_text())
    add_method = _extract_cpp_method(patched, "CompiledSegment* NativeComputeGraph::addPendingSegment(")
    close_method = _extract_cpp_method(patched, "synStatus NativeComputeGraph::close()")

    source = tmp_path / "patched_fake_pool.cpp"
    binary = tmp_path / "patched_fake_pool"
    source.write_text(textwrap.dedent(r"""
        #include <cassert>
        #include <cstdint>
        #include <cstdio>
        #include <cstdlib>
        #include <cstring>
        #include <limits>
        #include <memory>
        #include <vector>

        using synStatus = int;
        constexpr synStatus synSuccess = 0;
        constexpr synStatus synInvalidArgument = 1;
        constexpr synStatus synOutOfDeviceMemory = 2;

        struct synLaunchTensorInfoExt {
            uint64_t tensorId = 0;
            uint64_t pTensorAddress = 0;
            int tensorType = 0;
            uint64_t tensorSize[8] = {};
        };
        struct NativeTensorBinding {
            uint64_t tensorId = 0;
            uint64_t deviceAddress = 0;
            int tensorType = 0;
            uint64_t sizes[8] = {};
        };
        struct NativeScalCommandBuffer {};

        struct Block {
            uint64_t address;
            uint64_t bytes;
            bool closed = false;
        };

        class ScalMemoryPool {
        public:
            uint64_t allocate(uint64_t bytes) {
                const uint64_t address = next_address_;
                next_address_ += 0x10000;
                blocks_.push_back({address, bytes, false});
                return address;
            }
            void close(uint64_t address) {
                for (auto& block : blocks_) {
                    if (block.address == address) block.closed = true;
                }
            }
            size_t open_blocks() const {
                size_t result = 0;
                for (const auto& block : blocks_) result += !block.closed;
                return result;
            }
        private:
            uint64_t next_address_ = 0x100000;
            std::vector<Block> blocks_;
        };

        class NativeRecipeMemory {
        public:
            NativeRecipeMemory(ScalMemoryPool& global, ScalMemoryPool& shared)
                : global_(global), shared_(shared) {}
            synStatus initialize(uint64_t globalSize, uint64_t arcSize, uint64_t workspaceSize) {
                if (initialized_) return 3;
                global_bytes_ = globalSize;
                arc_bytes_ = arcSize;
                if (workspaceSize) {
                    workspace_bytes_ = workspaceSize + 8191;
                    workspace_address_ = global_.allocate(workspace_bytes_);
                }
                initialized_ = true;
                return synSuccess;
            }
            bool initialized() const { return initialized_; }
            bool closed() const { return closed_; }
            synStatus close() {
                if (workspace_address_) global_.close(workspace_address_);
                closed_ = true;
                return synSuccess;
            }
            uint64_t workspaceAllocationBytes() const { return workspace_bytes_; }
            uint64_t workspaceDeviceAddress() const { return workspace_address_; }
            uint64_t globalAllocationBytes() const { return global_bytes_; }
            uint64_t arcAllocationBytes() const { return arc_bytes_; }
        private:
            ScalMemoryPool& global_;
            ScalMemoryPool& shared_;
            uint64_t global_bytes_ = 0;
            uint64_t arc_bytes_ = 0;
            uint64_t workspace_bytes_ = 0;
            uint64_t workspace_address_ = 0;
            bool initialized_ = false;
            bool closed_ = false;
        };

        struct CompiledSegment {
            std::unique_ptr<NativeRecipeMemory> memory;
            NativeScalCommandBuffer commands;
            std::vector<NativeTensorBinding> bindings;
            uint64_t workspaceAddress = 0;
            uint64_t recipeSequence = 0;
            bool ready = false;
        };

        class ScalStreamComputeInterface {
        public:
            synStatus longSoWaitForLast(bool, uint64_t, const char*) { return synSuccess; }
        };

        class NativeComputeGraph {
        public:
            enum class State { CREATED, CAPTURING, INSTANTIATED, INVALID, CLOSED };
            NativeComputeGraph(ScalMemoryPool& global, ScalMemoryPool& shared)
                : m_globalPool(global), m_hbmSharedPool(shared) {}
            CompiledSegment* addPendingSegment(uint64_t globalSize, uint64_t arcSize, uint64_t recipeSequence,
                                                const synLaunchTensorInfoExt* launchTensors,
                                                uint32_t launchTensorCount, uint64_t workspaceAddress,
                                                uint64_t workspaceSize, synStatus& status);
            synStatus close();
            State m_state = State::CAPTURING;
            std::vector<CompiledSegment> m_segments;
            ScalMemoryPool& m_globalPool;
            ScalMemoryPool& m_hbmSharedPool;
            ScalStreamComputeInterface m_stream;
            uint64_t m_globalProgramBytes = 0;
            uint64_t m_arcProgramBytes = 0;
            uint64_t m_workspaceAllocationBytes = 0;
        };

    """) + add_method + "\n\n" + close_method + textwrap.dedent(r"""

        int main() {
            synLaunchTensorInfoExt tensor{};
            tensor.tensorSize[0] = 1;

            // Flag off: every segment owns a distinct scratch arena.
            setenv("VLLM_HPU_DSV41_NATIVE_WORKSPACE_REUSE", "0", 1);
            {
                ScalMemoryPool pool;
                NativeComputeGraph graph(pool, pool);
                synStatus status = synSuccess;
                auto* first = graph.addPendingSegment(1, 1, 1, &tensor, 1, 0, 8192, status);
                auto* second = graph.addPendingSegment(1, 1, 2, &tensor, 1, 0, 8192, status);
                assert(first && second && status == synSuccess);
                assert(first->workspaceAddress != second->workspaceAddress);
                assert(graph.m_workspaceAllocationBytes == 2 * (8192 + 8191));
            }

            // Flag on: reuse adequate arenas, grow geometrically, retain old addresses.
            setenv("VLLM_HPU_DSV41_NATIVE_WORKSPACE_REUSE", "1", 1);
            {
                ScalMemoryPool pool;
                NativeComputeGraph graph(pool, pool);
                synStatus status = synSuccess;
                auto* a = graph.addPendingSegment(1, 1, 1, &tensor, 1, 0, 8192, status);
                auto* b = graph.addPendingSegment(1, 1, 2, &tensor, 1, 0, 8192, status);
                auto* c = graph.addPendingSegment(1, 1, 3, &tensor, 1, 0, 8193, status);
                auto* d = graph.addPendingSegment(1, 1, 4, &tensor, 1, 0, 16000, status);
                assert(a && b && c && d && status == synSuccess);
                assert(a->workspaceAddress == b->workspaceAddress);
                assert(c->workspaceAddress != a->workspaceAddress);
                assert(d->workspaceAddress == c->workspaceAddress);
                assert(graph.m_workspaceAllocationBytes == (8192 + 8191) + (16384 + 8191));
                assert(pool.open_blocks() == 2);
                assert(graph.close() == synSuccess);
                assert(pool.open_blocks() == 0);
                assert(graph.m_segments.empty());
                assert(graph.m_state == NativeComputeGraph::State::CLOSED);
            }

            // Flag on: a new geometric arena that would exceed the 2 GiB cap fails.
            {
                ScalMemoryPool pool;
                NativeComputeGraph graph(pool, pool);
                synStatus status = synSuccess;
                assert(graph.addPendingSegment(1, 1, 1, &tensor, 1, 0, 1ULL << 30, status));
                status = synSuccess;
                auto* rejected = graph.addPendingSegment(1, 1, 2, &tensor, 1, 0, (1ULL << 30) + 1, status);
                assert(rejected == nullptr);
                assert(status == synOutOfDeviceMemory);
                assert(graph.m_state == NativeComputeGraph::State::INVALID);
                assert(graph.m_workspaceAllocationBytes == (1ULL << 30) + 8191);
            }
            return 0;
        }
    """))
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    default_run = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
    assert "NATIVE_WORKSPACE_REUSE requested=" not in default_run.stderr

    logged_env = dict(os.environ)
    logged_env[builder.WORKSPACE_LOG_ENV] = "1"
    logged_run = subprocess.run([str(binary)], check=True, text=True, capture_output=True, env=logged_env)
    assert logged_run.stderr.count("NATIVE_WORKSPACE_REUSE requested=") >= 4


@pytest.mark.skipif(not (BASE / "compile_commands.json").is_file(), reason="native build base unavailable")
def test_actual_git_head_patch_compiles(tmp_path):
    base = BASE.resolve()
    compile_cwd, command, source = builder._load_compile_command(base)
    build = tmp_path / "candidate"
    build.mkdir()
    candidate_source, _ = builder._materialize_source(build, source, "HEAD")
    original = candidate_source.read_text()
    candidate_source.write_text(builder.patch_source(original))
    include_root = candidate_source.parents[3]
    object_path = build / "stream_compute_scal.cpp.o"
    compile_command = builder._compile_command(command, source, candidate_source, object_path, include_root)
    subprocess.run(compile_command, cwd=compile_cwd, check=True)
    assert object_path.is_file()


@pytest.mark.skipif(not (BASE / "lib/libSynapse.so").is_file(), reason="native build base unavailable")
def test_base_recipe_uses_compile_commands_and_ninja_not_legacy_commands():
    compile_cwd, command, source = builder._load_compile_command(BASE.resolve())
    assert command
    assert source.name == "stream_compute_scal.cpp"
    link = builder._load_ninja_link(BASE.resolve(), shutil.which("ninja") or "ninja")
    assert "-o" in link
    assert any(value.endswith("stream_compute_scal.cpp.o") for value in link)
