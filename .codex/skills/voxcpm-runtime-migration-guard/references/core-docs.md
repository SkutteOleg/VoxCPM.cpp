# Core Docs

Use this reference to decide which repository documents to load for the current VoxCPM migration task.

## Always start here

- `VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md`
  - Repository-level runtime plan.
  - Read for any planning, migration, optimization, or review task.

## Contract and runtime principles

- `docs/torch_to_ggml_migration_guide_zh.md`
  - Use for generic Torch -> GGML rules.
  - Focus areas: contract-first migration, weight ownership, memory separation, graph reuse, host/backend boundary.
  - Search patterns: `Contract`, `Model Loader`, `Memory`, `Graph / Scheduler`, `tensor_get -> std::vector -> tensor_set`.

## VoxCPM target architecture

- `docs/voxcpm_torch_to_ggml_complete_refactor_cookbook_zh.md`
  - Use for the target end-state of VoxCPM runtime.
  - Focus areas: `DecodeState`, `Output Buffer`, `Persistent State`, `Graph Cache`, `prefill`, `decode_step`, migration order.
  - Search patterns: `WeightStore`, `Backend`, `Output Buffer`, `Persistent State`, `Graph Cache`, `阶段一`, `阶段二`, `阶段三`.

## Current implemented constraints

- `docs/voxcpm_shared_weight_store_refactor.md`
  - Use when touching model loading, shared ownership, or memory baseline assumptions.
  - Confirms the repository already depends on a single shared `WeightStore` pattern.

- `docs/voxcpm_decode_refactor_summary_zh.md`
  - Use when touching the current decode path or benchmarking current hot-path structure.
  - Helps distinguish conservative existing optimizations from the planned full re-architecture.

- `docs/voxcpm_cpp_backend.md`
  - Use when touching backend initialization, Vulkan/CPU semantics, or deciding whether scheduler work is justified yet.

## Current code hotspots

Read these files before changing hot-path boundaries:

- `src/voxcpm.cpp`
- `include/voxcpm/voxcpm.h`
- `include/voxcpm/backend.h`
- `src/backend.cpp`
- `src/weight-store.cpp`

## Fast grep patterns

Use these patterns to find likely boundary problems quickly:

```bash
rg -n "tensor_get\(|tensor_set\(|std::vector<float>|load_from_gguf\(|reserve_compute_memory|transfer_stats" src include tests docs
```

```bash
rg -n "WeightStore|Output Buffer|Persistent State|Graph Cache|Host Fallback|BufferUsage" docs/voxcpm_torch_to_ggml_complete_refactor_cookbook_zh.md docs/torch_to_ggml_migration_guide_zh.md
```
