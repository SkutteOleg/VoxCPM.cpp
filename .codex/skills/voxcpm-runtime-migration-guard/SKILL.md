---
name: voxcpm-runtime-migration-guard
description: Guardrails for VoxCPM.cpp Torch-to-GGML runtime refactors, migration planning, and hot-path optimization. Use when changing `src/voxcpm.cpp`, `include/voxcpm/voxcpm.h`, backend, memory, state, output, graph-cache, MiniCPM, LocalEncoder, LocDiT, UnifiedCFM, or AudioVAE paths; when planning module migration order; when reviewing patches for memory usage or Host/Device transfer regressions; or when you need to keep work aligned with the repository runtime refactor plan.
---

# VoxCPM Runtime Migration Guard

Use this skill to keep VoxCPM.cpp migration work aligned with the repository runtime plan instead of drifting back to host-vector orchestration.

## Quick Start

1. Read [`VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md`](../../../VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md).
2. Read the relevant document pointers in [`references/core-docs.md`](./references/core-docs.md).
3. If the task touches runtime, backend, decode, or module boundaries, run:

```bash
./.codex/skills/voxcpm-runtime-migration-guard/scripts/audit-runtime-boundaries.sh
```

4. Classify the task before editing:
   - contract / export
   - runtime skeleton
   - module migration
   - hot-path optimization
   - review / regression check
5. Prefer adding or updating `state`, `output`, `graph-cache`, or new `runtime` layers over extending host-vector helper chains in `src/voxcpm.cpp`.
6. After each meaningful task step, update the progress section in [`VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md`](../../../VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md).

## Non-Negotiables

- Keep a single shared `WeightStore` for the whole model.
- Keep `KV / State / Output / Compute` separate by owner, lifetime, and buffer usage.
- Do not add new hot-path `tensor_get -> std::vector<float> -> tensor_set` chains.
- Do not hide host fallback inside helper functions; isolate it as explicit output or cross-state staging.
- Do not optimize before contract, layout, and state boundaries are clear.
- Treat `ggml_context(no_alloc=true)` as the repository runtime choice, not as a universal limitation of GGML.
- For `mul_mat`, attention, conv, and reshape/view/permute-heavy paths, record operator-specific layout semantics explicitly instead of relying on generic shape rules.

## Workflow

### 1. Build the right context

Always start from the repository plan, then load only the relevant references:

- For overall architecture and migration order, read [`VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md`](../../../VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md).
- For contract and runtime principles, read `docs/torch_to_ggml_migration_guide_zh.md`.
- For VoxCPM-specific target architecture, read `docs/voxcpm_torch_to_ggml_complete_refactor_cookbook_zh.md`.
- For current shared-weight assumptions, read `docs/voxcpm_shared_weight_store_refactor.md`.
- For current decode-path behavior, read `docs/voxcpm_decode_refactor_summary_zh.md`.

Read only what the current task needs.

### 2. Decide whether the task belongs to legacy or new runtime

Default rule:

- If the task is a bug fix needed to keep current code usable, patch the legacy path minimally.
- If the task changes runtime boundaries, memory ownership, decode state, output handling, graph reuse, or host/device traffic, move it toward the new runtime shape.

Do not treat `src/voxcpm.cpp` as the final home for new architecture unless there is no safer bridge.

### 3. Guard the hot path

When editing decode or prefill paths, prefer these boundaries:

- module-to-module: `ggml_tensor *`, view, or persistent state handle
- cross-step: persistent state object
- cross-graph stable result: output pool or cross-state object
- host materialization: only final user-visible output, explicit staging fallback, or small debug/scalar results

If a patch adds a new host vector to bridge two runtime stages, document why the backend-resident route is not safe yet and what change should remove the fallback later.

### 4. Preserve the migration order

Use this order unless the task is a blocking fix:

1. runtime skeleton
2. simple modules: embedding / linear / stop / FSQ
3. backbone: MiniCPM / LocEnc / LocDiT
4. generation chain: UnifiedCFM / prefill / decode_step / AudioVAE
5. optimization: graph reuse / reduced transfers / scheduler / offload

Do not jump to scheduler or backend-specific tuning while `State` and `Output` are still missing or unclear.

### 5. Measure before claiming progress

If the task touches runtime behavior, collect at least one of:

- transfer deltas from backend stats
- peak or steady-state RSS observations
- count reduction in obvious hot-path `tensor_get/tensor_set` chains
- graph cache reuse evidence

Use the audit script for quick structure checks and the existing tests/logging for behavior checks.

## Task-Specific Guidance

### Architecture planning

- Update the repository plan if the task changes boundaries, naming, migration order, or success criteria.
- Prefer changing the plan first, then the code.
- Keep the plan focused on ownership, lifetime, layout contract, and measurable outcomes.
- Keep the `当前任务进度` section current after each task push.
- Do not keep appending descriptive prose to the plan. Prefer updating status tables, checklists, milestone bullets, and a short recent-update line.
- Only modify existing descriptive text outside the progress section when goals, boundaries, ordering, or acceptance criteria actually change.

### Module migration

Before editing a module, write down or verify:

- PyTorch input shape
- GGML runtime shape
- GGUF storage shape
- layout / stride / contiguous assumptions
- operator-specific semantics for key ops such as `mul_mat`, attention, conv, and any path with implicit transpose or view-only reshaping
- whether output should stay backend-resident or be published to an output pool

If any of these are unclear, stop and resolve the contract first.

### Memory or transfer optimization

Prioritize changes in this order:

1. eliminate unnecessary host materialization
2. separate `State` / `Output` from compute tensors
3. reduce repeated graph rebuilds and repeated copies
4. only then consider scheduler/offload or backend-specific repack

### Review work

When reviewing a patch, look for these regressions first:

- new per-module GGUF loading paths
- new uses of compute tensors as long-lived outputs
- new host-vector bridges in decode or prefill
- state objects that still live in graph-local allocations
- optimizations justified without measurement

## Verification

Run the audit script when the task touches runtime structure:

```bash
./.codex/skills/voxcpm-runtime-migration-guard/scripts/audit-runtime-boundaries.sh
```

Then run only the most relevant tests for the touched area. Prefer module-level tests first, then `test_voxcpm`, then example-level end-to-end validation.

If architecture assumptions changed, update [`VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md`](../../../VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md) in the same change.
If the task advanced implementation without changing architecture assumptions, update only the progress section unless a textual correction is truly necessary.

## References

- Load [`references/core-docs.md`](./references/core-docs.md) for document selection guidance.
- Load [`references/metrics-and-guardrails.md`](./references/metrics-and-guardrails.md) for measurement rules, current hotspots, and review checklists.
