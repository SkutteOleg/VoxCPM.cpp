# AGENTS.md

## Repository Instructions

- Treat [`VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md`](./VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md) as the primary runtime migration plan for this repository.
- When a task touches `src/voxcpm.cpp`, `include/voxcpm/voxcpm.h`, backend, memory ownership, decode state, output handling, graph cache, MiniCPM, LocalEncoder, LocDiT, UnifiedCFM, or AudioVAE, use the project skill `$voxcpm-runtime-migration-guard` at [`.codex/skills/voxcpm-runtime-migration-guard/SKILL.md`](./.codex/skills/voxcpm-runtime-migration-guard/SKILL.md).
- Do not launch compile or build commands in parallel. Treat this as a hard rule: only one compile/build command may run at a time.
- Before substantial runtime edits, read the plan and the skill references, then run:

```bash
./.codex/skills/voxcpm-runtime-migration-guard/scripts/audit-runtime-boundaries.sh
```

## Runtime Red Lines

- Keep a single shared `WeightStore`; do not reintroduce per-module GGUF loading.
- Keep `KV / State / Output / Compute` separate by ownership and lifetime.
- Do not add new hot-path `tensor_get -> std::vector<float> -> tensor_set` chains.
- Prefer backend-resident tensors or persistent state handles over host vectors across runtime stage boundaries.
- Do not jump to scheduler/offload optimization before contract, state, and output boundaries are clear.
- Treat `ggml_context(no_alloc=true)` as the repository runtime design choice, not as a universal limitation of GGML.
- For `mul_mat`, attention, conv, and reshape/view/permute-heavy paths, document operator-specific layout, transpose, and contiguous semantics explicitly.

## Success Criteria

- Drive memory usage toward a bounded model of `weights + KV/state/output + compute arena`.
- Reduce Host/Device transfer volume on decode hot paths.
- Keep migration aligned with this order: runtime skeleton, simple modules, backbone, generation chain, then optimization.
- When runtime behavior changes, include tests or measurements that support the change.

## File Priorities

- Prefer adding new runtime layers over extending legacy helper chains in `src/voxcpm.cpp`.
- If architecture assumptions change, update `VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md` in the same change.
