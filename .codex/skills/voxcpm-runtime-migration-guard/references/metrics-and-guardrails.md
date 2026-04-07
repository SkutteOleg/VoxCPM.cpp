# Metrics And Guardrails

This reference defines what counts as success for VoxCPM runtime migration work.

## Primary outcomes

Every meaningful runtime refactor should improve or protect at least one of these:

1. lower peak or steady-state memory
2. lower Host/Device transfer volume
3. clearer ownership and lifetime boundaries
4. less graph rebuild or less repeated copying

## Memory rules

Track these whenever runtime ownership changes:

- model-load RSS
- prefill peak RSS
- multi-step decode peak RSS
- multi-step decode steady-state RSS drift

Treat these as regressions unless explained in writing:

- reintroducing more than one full weight buffer
- new persistent host copies of hidden states or patches
- decode RSS growing roughly linearly with step count
- long-lived outputs living inside compute allocations

## Transfer rules

Use backend transfer stats when available.

Track at least:

- `host_to_device_bytes`
- `device_to_host_bytes`
- `device_to_device_bytes`
- per-stage deltas around decode hot-path segments

Red flags:

- `tensor_get -> std::vector<float> -> tensor_set` in decode or prefill
- module boundaries that shuttle full hidden or patch tensors through host memory
- optimizations that reduce compute time but increase transfer volume without justification
- contract text that treats project choices such as `no_alloc=true` as if they were universal GGML rules
- patches to `mul_mat`, attention, conv, or reshape/view/permute paths that do not document operator-specific layout or transpose semantics

## Current repository hotspot

The current legacy runtime still shows many host/device round-trips in `src/voxcpm.cpp`.
That file is the main place to audit whenever a patch claims memory or transfer improvements.

## Review checklist

Ask these questions in order:

1. Did the patch preserve a single shared `WeightStore`?
2. Did the patch keep `KV / State / Output / Compute` separate?
3. Did the patch reduce or at least avoid adding host-vector bridges on the hot path?
4. Did the patch keep module migration aligned with the repository plan order?
5. Did the patch include tests, traces, or measurements that support its claim?

## When host fallback is acceptable

Host fallback is acceptable only when all of these are true:

- the result must survive across graph executions
- there is not yet a safe backend-resident binding path
- the fallback is explicit and isolated as staging or cross-state
- the change records a follow-up direction to remove it later

Host fallback is not acceptable when it silently becomes the default module boundary.

## Practical target

Use the latest reproducible benchmark or trace run as baseline.

Target direction:

- early migration phases: no increase in transfer volume
- after `State` and `Output` land: remove large-tensor host round-trips from decode hot path
- later optimization phases: drive total hot-path H2D + D2H clearly downward, ideally by 30% to 50% versus baseline on the same model/backend/input

Treat this percentage as an engineering target, not a universal law. Backend maturity and staging constraints can limit short-term gains, but those tradeoffs must be stated explicitly.
