#!/usr/bin/env bash
set -euo pipefail

if ! command -v rg >/dev/null 2>&1; then
  echo "[error] rg is required for this audit script." >&2
  exit 1
fi

repo_root="${1:-$(pwd)}"
cd "$repo_root"

count_matches() {
  local pattern="$1"
  shift
  rg -n "$pattern" "$@" 2>/dev/null | wc -l | tr -d ' '
}

has_match() {
  local pattern="$1"
  shift
  if rg -q "$pattern" "$@" 2>/dev/null; then
    echo "yes"
  else
    echo "no"
  fi
}

echo "== VoxCPM Runtime Boundary Audit =="
echo "repo: $repo_root"

echo
echo "[1] BufferUsage coverage"
rg -n "enum class BufferUsage|Weights|KVCache|State|Output|Compute" include/voxcpm/backend.h src/backend.cpp || true
printf 'has State: %s\n' "$(has_match 'State' include/voxcpm/backend.h src/backend.cpp)"
printf 'has Output: %s\n' "$(has_match 'Output' include/voxcpm/backend.h src/backend.cpp)"

echo
echo "[2] Shared weight loading entry points"
rg -n "load_from_gguf\(|load_from_store\(|ggml_backend_alloc_ctx_tensors|alloc_ctx_tensors" src include || true

echo
echo "[3] Hot-path host/device calls in legacy runtime"
rg -n "tensor_get\(|tensor_set\(|std::vector<float>" src/voxcpm.cpp include/voxcpm/voxcpm.h || true

echo
echo "[4] Transfer-stat instrumentation"
rg -n "transfer_stats|host_to_device_bytes|device_to_host_bytes|device_to_device_bytes" src include || true

echo
echo "[5] Summary counts"
printf 'tensor_set in src/voxcpm.cpp: %s\n' "$(count_matches 'tensor_set\(' src/voxcpm.cpp)"
printf 'tensor_get in src/voxcpm.cpp: %s\n' "$(count_matches 'tensor_get\(' src/voxcpm.cpp)"
printf 'std::vector<float> in src/voxcpm.cpp + voxcpm.h: %s\n' "$(count_matches 'std::vector<float>' src/voxcpm.cpp include/voxcpm/voxcpm.h)"
printf 'load_from_gguf in src/include: %s\n' "$(count_matches 'load_from_gguf\(' src include)"
printf 'load_from_store in src/include: %s\n' "$(count_matches 'load_from_store\(' src include)"

echo
echo "[6] Quick interpretation"
state_present="$(has_match 'State' include/voxcpm/backend.h src/backend.cpp)"
output_present="$(has_match 'Output' include/voxcpm/backend.h src/backend.cpp)"
legacy_tget="$(count_matches 'tensor_get\(' src/voxcpm.cpp)"
legacy_tset="$(count_matches 'tensor_set\(' src/voxcpm.cpp)"

if [[ "$state_present" == "no" || "$output_present" == "no" ]]; then
  echo "- backend buffer taxonomy is not yet at the target runtime shape."
else
  echo "- backend buffer taxonomy includes State and Output."
fi

if [[ "$legacy_tget" -gt 0 || "$legacy_tset" -gt 0 ]]; then
  echo "- legacy runtime still contains host/device boundary traffic in src/voxcpm.cpp; treat new additions here as suspicious unless bridging." 
else
  echo "- no direct tensor_get/tensor_set calls were found in src/voxcpm.cpp."
fi

echo "- compare this report before and after runtime changes; counts alone are not proof, but they help catch direction drift."
